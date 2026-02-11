'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Core deduplication and linkage orchestration: dataset loading, preprocessing, model retrieval/training, clustering, and exports."
'''

from __future__ import annotations

## Standard library imports
import json
from pathlib import Path
from typing import Any, Dict, Tuple

## Third-party imports
import pandas as pd

## Local imports
from src.core.config import RAW_DATA_DIR, ensure_dir, resolve_path
from src.core.errors import DeduplicationError, ValidationError
from src.model.cleaning import preprocess_dataset
from src.model.candidates import build_full_name_candidates, build_address_candidates
from src.model.fuzzy_analysis import get_clusters, get_or_train_dedupe
from src.utils.logging_utils import get_logger
from src.utils.utils import decode_base64_to_file, read_csv_as_dict_and_df

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("model.deduplication")

## ============================================================
## CONSTANTS
## ============================================================
DEFAULT_ID_COLUMN = "id"
DEFAULT_ENCODING = "utf-8"

## NOTE: By default, we keep candidate generation optional
DEFAULT_ENABLE_CANDIDATES = True

## ============================================================
## DATASET LOADING
## ============================================================
def _load_json_records(file_path: Path) -> pd.DataFrame:
    """
        Load a JSON dataset into a DataFrame

        Supported JSON formats:
            - list[dict]
            - {"records": list[dict]}

        Args:
            file_path: Path to a JSON file

        Returns:
            Loaded dataframe

        Raises:
            DeduplicationError: If JSON format is invalid
    """
    
    try:
        payload = json.loads(file_path.read_text(encoding=DEFAULT_ENCODING))
    except Exception as exc:
        raise DeduplicationError(
            message="Failed to read JSON dataset",
            details=str(exc),
        )

    ## Normalize supported formats
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise DeduplicationError(
            message="Invalid JSON dataset format",
            details="Expected list[dict] or {'records': list[dict]}",
        )

    return pd.DataFrame(records)

def load_dataset_from_payload(payload: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
        Load dataset using a request payload

        Supported inputs (best-effort):
            - payload["input_path"]: path to CSV or JSON dataset
            - payload["base64_file"]: base64 content + payload["filename"]
            - payload["raw_dir_filename"]: filename inside data/raw/

        Output:
            - data_d: dict[int, dict[str, Any]] for dedupe
            - df: pandas DataFrame

        Args:
            payload: Request payload

        Returns:
            data_d and df

        Raises:
            ValidationError: If payload does not describe a dataset
            DeduplicationError: If loading fails
    """
    
    ## Ensure raw directory exists
    ensure_dir(RAW_DATA_DIR)

    ## ------------------------------------------------------------
    ## Resolve input file path
    ## ------------------------------------------------------------
    input_path: Path | None = None

    ## Case 1: explicit path
    if payload.get("input_path"):
        input_path = resolve_path(str(payload["input_path"]))

    ## Case 2: base64 file upload
    if input_path is None and payload.get("base64_file") and payload.get("filename"):
        filename = str(payload["filename"]).strip()
        if filename == "":
            raise ValidationError(message="filename must be provided when using base64_file")
        input_path = decode_base64_to_file(
            base64_content=str(payload["base64_file"]),
            output_path=RAW_DATA_DIR / filename,
        )

    ## Case 3: file located in data/raw/
    if input_path is None and payload.get("raw_dir_filename"):
        filename = str(payload["raw_dir_filename"]).strip()
        if filename == "":
            raise ValidationError(message="raw_dir_filename cannot be empty")
        input_path = (RAW_DATA_DIR / filename).resolve()

    if input_path is None:
        raise ValidationError(
            message="No dataset input provided",
            details="Use one of: input_path, base64_file+filename, raw_dir_filename",
        )

    if not input_path.exists():
        raise DeduplicationError(
            message="Dataset file not found",
            details=str(input_path),
        )

    ## ------------------------------------------------------------
    ## Load based on extension
    ## ------------------------------------------------------------
    suffix = input_path.suffix.lower().strip(".")

    if suffix == "csv":
        ## Read as dict + dataframe for dedupe pipeline
        data_d, df = read_csv_as_dict_and_df(
            filename=input_path,
            id_column=payload.get("id_column", DEFAULT_ID_COLUMN),
            limit=payload.get("limit", None),
            cleaning_fn=None,
            encoding=payload.get("encoding", DEFAULT_ENCODING),
        )
        return data_d, df

    if suffix == "json":
        df = _load_json_records(input_path)

        ## Build a dedupe-friendly dict using row index or id_column
        id_col = str(payload.get("id_column", DEFAULT_ID_COLUMN))
        data_d: Dict[int, Dict[str, Any]] = {}

        for idx, row in df.iterrows():
            ## Prefer id column if available
            if id_col in df.columns and pd.notna(row.get(id_col)):
                try:
                    row_id = int(row.get(id_col))
                except Exception:
                    row_id = int(idx) + 1
            else:
                row_id = int(idx) + 1

            data_d[row_id] = row.to_dict()

        return data_d, df

    raise DeduplicationError(
        message="Unsupported dataset format",
        details=f"Supported: .csv, .json. Got: {input_path.suffix}",
    )

## ============================================================
## PREPROCESSING
## ============================================================
def preprocess_for_deduplication(df: pd.DataFrame, enable_candidates: bool = DEFAULT_ENABLE_CANDIDATES) -> pd.DataFrame:
    """
        Preprocess a dataframe prior to dedupe training or clustering

        Steps:
            1) preprocess_dataset: scalar normalization + nested-list parsing + enrichments
            2) Optional: candidate generation (names, addresses)

        Args:
            df: Raw input dataframe
            enable_candidates: Whether to generate all_full_names and all_addresses

        Returns:
            Preprocessed dataframe
    """
    ## 1) Normalize core fields
    df_out = preprocess_dataset(df)

    ## 2) Optional candidate generation
    if enable_candidates:
        ## These may be computationally heavy depending on list sizes
        df_out = build_full_name_candidates(df_out)
        df_out = build_address_candidates(df_out)

    return df_out

def dataframe_to_dedupe_dict(df: pd.DataFrame, id_column: str = DEFAULT_ID_COLUMN) -> Dict[int, Dict[str, Any]]:
    """
        Convert a dataframe to a dedupe-compatible record dictionary

        Args:
            df: Input dataframe
            id_column: Column containing record id

        Returns:
            Dictionary mapping record_id -> record dict
    """
    data_d: Dict[int, Dict[str, Any]] = {}

    for idx, row in df.iterrows():
        ## Prefer explicit id when possible
        if id_column in df.columns and pd.notna(row.get(id_column)):
            try:
                rid = int(row.get(id_column))
            except Exception:
                rid = int(idx) + 1
        else:
            rid = int(idx) + 1

        data_d[rid] = row.to_dict()

    return data_d

## ============================================================
## DEDUPLICATION (CLUSTERING)
## ============================================================
def run_dataset_deduplication(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
        Run dataset deduplication and return per-record cluster membership

        Expected payload keys (best-effort):
            - model_id: int
            - num_processes: int
            - enable_active_learning: bool (interactive, usually False in API)
            - cluster_threshold: float
            - enable_candidates: bool
            - dataset input keys: input_path OR raw_dir_filename OR base64_file+filename

        Args:
            payload: Request payload

        Returns:
            Response-like dict containing cluster_membership
    """
    
    ## ------------------------------------------------------------
    ## Load dataset
    ## ------------------------------------------------------------
    data_d_raw, df_raw = load_dataset_from_payload(payload)

    ## ------------------------------------------------------------
    ## Preprocess
    ## ------------------------------------------------------------
    enable_candidates = bool(payload.get("enable_candidates", DEFAULT_ENABLE_CANDIDATES))
    df_clean = preprocess_for_deduplication(df_raw.copy(), enable_candidates=enable_candidates)

    ## Convert to record dict for dedupe (use cleaned data)
    id_column = str(payload.get("id_column", DEFAULT_ID_COLUMN))
    data_d = dataframe_to_dedupe_dict(df_clean, id_column=id_column)

    ## ------------------------------------------------------------
    ## Train or load model
    ## ------------------------------------------------------------
    try:
        model_id = int(payload.get("model_id", 1))
    except Exception as exc:
        raise ValidationError(message="model_id must be an integer", details=str(exc))

    deduper_model = get_or_train_dedupe(
        model_id=model_id,
        data_d=data_d,
        df_columns=list(df_clean.columns),
        num_processes=int(payload.get("num_processes", 1)),
        enable_active_learning=bool(payload.get("enable_active_learning", False)),
        use_existing_training=bool(payload.get("use_existing_training", True)),
    )

    ## ------------------------------------------------------------
    ## Cluster / partition
    ## ------------------------------------------------------------
    cluster_membership = get_clusters(
        deduper_model=deduper_model,
        data_d=data_d,
        cluster_threshold=payload.get("cluster_threshold", None),
    )

    ## ------------------------------------------------------------
    ## Build response payload
    ## ------------------------------------------------------------
    return {
        "cluster_membership": cluster_membership,
        "meta": {
            "model_id": model_id,
            "records_count": int(len(data_d)),
            "columns_count": int(df_clean.shape[1]),
            "enable_candidates": enable_candidates,
        },
    }


## ============================================================
## LINKAGE (RECORD -> DATASET)
## ============================================================
def run_record_to_dataset_linkage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Record-to-dataset linkage.

    Expected payload:
        - record: dict (single record to match)
        - model_id: int
        - match_threshold: float (default 0.5)
        - dataset input keys (same as run_dataset_deduplication)

    Returns:
        {
            "matches": [
                {
                    "record_id": int,
                    "confidence_score": float
                }
            ],
            "meta": {...}
        }
    """

    ## Validate record
    if "record" not in payload or not isinstance(payload["record"], dict):
        raise ValidationError(message="Payload must contain a 'record' dictionary")

    record_input = payload["record"]

    ## Load dataset
    data_d_raw, df_raw = load_dataset_from_payload(payload)

    ## Preprocess dataset
    enable_candidates = bool(payload.get("enable_candidates", DEFAULT_ENABLE_CANDIDATES))
    df_clean = preprocess_for_deduplication(df_raw.copy(), enable_candidates=enable_candidates)

    id_column = str(payload.get("id_column", DEFAULT_ID_COLUMN))
    data_d = dataframe_to_dedupe_dict(df_clean, id_column=id_column)

    ## Preprocess single record
    df_record = pd.DataFrame([record_input])
    df_record_clean = preprocess_for_deduplication(df_record.copy(), enable_candidates=enable_candidates)
    record_dict = dataframe_to_dedupe_dict(df_record_clean, id_column=id_column)

    ## force record id to negative to avoid collision
    record_id = -1
    record_dict = {record_id: list(record_dict.values())[0]}

    ## Load or train model
    try:
        model_id = int(payload.get("model_id", 1))
    except Exception as exc:
        raise ValidationError(message="model_id must be an integer", details=str(exc))

    deduper_model = get_or_train_dedupe(
        model_id=model_id,
        data_d=data_d,
        df_columns=list(df_clean.columns),
        num_processes=int(payload.get("num_processes", 1)),
        enable_active_learning=False,
        use_existing_training=True,
    )

    ## Score record vs dataset
    try:
        match_threshold = float(payload.get("match_threshold", 0.5))

        matches = deduper_model.match(
            record_dict,
            data_d,
            threshold=match_threshold,
        )

    except Exception as exc:
        raise DeduplicationError(
            message="Failed during record-to-dataset matching",
            details=str(exc),
        )

    ## Format output
    formatted_matches = []

    for match in matches:
        record_pair, score = match
        matched_id = record_pair[1]

        formatted_matches.append({
            "record_id": int(matched_id),
            "confidence_score": float(score),
        })

    formatted_matches = sorted(
        formatted_matches,
        key=lambda x: x["confidence_score"],
        reverse=True,
    )

    return {
        "matches": formatted_matches,
        "meta": {
            "model_id": model_id,
            "match_threshold": match_threshold,
            "matches_found": len(formatted_matches),
        },
    }