'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "I/O helper utilities: CSV read/write helpers and base64 decoding to file."
'''

from __future__ import annotations

## Standard library imports
import os
import redis

try:
    from feast import FeatureStore
except:
    FeatureStore = None
    
import csv
import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

## Third-party imports
import pandas as pd

## Local imports
from src.core.errors import ValidationError
from src.utils.utils_core import safe_int

## ============================================================
## GLOBALS
## ============================================================
FEATURE_STORE_MODE = os.getenv("FEATURE_STORE_MODE", "redis")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

## Increase CSV field size to handle large text fields safely
csv.field_size_limit(3_000_000)

## ============================================================
## CSV HELPERS (LOCAL ONLY)
## ============================================================
def read_csv_as_dict_and_df(
    filename: str | Path,
    *,
    id_column: str = "id",
    limit: int | None = None,
    cleaning_fn: Callable[[Any], Any] | None = None,
    encoding: str = "utf-8",
) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
        Read a CSV file and produce both dict records and a DataFrame

        Design:
            - Local filesystem only (no cloud / no mongo)
            - Optional per-cell cleaning function
            - Dict keys are integer ids from id_column or auto-incremented

        Args:
            filename: Path to CSV file
            id_column: Column containing unique record id
            limit: Max number of rows to read (None means no limit)
            cleaning_fn: Optional function applied to each cell value
            encoding: File encoding

        Returns:
            Tuple:
                - data_d: dict[int, dict[str, Any]] mapping id -> row dict
                - df_raw: pandas DataFrame with all rows
    """
    
    file_path = Path(filename).expanduser().resolve()
    data_d: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    with file_path.open("r", encoding=encoding, newline="") as f_in:
        reader = csv.DictReader(f_in)

        count = 0
        for row in reader:
            ## Apply optional cleaning per value
            if cleaning_fn is not None:
                cleaned_row = {k: cleaning_fn(v) for k, v in row.items()}
            else:
                cleaned_row = dict(row)

            ## Resolve row id
            if id_column in cleaned_row and cleaned_row[id_column] not in (None, ""):
                row_id = safe_int(cleaned_row[id_column], default=None)
                if row_id is None:
                    row_id = len(data_d) + 1
            else:
                row_id = len(data_d) + 1

            data_d[row_id] = cleaned_row
            rows.append(cleaned_row)

            count += 1
            if limit is not None and count >= limit:
                break

    df_raw = pd.DataFrame(rows)
    
    return data_d, df_raw

def write_csv_with_cluster_metadata(
    input_csv: str | Path,
    output_csv: str | Path,
    cluster_membership: Dict[int, Dict[str, Any]],
    *,
    id_column: str = "id",
    cluster_id_key: str = "Cluster ID",
    score_key: str = "confidence_score",
    keep_columns: Optional[List[str]] = None,
    flag_sensitive_filter: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """
        Write a CSV enriched with clustering metadata

        Notes:
            - cluster_membership is expected as: {row_id: {"Cluster ID": x, "confidence_score": y, ...}}
            - If flag_sensitive_filter is True, only keep cluster fields + keep_columns
            - Adds a computed Cluster_size column at the end

        Args:
            input_csv: Original input CSV path
            output_csv: Output CSV path
            cluster_membership: Mapping row_id -> metadata dict to merge into row
            id_column: Column used to fetch row_id
            cluster_id_key: Key name for cluster id in output
            score_key: Key name for confidence score in output
            keep_columns: Columns to keep if sensitive filtering is enabled
            flag_sensitive_filter: If True, filter output columns
            encoding: File encoding

        Returns:
            Path to the written CSV
    """
    
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ## Build allowed columns list for sensitive mode
    allowed_columns: List[str] = [cluster_id_key, score_key]
    if keep_columns:
        allowed_columns.extend(keep_columns)

    with input_path.open("r", encoding=encoding, newline="") as f_in, \
            output_path.open("w", encoding=encoding, newline="") as f_out:

        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        fieldnames = [cluster_id_key, score_key] + list(reader.fieldnames)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_id = safe_int(row.get(id_column), default=None)
            if row_id is None:
                ## Fallback to position-based index if id is missing
                row_id = len(cluster_membership) + 1

            if row_id in cluster_membership:
                row.update(cluster_membership[row_id])

            if flag_sensitive_filter:
                writer.writerow({k: row.get(k) for k in allowed_columns if k in row})
            else:
                writer.writerow(row)

    ## Add Cluster_size using pandas (stable and simple)
    df_out = pd.read_csv(output_path, sep=",", encoding=encoding)
    if cluster_id_key in df_out.columns:
        df_out["Cluster_size"] = df_out.groupby(cluster_id_key)[cluster_id_key].transform("size")
        df_out = df_out.sort_values([cluster_id_key], ascending=[True])

    df_out.to_csv(output_path, sep=",", encoding=encoding, index=False)
    return output_path

## ============================================================
## BASE64 HELPERS
## ============================================================
def decode_base64_to_file(base64_content: str, output_path: str | Path) -> Path:
    """
        Decode base64 content into a file on disk

        Args:
            base64_content: Base64-encoded file content
            output_path: Output file path

        Returns:
            Path to the written file

        Raises:
            ValidationError: If base64 is invalid
    """

    try:
        decoded = base64.b64decode(base64_content, validate=True)
    except Exception as exc:
        raise ValidationError(
            message="Invalid base64 content provided",
            details=str(exc),
        )

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(decoded)
    
    return out_path

## ============================================================
## FEATURE ENGINEERING
## ============================================================
def build_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
        Build feature dictionary from a raw input row

        Design:
            - Lightweight feature engineering
            - Works for text and numeric data
            - No external dependency (pure Python)

        Args:
            row: Input record as dictionary

        Returns:
            dict[str, Any]: Feature dictionary
    """

    features: Dict[str, Any] = {}

    for k, v in row.items():
        ## Text features
        if isinstance(v, str):
            features[f"{k}_length"] = len(v)
            features[f"{k}_lower"] = v.lower()

        ## Numeric features
        if isinstance(v, (int, float)):
            features[f"{k}_scaled"] = v

    return features
    
## ============================================================
## FEATURE STORE - REDIS (LOCAL)
## ============================================================
def push_features_redis(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Store features in Redis

        Args:
            entity_id: Unique identifier
            features: Feature dictionary

        Returns:
            None
    """

    ## Store as Redis hash
    redis_client.hset(entity_id, mapping=features)


def get_features_redis(entity_id: str) -> Dict[str, Any]:
    """
        Retrieve features from Redis

        Args:
            entity_id: Unique identifier

        Returns:
            dict[str, Any]: Feature dictionary
    """

    ## Fetch from Redis
    return redis_client.hgetall(entity_id)    
    
## ============================================================
## FEATURE STORE - FEAST (ONLINE)
## ============================================================
def get_feast_store():
    """
        Initialize Feast Feature Store

        Returns:
            FeatureStore instance

        Raises:
            ImportError: If Feast is not installed
    """

    if FeatureStore is None:
        raise ImportError("Feast not installed")

    ## Initialize store
    return FeatureStore(repo_path=FEAST_REPO_PATH)

def push_features_feast(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Push features to Feast online store

        Args:
            entity_id: Unique identifier
            features: Feature dictionary

        Returns:
            None
    """

    import pandas as pd

    ## Convert to DataFrame
    df = pd.DataFrame([{**features, "entity_id": entity_id}])

    ## Push to Feast
    store = get_feast_store()
    store.write_to_online_store(df)

def get_features_feast(entity_id: str) -> Dict[str, Any]:
    """
        Retrieve features from Feast

        Args:
            entity_id: Unique identifier

        Returns:
            dict[str, Any]: Feature dictionary
    """

    ## Query Feast
    store = get_feast_store()
    result = store.get_online_features(
        features=["features:*"],
        entity_rows=[{"entity_id": entity_id}]
    ).to_dict()

    return result    
     
## ============================================================
## FEATURE STORE - UNIFIED API
## ============================================================
def push_features(entity_id: str, features: Dict[str, Any]) -> None:
    """
        Unified interface to store features

        Args:
            entity_id: Unique identifier
            features: Feature dictionary

        Returns:
            None
    """

    ## Route based on mode
    if FEATURE_STORE_MODE == "redis":
        push_features_redis(entity_id, features)
    else:
        push_features_feast(entity_id, features)

def get_features(entity_id: str) -> Dict[str, Any]:
    """
        Unified interface to retrieve features

        Args:
            entity_id: Unique identifier

        Returns:
            dict[str, Any]: Feature dictionary
    """

    ## Route based on mode
    if FEATURE_STORE_MODE == "redis":
        return get_features_redis(entity_id)
    else:
        return get_features_feast(entity_id)