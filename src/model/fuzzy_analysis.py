'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Fuzzy deduplication helpers: dedupe model setup, (optional) active learning, clustering, and config-driven fields."
'''

from __future__ import annotations

## ============================================================
## Standard library imports
## ============================================================

from pathlib import Path
from typing import Any, Dict, Iterable, List

## ============================================================
## Third-party imports
## ============================================================

import dedupe

## Prefer rapidfuzz if available, fallback to fuzzywuzzy-like API behavior
try:
    from rapidfuzz import fuzz  ## type: ignore
except Exception:  ## pragma: no cover
    from fuzzywuzzy import fuzz  ## type: ignore

## ============================================================
## Local imports
## ============================================================

from src.core.config import ACTIVE_LEARNING_DIR, ensure_dir, load_data_control_config
from src.utils.logging_utils import get_logger

from src.model.dedupe_runtime import (
    load_static_deduper,
    create_trainable_deduper,
    prepare_training,
    run_console_active_learning,
    train_and_persist,
    get_clusters,
)


## ============================================================
## LOGGER
## ============================================================

LOGGER = get_logger("model.fuzzy_analysis")


## ============================================================
## DEFAULTS
## ============================================================

## NOTE: Keep filenames consistent with your historical artifacts
DEFAULT_MODEL_PREFIX = "trained_model_config"
DEFAULT_SETTINGS_PREFIX = "variables_predicates_weights"

## NOTE: Fallback fields if config-driven generation is not possible
DEFAULT_FIELDS = [
    {"field": "birth_family_name", "type": "Text", "has missing": True},
    {"field": "birth_first_name", "type": "Text", "has missing": True},
    {"field": "usage_family_name", "type": "Text", "has missing": True},
    {"field": "usage_first_name", "type": "Text", "has missing": True},
    {"field": "addresses_street", "type": "Text", "has missing": True},
    {"field": "addresses_postal_code", "type": "Text", "has missing": True},
    {"field": "civility", "type": "Exact", "has missing": True},
    {"field": "sex", "type": "Exact", "has missing": True},
]

## NOTE: Default cluster threshold used by dedupe.partition
DEFAULT_CLUSTER_THRESHOLD = 0.5

## NOTE: Comparator threshold for "list-of-values" style matching
DEFAULT_LIST_SIMILARITY_THRESHOLD = 85


## ============================================================
## PATH HELPERS
## ============================================================

def resolve_training_path(model_id: int) -> Path:
    """
        Resolve the training file path for a given model id

        Args:
            model_id: Model identifier

        Returns:
            Absolute Path to the training JSON file
    """
    ensure_dir(ACTIVE_LEARNING_DIR)
    return (Path(ACTIVE_LEARNING_DIR) / f"{DEFAULT_MODEL_PREFIX}_{model_id}").resolve()


def resolve_settings_path(model_id: int) -> Path:
    """
        Resolve the settings file path for a given model id

        Args:
            model_id: Model identifier

        Returns:
            Absolute Path to the settings binary file
    """
    ensure_dir(ACTIVE_LEARNING_DIR)
    return (Path(ACTIVE_LEARNING_DIR) / f"{DEFAULT_SETTINGS_PREFIX}_{model_id}").resolve()


## ============================================================
## CUSTOM COMPARATOR
## ============================================================

def custom_multiple_high(field_1: Iterable[Any], field_2: Iterable[Any]) -> int:
    """
        Custom comparator for list-like fields (emails, phones, address tokens)

        Rule:
            - Return 1 if any pair reaches similarity > DEFAULT_LIST_SIMILARITY_THRESHOLD
            - Return 0 otherwise

        Args:
            field_1: First iterable of values
            field_2: Second iterable of values

        Returns:
            1 if similar enough, else 0
    """
    ## Convert inputs to lists (defensive)
    a = list(field_1) if field_1 is not None else []
    b = list(field_2) if field_2 is not None else []

    ## Compare pairwise with early exit
    for x in a:
        for y in b:
            score = fuzz.ratio(str(x), str(y))
            if score > DEFAULT_LIST_SIMILARITY_THRESHOLD:
                return 1

    return 0


## ============================================================
## FIELD GENERATION
## ============================================================

def build_fields_from_data_control(df_columns: List[str], data_control: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
        Build dedupe field definitions from data_control.json

        Supported config style (best-effort):
            - data_control["variables_setting"] is a dict:
                {
                    "birth_family_name": {"type": "Text"},
                    "emails": {"type": "Exact", "has missing": True},
                    ...
                }

        Design:
            - Add "field" key automatically
            - Add "has missing" if not explicitly provided (default True for safety)
            - Add custom comparator for list-like fields (emails, phones, *_list, addresses_*)

        Args:
            df_columns: Input dataframe column names
            data_control: Loaded configuration dict

        Returns:
            List of dedupe field dictionaries
    """
    ## Extract the variables_setting map if present
    variables_setting = data_control.get("variables_setting", {})
    if not isinstance(variables_setting, dict) or not variables_setting:
        LOGGER.warning("No valid variables_setting found in data_control.json. Using DEFAULT_FIELDS.")
        return list(DEFAULT_FIELDS)

    fields: List[Dict[str, Any]] = []

    ## Build fields only for columns present in the dataset (avoid mismatches)
    for field_name, cfg in variables_setting.items():
        if field_name not in df_columns:
            continue

        ## Ensure dict config
        if not isinstance(cfg, dict):
            continue

        ## Start with a shallow copy to avoid mutating the config in memory
        field_cfg: Dict[str, Any] = dict(cfg)

        ## Force required "field" key
        field_cfg["field"] = field_name

        ## Default missing handling if unspecified
        field_cfg.setdefault("has missing", True)

        ## Add custom comparator for list-like fields
        is_list_like = (
            field_name.startswith("emails")
            or field_name.startswith("telephones")
            or field_name.endswith("_family_name_list")
            or field_name.endswith("_first_name_list")
            or field_name.startswith("addresses_")
        )
        if is_list_like:
            field_cfg["comparator"] = custom_multiple_high

        fields.append(field_cfg)

    ## Fallback if the config exists but none matched df_columns
    if not fields:
        LOGGER.warning("variables_setting produced no usable fields. Using DEFAULT_FIELDS.")
        return list(DEFAULT_FIELDS)

    return fields


## ============================================================
## DEDUPE MODEL ORCHESTRATION
## ============================================================

def get_or_train_dedupe(
    *,
    model_id: int,
    data_d: Dict[int, Dict[str, Any]],
    df_columns: List[str],
    num_processes: int = 1,
    enable_active_learning: bool = False,
    use_existing_training: bool = True,
) -> dedupe.api.Dedupe:
    """
        Get a dedupe model

        Behavior:
            - If settings file exists -> load StaticDedupe and return
            - Else -> build fields, create trainable Dedupe, prepare training
                    optionally run active learning, train, persist, return StaticDedupe

        Args:
            model_id: Identifier used for persistence filenames
            data_d: Record dictionary used by dedupe
            df_columns: Dataset columns (used to generate fields from config)
            num_processes: Number of parallel processes for dedupe training
            enable_active_learning: If True, starts interactive labeling
            use_existing_training: If True, loads existing labeled pairs if available

        Returns:
            A dedupe model instance usable for partitioning
    """
    ## Resolve persistence paths
    settings_path = resolve_settings_path(model_id)
    training_path = resolve_training_path(model_id)

    ## Load config and build fields
    try:
        data_control = load_data_control_config()
    except Exception as exc:
        LOGGER.warning("Could not load data_control.json (%s). Using fallback DEFAULT_FIELDS.", exc)
        data_control = {}

    ## If settings already exist, we can skip training
    if settings_path.exists():
        LOGGER.info("Loading StaticDedupe settings from: %s", settings_path)
        return load_static_deduper(settings_path)

    ## Build fields (config-driven or fallback)
    fields = build_fields_from_data_control(df_columns, data_control)

    ## Create trainable deduper
    LOGGER.info("Creating trainable deduper (num_processes=%s)", num_processes)
    deduper_trainable = create_trainable_deduper(fields, num_processes=num_processes)

    ## Prepare training data (and optional existing labels)
    prepare_training(
        deduper=deduper_trainable,
        data_d=data_d,
        training_path=training_path,
        use_existing_training=use_existing_training,
    )

    ## Optional interactive active learning
    if enable_active_learning:
        LOGGER.info("Starting interactive active learning...")
        run_console_active_learning(deduper_trainable)
    else:
        LOGGER.info("Active learning disabled. Skipping interactive labeling.")

    ## Train and persist
    train_and_persist(
        deduper=deduper_trainable,
        training_path=training_path,
        settings_path=settings_path,
    )

    ## Return a StaticDedupe model for partitioning stability
    LOGGER.info("Reloading StaticDedupe from saved settings for stable partitioning")
    return load_static_deduper(settings_path)