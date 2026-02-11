'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Pipeline dispatcher: route function names to business handlers and return standardized responses."
'''

from __future__ import annotations

## Standard library imports
from typing import Any, Callable, Dict

## Local imports
from src.core.controls import build_success_response, safe_run
from src.core.errors import ValidationError
from src.model.active_learning import load_full_active_learning_state
from src.model.deduplication import (
    run_dataset_deduplication,
    run_record_to_dataset_linkage,
)

## NEW: EDA entrypoint
from src.eda.reports import run_eda_analysis

from src.utils.logging_utils import get_logger

## LOGGER
LOGGER = get_logger("pipeline")

## TYPES
Handler = Callable[[Any], Dict[str, Any]]

## ============================================================
## HANDLERS
## ============================================================
def handle_train_model(payload: Any) -> Dict[str, Any]:
    """
        Trigger model training or warmup logic

        Design choice:
            - No standalone training pipeline is enforced here
            - Training is performed implicitly during deduplication
            - This handler preserves backward API compatibility

        Args:
            payload: Validated TrainModelPayload object

        Returns:
            Dictionary containing training metadata or instructions
    """

    LOGGER.info("trainModel called. Using train-on-demand strategy.")

    return run_dataset_deduplication(payload)

def handle_dataset_deduplication(payload: Any) -> Dict[str, Any]:
    """
        Run dataset deduplication and clustering

        Args:
            payload: Validated DatasetDeduplicationPayload object

        Returns:
            Dictionary containing clusters and metadata
    """

    return run_dataset_deduplication(payload)

def handle_record_dataset_linkage(payload: Any) -> Dict[str, Any]:
    """
        Link a single record to an existing dataset

        Args:
            payload: Validated RecordLinkagePayload object

        Returns:
            Dictionary containing linkage results
    """

    return run_record_to_dataset_linkage(payload)

def handle_get_models_info(payload: Any) -> Dict[str, Any]:
    """
        Return metadata about persisted active learning models

        Args:
            payload: Optional payload containing model_id

        Returns:
            Dictionary containing model state and metadata

        Raises:
            ValidationError: If model_id is invalid
    """

    try:
        model_id = int(getattr(payload, "model_id", 1))
    except Exception as exc:
        raise ValidationError(
            message="model_id must be an integer",
            details=str(exc),
        )

    state = load_full_active_learning_state(model_id)

    return {
        "model_id": model_id,
        "state": state,
    }

def handle_eda_analysis(payload: Any) -> Dict[str, Any]:
    """
        Run EDA analysis (stats + reports + plots) on a dataset

        Design:
            - Uses the same dataset input conventions as deduplication payloads
            - Delegates all business logic to src.eda.reports.run_eda_analysis

        Args:
            payload: Validated EDAPayload object (or dict-like payload)

        Returns:
            Dictionary containing EDA artifacts metadata (paths, summaries)
    """

    return run_eda_analysis(payload)

## ============================================================
## DISPATCH MAP
## ============================================================
FUNCTION_DISPATCH: Dict[str, Dict[str, Any]] = {
    "trainModel": {
        "handler": handle_train_model,
        "success_message": "Model training completed",
    },
    "datasetDeduplicationCluster": {
        "handler": handle_dataset_deduplication,
        "success_message": "Dataset deduplication completed",
    },
    "recordDatasetLinkage": {
        "handler": handle_record_dataset_linkage,
        "success_message": "Record-to-dataset linkage completed",
    },
    "getModelsInfo": {
        "handler": handle_get_models_info,
        "success_message": "Models info loaded",
    },
    ## NEW: EDA
    "edaAnalysis": {
        "handler": handle_eda_analysis,
        "success_message": "EDA analysis completed",
    },
}

## ============================================================
## PUBLIC DISPATCHER
## ============================================================
def run_pipeline(function_name: str, payload: Any | None = None) -> Dict[str, Any]:
    """
        Execute a pipeline function by name and return a standardized response

        High-level workflow:
            1) Resolve handler from FUNCTION_DISPATCH
            2) Execute handler safely (error normalization handled by safe_run)
            3) Wrap successful output into a standardized response

        Design choice:
            - Payload is already validated by FastAPI (Pydantic)
            - Pipeline is validation-free and orchestration-only

        Args:
            function_name: Internal function name
            payload: Validated Pydantic payload or None

        Returns:
            Standardized API-ready response dictionary
    """

    payload = payload or {}

    if function_name not in FUNCTION_DISPATCH:
        return {
            "code": "404",
            "type": "UNKNOWN_FUNCTION",
            "message": f"Unknown function_name: {function_name}",
        }

    handler: Handler = FUNCTION_DISPATCH[function_name]["handler"]
    success_message: str = FUNCTION_DISPATCH[function_name]["success_message"]

    execution_result = safe_run(handler, payload)

    ## Pass-through error responses
    if execution_result.get("code") not in (None, "200"):
        return execution_result

    return build_success_response(
        message=success_message,
        data={
            "function_name": function_name,
            "result": execution_result,
        },
    )