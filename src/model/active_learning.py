'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Active learning persistence layer: loading, saving, and updating fuzzy model configurations."
'''

from __future__ import annotations

## Standard library imports
import json
from pathlib import Path
from typing import Any, Dict, Optional

## Local imports
from src.core.config import ACTIVE_LEARNING_DIR, ensure_dir
from src.core.errors import ActiveLearningError
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("model.active_learning")

## ============================================================
## DEFAULTS
## ============================================================
DEFAULT_MODEL_PREFIX = "trained_model_config"
DEFAULT_WEIGHTS_PREFIX = "variables_predicates_weights"

## ============================================================
## PATH RESOLUTION
## ============================================================
def _resolve_model_path(
    model_id: int,
    *,
    prefix: str,
) -> Path:
    """
        Resolve a model-related file path inside active_learning directory

        Args:
            model_id: Numeric model identifier
            prefix: Filename prefix

        Returns:
            Absolute Path to the model file
    """
    
    ensure_dir(ACTIVE_LEARNING_DIR)
    filename = f"{prefix}_{model_id}"
    
    return (ACTIVE_LEARNING_DIR / filename).resolve()

## ============================================================
## LOADERS
## ============================================================
def load_active_learning_model(model_id: int) -> Dict[str, Any]:
    """
        Load a persisted active learning model configuration

        Args:
            model_id: Model identifier

        Returns:
            Loaded model configuration dictionary

        Raises:
            ActiveLearningError: If loading fails
    """
    
    path = _resolve_model_path(model_id, prefix=DEFAULT_MODEL_PREFIX)

    if not path.exists():
        raise ActiveLearningError(
            message="Active learning model not found",
            details=f"Missing file: {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ActiveLearningError(
            message="Failed to load active learning model",
            details=str(exc),
        )

def load_predicate_weights(model_id: int) -> Dict[str, Any]:
    """
        Load predicate weights for a given active learning model

        Args:
            model_id: Model identifier

        Returns:
            Loaded predicate weights dictionary

        Raises:
            ActiveLearningError: If loading fails
    """
    
    path = _resolve_model_path(model_id, prefix=DEFAULT_WEIGHTS_PREFIX)

    if not path.exists():
        raise ActiveLearningError(
            message="Predicate weights not found",
            details=f"Missing file: {path}",
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ActiveLearningError(
            message="Failed to load predicate weights",
            details=str(exc),
        )

## ============================================================
## SAVERS
## ============================================================
def save_active_learning_model(
    model_id: int,
    payload: Dict[str, Any],
) -> Path:
    """
        Save an active learning model configuration to disk

        Args:
            model_id: Model identifier
            payload: Model configuration dictionary

        Returns:
            Path to the written model file

        Raises:
            ActiveLearningError: If saving fails
    """
    
    path = _resolve_model_path(model_id, prefix=DEFAULT_MODEL_PREFIX)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ActiveLearningError(
            message="Failed to save active learning model",
            details=str(exc),
        )

    LOGGER.info("Active learning model saved: %s", path)
    
    return path

def save_predicate_weights(
    model_id: int,
    payload: Dict[str, Any],
) -> Path:
    """
        Save predicate weights for an active learning model

        Args:
            model_id: Model identifier
            payload: Predicate weights dictionary

        Returns:
            Path to the written weights file

        Raises:
            ActiveLearningError: If saving fails
    """
    
    path = _resolve_model_path(model_id, prefix=DEFAULT_WEIGHTS_PREFIX)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ActiveLearningError(
            message="Failed to save predicate weights",
            details=str(exc),
        )

    LOGGER.info("Predicate weights saved: %s", path)
    
    return path

## ============================================================
## HIGH-LEVEL HELPERS
## ============================================================
def load_full_active_learning_state(model_id: int) -> Dict[str, Any]:
    """
        Load both model configuration and predicate weights

        Args:
            model_id: Model identifier

        Returns:
            Dictionary containing model + weights
    """
    
    model_cfg = load_active_learning_model(model_id)
    weights_cfg = load_predicate_weights(model_id)

    return {
        "model": model_cfg,
        "weights": weights_cfg,
    }

def save_full_active_learning_state(
    model_id: int,
    *,
    model_payload: Dict[str, Any],
    weights_payload: Dict[str, Any],
) -> Dict[str, Path]:
    """
        Save both model configuration and predicate weights

        Args:
            model_id: Model identifier
            model_payload: Model configuration dictionary
            weights_payload: Predicate weights dictionary

        Returns:
            Dictionary with written file paths
    """
    
    model_path = save_active_learning_model(model_id, model_payload)
    weights_path = save_predicate_weights(model_id, weights_payload)

    return {
        "model_path": model_path,
        "weights_path": weights_path,
    }