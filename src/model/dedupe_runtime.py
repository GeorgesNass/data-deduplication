'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Dedupe runtime utilities: model loading/creation, training persistence, optional console labeling, and clustering."
'''

from __future__ import annotations

## Standard library imports
from pathlib import Path
from typing import Any, Dict, Tuple

## Third-party imports
import dedupe

## Local imports
from src.core.errors import DeduplicationError
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("model.dedupe_runtime")

## ============================================================
## DEDUPE MODEL LOADING / TRAINING
## ============================================================
def load_static_deduper(settings_path: Path) -> dedupe.StaticDedupe:
    """
        Load a StaticDedupe instance from a settings file

        Args:
            settings_path: Path to variables_predicates_weights_<id> (binary)

        Returns:
            StaticDedupe instance

        Raises:
            DeduplicationError: If loading fails
    """
    
    if not settings_path.exists():
        raise DeduplicationError(
            message="Dedupe settings file not found",
            details=str(settings_path),
        )

    try:
        with settings_path.open("rb") as f:
            return dedupe.StaticDedupe(f)
    except Exception as exc:
        raise DeduplicationError(
            message="Failed to load dedupe settings",
            details=str(exc),
        )

def create_trainable_deduper(fields: list[dict[str, Any]], num_processes: int = 1) -> dedupe.Dedupe:
    """
        Create a trainable dedupe.Dedupe instance

        Args:
            fields: Dedupe field definitions
            num_processes: Parallel processes used by dedupe

        Returns:
            dedupe.Dedupe instance
    """
    
    return dedupe.Dedupe(fields, num_processes=max(1, int(num_processes)))

def prepare_training(
    deduper: dedupe.Dedupe,
    data_d: Dict[int, Dict[str, Any]],
    training_path: Path,
    use_existing_training: bool,
) -> None:
    """
        Prepare training using data and optional persisted labels

        Args:
            deduper: Trainable dedupe instance
            data_d: Record dictionary used by dedupe
            training_path: Path to trained_model_config_<id> (JSON)
            use_existing_training: If True, load existing labeled pairs when present
    """
    
    ## If existing labels exist and user allows, load them
    if training_path.exists() and use_existing_training:
        LOGGER.info("Loading labeled examples from: %s", training_path)
        with training_path.open("rb") as f:
            deduper.prepare_training(data_d, f)
        return

    ## Otherwise, start from scratch
    LOGGER.info("Preparing training from scratch (no prior labels used)")
    deduper.prepare_training(data_d)

def run_console_active_learning(deduper: dedupe.Dedupe) -> None:
    """
        Run interactive console labeling (active learning)

        Important:
            - This requires a TTY (interactive terminal)
            - In API contexts, this should usually be disabled

        Args:
            deduper: Trainable dedupe instance
    """
    ## Use dedupe built-in interactive labeling
    dedupe.console_label(deduper)

def train_and_persist(
    deduper: dedupe.Dedupe,
    training_path: Path,
    settings_path: Path,
) -> None:
    """
        Train a dedupe model and persist labels + settings

        Args:
            deduper: Trainable dedupe instance
            training_path: Path where labeled training examples are saved (text)
            settings_path: Path where dedupe settings are saved (binary)
    """
    
    ## Train the model
    LOGGER.info("Training dedupe model...")
    deduper.train()

    ## Persist labeled training data
    LOGGER.info("Writing training file: %s", training_path)
    with training_path.open("w", encoding="utf-8") as tf:
        deduper.write_training(tf)

    ## Persist settings (predicates + weights)
    LOGGER.info("Writing settings file: %s", settings_path)
    with settings_path.open("wb") as sf:
        deduper.write_settings(sf)

## ============================================================
## CLUSTERING
## ============================================================
def get_clusters(
    deduper_model: dedupe.api.Dedupe,
    data_d: Dict[int, Dict[str, Any]],
    cluster_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    """
        Partition records into clusters and return cluster metadata per record

        Output format:
            {
                record_id: {
                    "Cluster ID": int,
                    "confidence_score": float
                }
            }

        Args:
            deduper_model: Trained dedupe model (StaticDedupe or Dedupe)
            data_d: Record dictionary passed to dedupe.partition
            cluster_threshold: Partition threshold

        Returns:
            Per-record cluster membership mapping

        Raises:
            DeduplicationError: If partitioning fails
    """
    
    ## Run partitioning
    try:
        clustered_dupes = deduper_model.partition(data_d, float(cluster_threshold))
    except Exception as exc:
        raise DeduplicationError(
            message="Failed to partition records into clusters",
            details=str(exc),
        )

    ## Build membership mapping
    cluster_membership: Dict[int, Dict[str, Any]] = {}

    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        for record_id, score in zip(records, scores):
            cluster_membership[int(record_id)] = {
                "Cluster ID": int(cluster_id),
                "confidence_score": float(score),
            }

    return cluster_membership