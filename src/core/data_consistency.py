'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Data consistency for data deduplication pipeline: dataset, text, schema, duplicates, thresholds"
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger
from src.utils.data_utils import (
    normalize_data,
    validate_schema,
    validate_types,
    compute_quality_score,
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_consistency")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _add_issue(
    issues: List[Dict[str, Any]],
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
        Append issue and log it

        Args:
            issues: Issue list
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            None
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    issues.append(issue)

    ## Log
    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## VALIDATIONS
## ============================================================
def _validate_dataset(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate dataset structure

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    records = data.get("records")

    ## Dataset must exist
    if not records:
        _add_issue(issues, "dataset_missing", "error", "Dataset is required")
        return

    if not isinstance(records, list):
        _add_issue(issues, "dataset_type", "error", "Dataset must be list")
        return

    if len(records) == 0:
        _add_issue(issues, "dataset_empty", "error", "Dataset is empty")
        return

    for idx, record in enumerate(records):

        ## Each record must be dict
        if not isinstance(record, dict):
            _add_issue(issues, "record_format", "error", "Record must be dict", {"index": idx})
            continue

        ## Text field required
        text = record.get("text")

        if not isinstance(text, str) or not text.strip():
            _add_issue(issues, "text_invalid", "error", "Invalid text", {"index": idx})

def _validate_duplicates(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Check duplicates in dataset

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    records = data.get("records", [])

    seen = set()

    for idx, record in enumerate(records):

        text = record.get("text", "")

        ## Normalize for comparison
        key = text.strip().lower()

        if key in seen:
            _add_issue(
                issues,
                "duplicate_detected",
                "warning",
                "Duplicate detected",
                {"index": idx},
            )
        else:
            seen.add(key)

def _validate_thresholds(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate similarity threshold

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    threshold = data.get("similarity_threshold", 0.8)

    ## Must be float between 0 and 1
    if not isinstance(threshold, (float, int)):
        _add_issue(issues, "threshold_type", "error", "Threshold must be numeric")
        return

    if not (0 <= threshold <= 1):
        _add_issue(issues, "threshold_range", "error", "Threshold must be between 0 and 1")

def _validate_structure(
    data: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    """
        Validate schema and types

        Args:
            data: Input data
            issues: Issue list

        Returns:
            None
    """

    ## Schema validation
    for s in validate_schema(data):
        _add_issue(issues, s["rule"], s["level"], s["message"])

    ## Type validation
    for t in validate_types(data):
        _add_issue(issues, t["rule"], t["level"], t["message"])

## ============================================================
## QUALITY
## ============================================================
def _compute_quality(
    data: Dict[str, Any],
) -> float:
    """
        Compute quality score

        Args:
            data: Input data

        Returns:
            float
    """

    return compute_quality_score(data)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_consistency(
    data: Dict[str, Any],
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data consistency for deduplication pipeline

        Args:
            data: Input data
            strict: Raise error if inconsistency

        Returns:
            Dict[str, Any]
    """

    issues: List[Dict[str, Any]] = []

    try:
        ## Normalize input
        data = normalize_data(data)

        ## Validate dataset
        _validate_dataset(data, issues)

        ## Validate duplicates
        _validate_duplicates(data, issues)

        ## Validate thresholds
        _validate_thresholds(data, issues)

        ## Validate schema/types
        _validate_structure(data, issues)

        ## Compute quality
        quality_score = _compute_quality(data)

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_consistent": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "quality_score": quality_score,
            "issues": issues,
        }

        ## Strict mode
        if strict and errors:
            raise ValidationError("Data consistency failed")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Consistency pipeline failed") from exc