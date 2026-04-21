'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized data drift detection for deduplication: text distribution and duplicate ratio monitoring."
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.drift_utils import (
    compute_ks_test,
    compute_chi2_test,
    compute_text_stats,
    compute_duplicate_ratio,
    generate_evidently_report
)

try:
    from src.core.errors import ValidationError, DataError
except Exception:
    ValidationError = ValueError
    DataError = RuntimeError

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("data_drift")

## ============================================================
## ISSUE HANDLING
## ============================================================
def _create_issue(
    rule: str,
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
        Create standardized issue object

        Args:
            rule: Rule name
            level: Severity level
            message: Description
            details: Optional metadata

        Returns:
            Issue dictionary
    """

    issue = {
        "rule": rule,
        "level": level,
        "message": message,
        "details": details or {},
    }

    logger.debug(f"Issue created: {rule} - {level}")

    return issue

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
            level: Severity
            message: Description
            details: Metadata
    """

    issue = _create_issue(rule, level, message, details)
    issues.append(issue)

    if level == "error":
        logger.error(f"{rule} - {message}")
    else:
        logger.warning(f"{rule} - {message}")

## ============================================================
## DRIFT DETECTION
## ============================================================
def _detect_numeric_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift for numerical feature

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_ks_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_numeric",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

def _detect_categorical_drift(
    ref: pd.Series,
    cur: pd.Series,
    column: str,
    threshold: float,
    issues: List[Dict[str, Any]],
) -> float:
    """
        Detect drift for categorical feature

        Args:
            ref: Reference series
            cur: Current series
            column: Column name
            threshold: p-value threshold
            issues: Issue container

        Returns:
            p_value
    """

    stat, p_value = compute_chi2_test(ref, cur)

    if p_value < threshold:
        _add_issue(
            issues,
            "drift_categorical",
            "warning",
            f"Drift detected in {column}",
            {"p_value": float(p_value)},
        )

    return float(p_value)

## ============================================================
## MAIN ENTRYPOINT
## ============================================================
def run_data_drift(
    df_ref: pd.DataFrame,
    df_current: pd.DataFrame,
    p_value_threshold: float = 0.05,
    strict: bool = False,
) -> Dict[str, Any]:
    """
        Run data drift detection for deduplication project

        High-level workflow:
            1) Compute text statistics
            2) Compute duplicate ratio
            3) Detect drift per feature
            4) Aggregate issues and compute score

        Args:
            df_ref: Reference dataset
            df_current: Current dataset
            p_value_threshold: Statistical threshold
            strict: Raise error if drift detected

        Returns:
            Result dictionary with drift score and issues
    """

    issues: List[Dict[str, Any]] = []

    try:
        if df_ref.empty or df_current.empty:
            raise ValidationError("Empty datasets provided")

        ## compute text features
        txt_ref = compute_text_stats(df_ref)
        txt_cur = compute_text_stats(df_current)

        ## compute duplicate ratio
        dup_ref = compute_duplicate_ratio(df_ref)
        dup_cur = compute_duplicate_ratio(df_current)

        drift_flags: List[bool] = []

        ## text features drift
        for col in txt_ref.columns:
            ref_series = txt_ref[col]
            cur_series = txt_cur[col]

            p_value = _detect_numeric_drift(
                ref_series, cur_series, col, p_value_threshold, issues
            )

            drift_flags.append(p_value < p_value_threshold)

        ## duplicate ratio drift
        dup_diff = abs(dup_ref - dup_cur)

        if dup_diff > 0.1:
            _add_issue(
                issues,
                "duplicate_ratio",
                "warning",
                "Duplicate ratio drift detected",
                {"ref": dup_ref, "current": dup_cur},
            )
            drift_flags.append(True)
        else:
            drift_flags.append(False)

        ## compute global drift score
        drift_score = 1.0 - (sum(drift_flags) / len(drift_flags)) if drift_flags else 1.0

        errors = [i for i in issues if i["level"] == "error"]

        result = {
            "is_drift_ok": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(issues) - len(errors),
            "drift_score": drift_score,
            "issues": issues,
        }

        logger.info(f"Drift score: {drift_score}")

        ## EVIDENTLY REPORT
        try:
            report_paths = generate_evidently_report(df_ref, df_current)
            result["evidently_report"] = report_paths
        except Exception as e:
            logger.warning(f"Evidently failed: {e}")
            
        if strict and drift_score < 1.0:
            logger.error("Strict mode failure")
            raise ValidationError("Data drift detected")

        return result

    except ValidationError:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        raise DataError("Data drift pipeline failed") from exc