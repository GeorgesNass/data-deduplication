'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Centralized custom exceptions and helpers for the data deduplication pipeline."
'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("errors")

## ============================================================
## ERROR CODES
## ============================================================
ERROR_CODE_CONFIGURATION = "configuration_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_SCHEMA = "schema_validation_error"
ERROR_CODE_PIPELINE = "pipeline_error"
ERROR_CODE_STEP = "step_execution_error"
ERROR_CODE_DEDUPLICATION = "deduplication_error"
ERROR_CODE_ACTIVE_LEARNING = "active_learning_error"
ERROR_CODE_API = "api_error"
ERROR_CODE_RESOURCE_NOT_FOUND = "resource_not_found"
ERROR_CODE_INTERNAL = "internal_error"

## ============================================================
## BASE EXCEPTION
## ============================================================
class ApplicationError(RuntimeError):
    """
        Base exception for the data deduplication pipeline

        High-level workflow:
            1) Normalize application failures
            2) Preserve contextual metadata
            3) Provide structured logging support

        Args:
            message: Human-readable error message
            error_code: Normalized error code
            details: Optional structured context payload
            cause: Original exception if available
            is_retryable: Whether retry may succeed
    """

    def __init__(
        self,
        message: str,
        error_code: str = ERROR_CODE_INTERNAL,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        is_retryable: bool = False,
    ) -> None:

        ## Store normalized metadata
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.is_retryable = is_retryable

        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
            Convert exception to a structured dictionary

            Returns:
                A normalized error payload
        """

        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "cause_type": self.cause.__class__.__name__
            if self.cause
            else None,
            "is_retryable": self.is_retryable,
        }

## ============================================================
## CONFIGURATION ERRORS
## ============================================================
class ConfigurationError(ApplicationError):
    """
        Raised when application configuration is invalid
    """

## ============================================================
## VALIDATION ERRORS
## ============================================================
class ValidationError(ApplicationError):
    """
        Raised when validation checks fail
    """

class SchemaValidationError(ApplicationError):
    """
        Raised when input data schema validation fails
    """

## ============================================================
## PIPELINE ERRORS
## ============================================================
class PipelineError(ApplicationError):
    """
        Raised when the pipeline orchestration fails
    """

class StepExecutionError(ApplicationError):
    """
        Raised when an individual pipeline step fails
    """

## ============================================================
## DOMAIN ERRORS
## ============================================================
class DeduplicationError(ApplicationError):
    """
        Raised when deduplication logic fails
    """

class ActiveLearningError(ApplicationError):
    """
        Raised when active learning logic fails
    """

class ApiError(ApplicationError):
    """
        Raised when an external API interaction fails
    """

class ResourceNotFoundError(ApplicationError):
    """
        Raised when a required file or artifact is missing
    """

class UnknownApplicationError(ApplicationError):
    """
        Raised when an unexpected exception must be normalized
    """

## ============================================================
## GENERIC HELPERS
## ============================================================
def raise_project_error(
    exc_type: Type[ApplicationError],
    message: str,
    *,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None,
    is_retryable: bool = False,
) -> None:
    """
        Log and raise a structured project exception

        High-level workflow:
            1) Build normalized payload
            2) Attach original cause metadata
            3) Emit structured error log
            4) Raise normalized exception

        Args:
            exc_type: Exception class to raise
            message: Human-readable error message
            error_code: Normalized error code
            details: Optional contextual payload
            cause: Original exception if available
            is_retryable: Whether retry may succeed
    """

    ## Build normalized payload
    payload = details.copy() if details else {}

    ## Attach original exception metadata
    if cause is not None:
        payload["cause_message"] = str(cause)
        payload["cause_type"] = cause.__class__.__name__

    ## Emit structured error log
    logger.error(
        "Deduplication pipeline error | type=%s | code=%s | message=%s | "
        "retryable=%s | details=%s",
        exc_type.__name__,
        error_code,
        message,
        is_retryable,
        payload,
    )

    ## Raise normalized exception
    raise exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=cause,
        is_retryable=is_retryable,
    )

def wrap_exception(
    exc: Exception,
    *,
    exc_type: Type[ApplicationError],
    message: str,
    error_code: str,
    details: Optional[Dict[str, Any]] = None,
) -> ApplicationError:
    """
        Wrap a raw exception into a structured project exception

        Args:
            exc: Original exception
            exc_type: Target exception type
            message: Human-readable error message
            error_code: Normalized error code
            details: Optional contextual payload

        Returns:
            Structured application error
    """

    ## Build contextual payload
    payload = details.copy() if details else {}

    ## Attach original exception metadata
    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    return exc_type(
        message=message,
        error_code=error_code,
        details=payload,
        cause=exc,
    )

def log_unhandled_exception(
    exc: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> UnknownApplicationError:
    """
        Normalize unexpected exceptions

        Args:
            exc: Original unexpected exception
            context: Optional execution context

        Returns:
            UnknownApplicationError
    """

    ## Build contextual payload
    payload = context.copy() if context else {}

    payload["cause_message"] = str(exc)
    payload["cause_type"] = exc.__class__.__name__

    logger.error(
        "Unhandled deduplication exception | type=%s | details=%s",
        exc.__class__.__name__,
        payload,
    )

    logger.debug("Unhandled traceback", exc_info=True)

    return UnknownApplicationError(
        message="Unexpected deduplication pipeline error",
        error_code=ERROR_CODE_INTERNAL,
        details=payload,
        cause=exc,
    )

## ============================================================
## SPECIALIZED HELPERS
## ============================================================
def log_and_raise_missing_path(
    path: str | Path,
    *,
    resource_name: str = "Required resource",
) -> None:
    """
        Log and raise missing resource error

        Args:
            path: Missing filesystem path
            resource_name: Human-readable resource label
    """

    ## Normalize path
    normalized_path = str(Path(path))

    raise_project_error(
        exc_type=ResourceNotFoundError,
        message=f"{resource_name} not found",
        error_code=ERROR_CODE_RESOURCE_NOT_FOUND,
        details={"path": normalized_path},
    )

def log_and_raise_schema_error(message: str) -> None:
    """
        Log and raise schema validation error
    """

    raise_project_error(
        exc_type=SchemaValidationError,
        message=message,
        error_code=ERROR_CODE_SCHEMA,
    )

def log_and_raise_deduplication_error(message: str) -> None:
    """
        Log and raise deduplication error
    """

    raise_project_error(
        exc_type=DeduplicationError,
        message=message,
        error_code=ERROR_CODE_DEDUPLICATION,
    )

def log_and_raise_active_learning_error(message: str) -> None:
    """
        Log and raise active learning error
    """

    raise_project_error(
        exc_type=ActiveLearningError,
        message=message,
        error_code=ERROR_CODE_ACTIVE_LEARNING,
    )

def log_and_raise_api_error(message: str) -> None:
    """
        Log and raise API error
    """

    raise_project_error(
        exc_type=ApiError,
        message=message,
        error_code=ERROR_CODE_API,
        is_retryable=True,
    )

def log_and_raise_pipeline_step(step: str, reason: str) -> None:
    """
        Log and raise pipeline step error

        Args:
            step: Pipeline step name
            reason: Failure reason
    """

    ## Build pipeline message
    message = f"Pipeline step failed [{step}]: {reason}"

    logger.error(message)

    raise StepExecutionError(
        message=message,
        error_code=ERROR_CODE_STEP,
        details={"step": step, "reason": reason},
    )