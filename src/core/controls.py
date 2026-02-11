'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Standardized API responses and safe execution wrappers (FastAPI)."
'''

from __future__ import annotations

## Standard library imports
from typing import Any, Callable, Dict, TypeVar

## Local imports
from src.core.errors import ApplicationError, ValidationError
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
LOGGER = get_logger("core.controls")

T = TypeVar("T")

## ============================================================
## CONSTANTS
## ============================================================
SUCCESS_CODE = "200"
BAD_REQUEST_CODE = "400"
SERVER_ERROR_CODE = "500"

SUCCESS_TYPE = "SUCCESS"
VALIDATION_ERROR_TYPE = "VALIDATION_ERROR"
UNKNOWN_ERROR_TYPE = "UNKNOWN_ERROR"

## ============================================================
## RESPONSE BUILDERS
## ============================================================
def build_success_response(
    *,
    message: str,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a standardized success response dictionary

    Design choice:
        - Responses are always JSON-serializable dictionaries
        - `data` is optional to support lightweight acknowledgements

    Args:
        message: Human-readable success message
        data: Optional payload data

    Returns:
        A standardized success response dictionary
    """
    
    payload: Dict[str, Any] = {
        "code": SUCCESS_CODE,
        "type": SUCCESS_TYPE,
        "message": message,
    }

    if data is not None:
        payload["data"] = data

    return payload


def build_error_response(
    *,
    code: str,
    error_type: str,
    message: str,
    details: str | None = None,
) -> Dict[str, Any]:
    """
    Build a standardized error response dictionary

    Design choice:
        - Error codes are returned as strings for consistency
        - Technical details are optional and not always exposed

    Args:
        code: HTTP-like status code as string
        error_type: Machine-readable error category
        message: Human-readable error message
        details: Optional technical details

    Returns:
        A standardized error response dictionary
    """
    
    payload: Dict[str, Any] = {
        "code": code,
        "type": error_type,
        "message": message,
    }

    if details is not None:
        payload["details"] = details

    return payload


## ============================================================
## SAFE EXECUTION WRAPPER
## ============================================================
def safe_run(
    handler: Callable[..., Dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute a business handler and normalize raised exceptions

    High-level workflow:
        1) Execute the provided handler with given arguments
        2) Catch known application errors and normalize them
        3) Catch unexpected exceptions and return a generic error

    Design choice:
        - FastAPI handles request validation (Pydantic)
        - This function focuses on business-layer robustness

    Args:
        handler: Business function returning a response dictionary
        *args: Positional arguments forwarded to the handler
        **kwargs: Keyword arguments forwarded to the handler

    Returns:
        A standardized response dictionary (success or error)
    """
    
    try:
        return handler(*args, **kwargs)

    except ValidationError as exc:
        LOGGER.warning("Validation error: %s", exc)
        return build_error_response(
            code=getattr(exc, "code", BAD_REQUEST_CODE),
            error_type=getattr(exc, "error_type", VALIDATION_ERROR_TYPE),
            message=getattr(exc, "message", str(exc)),
            details=getattr(exc, "details", None),
        )

    except ApplicationError as exc:
        LOGGER.warning("Application error: %s", exc)
        return build_error_response(
            code=getattr(exc, "code", SERVER_ERROR_CODE),
            error_type=getattr(exc, "error_type", UNKNOWN_ERROR_TYPE),
            message=getattr(exc, "message", str(exc)),
            details=getattr(exc, "details", None),
        )

    except Exception as exc:
        LOGGER.exception("Unhandled error during handler execution")
        return build_error_response(
            code=SERVER_ERROR_CODE,
            error_type=UNKNOWN_ERROR_TYPE,
            message="Internal server error",
            details=str(exc),
        )