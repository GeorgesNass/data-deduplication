'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Custom exception hierarchy for the deduplication pipeline and API layer."
'''

from __future__ import annotations

## Standard library imports
from typing import Any, Dict, Optional

## ============================================================
## BASE EXCEPTIONS
## ============================================================
class ApplicationError(Exception):
    """
        Base class for all application-specific errors

        Design:
            - Centralizes error typing
            - Enables uniform JSON serialization for API responses
            - Keeps optional context payload for debugging
    """

    def __init__(
        self,
        message: str,
        code: str = "500",
        error_type: str = "APPLICATION_ERROR",
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.error_type = error_type
        self.details = details
        self.context = context
        super().__init__(message)

    def __str__(self) -> str:
        """
            String representation for logging

            Returns:
                Human-readable error string
        """
        
        return f"{self.error_type}({self.code}): {self.message}"

    def __repr__(self) -> str:
        """
            Debug representation

            Returns:
                Debug string
        """
        
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, error_type={self.error_type!r}, "
            f"message={self.message!r}, details={self.details!r})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
            Serialize error to dictionary for API responses

            Returns:
                A JSON-serializable dictionary
        """
        
        payload: Dict[str, Any] = {
            "code": self.code,
            "type": self.error_type,
            "message": self.message,
        }

        if self.details is not None:
            payload["details"] = self.details

        if self.context is not None:
            payload["context"] = self.context

        return payload

## ============================================================
## CONFIGURATION ERRORS
## ============================================================
class ConfigurationError(ApplicationError):
    """
        Raised when configuration files or environment variables are invalid
    """

    def __init__(
        self,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="500",
            error_type="CONFIGURATION_ERROR",
            details=details,
            context=context,
        )

## ============================================================
## VALIDATION ERRORS
## ============================================================
class ValidationError(ApplicationError):
    """
        Raised when input data validation fails
    """

    def __init__(
        self,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="400",
            error_type="VALIDATION_ERROR",
            details=details,
            context=context,
        )

class SchemaValidationError(ValidationError):
    """
        Raised when a payload does not conform to the expected schema
    """

## ============================================================
## PIPELINE ERRORS
## ============================================================
class PipelineError(ApplicationError):
    """
        Raised when a pipeline execution step fails
    """

    def __init__(
        self,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="500",
            error_type="PIPELINE_ERROR",
            details=details,
            context=context,
        )

class StepExecutionError(PipelineError):
    """
        Raised when a specific pipeline step fails
    """

    def __init__(
        self,
        step_name: str,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        full_message = f"Step '{step_name}' failed: {message}"
        super().__init__(
            message=full_message,
            details=details,
            context=context,
        )

## ============================================================
## MODEL / DEDUPLICATION ERRORS
## ============================================================
class DeduplicationError(ApplicationError):
    """
        Raised when deduplication or fuzzy matching logic fails
    """

    def __init__(
        self,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="500",
            error_type="DEDUPLICATION_ERROR",
            details=details,
            context=context,
        )

class ActiveLearningError(ApplicationError):
    """
        Raised when active learning state or persistence fails
    """

    def __init__(
        self,
        message: str,
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="500",
            error_type="ACTIVE_LEARNING_ERROR",
            details=details,
            context=context,
        )

## ============================================================
## API ERRORS
## ============================================================
class ApiError(ApplicationError):
    """
        Raised for API-level errors not tied to validation
    """

    def __init__(
        self,
        message: str,
        code: str = "500",
        details: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            error_type="API_ERROR",
            details=details,
            context=context,
        )