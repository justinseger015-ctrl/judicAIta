"""
Custom exceptions for Judicaita.

This module defines a hierarchy of exceptions used throughout the application
to provide clear error handling and debugging information.
"""

from typing import Any, Dict, Optional


class JudicaitaError(Exception):
    """Base exception for all Judicaita errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class DocumentProcessingError(JudicaitaError):
    """Raised when document processing fails."""

    pass


class UnsupportedDocumentFormatError(DocumentProcessingError):
    """Raised when document format is not supported."""

    pass


class DocumentTooLargeError(DocumentProcessingError):
    """Raised when document exceeds size limits."""

    pass


class DocumentParsingError(DocumentProcessingError):
    """Raised when document parsing fails."""

    pass


class ModelInferenceError(JudicaitaError):
    """Raised when model inference fails."""

    pass


class ModelNotLoadedError(ModelInferenceError):
    """Raised when model is not properly loaded."""

    pass


class ModelTimeoutError(ModelInferenceError):
    """Raised when model inference times out."""

    pass


class CitationError(JudicaitaError):
    """Raised when citation processing fails."""

    pass


class CitationNotFoundError(CitationError):
    """Raised when a citation cannot be found."""

    pass


class CitationFormatError(CitationError):
    """Raised when citation format is invalid."""

    pass


class AuditError(JudicaitaError):
    """Raised when audit logging fails."""

    pass


class AuditLogWriteError(AuditError):
    """Raised when writing to audit log fails."""

    pass


class ComplianceViolationError(AuditError):
    """Raised when a compliance violation is detected."""

    pass


class ConfigurationError(JudicaitaError):
    """Raised when configuration is invalid."""

    pass


class APIError(JudicaitaError):
    """Raised when an API call fails."""

    pass


class RateLimitExceededError(APIError):
    """Raised when rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    pass


class ValidationError(JudicaitaError):
    """Raised when input validation fails."""

    pass
