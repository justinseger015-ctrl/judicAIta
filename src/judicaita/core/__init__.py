"""
Core module for Judicaita.

This module contains core functionality including configuration, exceptions,
and base classes used throughout the application.
"""

from judicaita.core.config import Settings, get_settings
from judicaita.core.exceptions import (
    APIError,
    AuditError,
    AuditLogWriteError,
    AuthenticationError,
    CitationError,
    CitationFormatError,
    CitationNotFoundError,
    ComplianceViolationError,
    ConfigurationError,
    DocumentParsingError,
    DocumentProcessingError,
    DocumentTooLargeError,
    JudicaitaError,
    ModelInferenceError,
    ModelNotLoadedError,
    ModelTimeoutError,
    RateLimitExceededError,
    UnsupportedDocumentFormatError,
    ValidationError,
)

__all__ = [
    "Settings",
    "get_settings",
    "JudicaitaError",
    "DocumentProcessingError",
    "UnsupportedDocumentFormatError",
    "DocumentTooLargeError",
    "DocumentParsingError",
    "ModelInferenceError",
    "ModelNotLoadedError",
    "ModelTimeoutError",
    "CitationError",
    "CitationNotFoundError",
    "CitationFormatError",
    "AuditError",
    "AuditLogWriteError",
    "ComplianceViolationError",
    "ConfigurationError",
    "APIError",
    "RateLimitExceededError",
    "AuthenticationError",
    "ValidationError",
]
