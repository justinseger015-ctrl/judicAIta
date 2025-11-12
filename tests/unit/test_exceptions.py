"""
Unit tests for exceptions module.
"""

import pytest

from judicaita.core.exceptions import (
    AuditError,
    CitationError,
    DocumentProcessingError,
    JudicaitaError,
    ModelInferenceError,
)


class TestExceptions:
    """Test suite for custom exceptions."""

    def test_base_exception(self) -> None:
        """Test JudicaitaError base exception."""
        error = JudicaitaError("Test error", error_code="TEST_ERROR")

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {}

    def test_exception_with_details(self) -> None:
        """Test exception with details."""
        details = {"file": "test.pdf", "size": 1024}
        error = DocumentProcessingError(
            "Failed to process",
            error_code="PROCESSING_FAILED",
            details=details,
        )

        assert error.details == details

    def test_exception_to_dict(self) -> None:
        """Test exception serialization to dict."""
        error = CitationError(
            "Citation not found",
            error_code="NOT_FOUND",
            details={"citation": "123 U.S. 456"},
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "NOT_FOUND"
        assert error_dict["message"] == "Citation not found"
        assert error_dict["details"]["citation"] == "123 U.S. 456"

    def test_exception_inheritance(self) -> None:
        """Test that specific exceptions inherit from base."""
        doc_error = DocumentProcessingError("test")
        model_error = ModelInferenceError("test")
        citation_error = CitationError("test")
        audit_error = AuditError("test")

        assert isinstance(doc_error, JudicaitaError)
        assert isinstance(model_error, JudicaitaError)
        assert isinstance(citation_error, JudicaitaError)
        assert isinstance(audit_error, JudicaitaError)

    def test_exception_default_error_code(self) -> None:
        """Test that exceptions use class name as default error code."""
        error = DocumentProcessingError("test")

        assert error.error_code == "DocumentProcessingError"
