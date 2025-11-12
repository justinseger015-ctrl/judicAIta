"""
Unit tests for configuration module.
"""

import pytest
from pydantic import ValidationError

from judicaita.core.config import Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_default_settings(self) -> None:
        """Test that default settings are valid."""
        # Use test key to avoid validation error
        settings = Settings(google_api_key="test-key")

        assert settings.app_name == "Judicaita"
        assert settings.app_version == "0.1.0"
        assert settings.model_temperature == 0.3
        assert settings.audit_log_enabled is True

    def test_custom_settings(self) -> None:
        """Test custom settings override defaults."""
        settings = Settings(
            google_api_key="test-key",
            app_name="CustomApp",
            model_temperature=0.5,
        )

        assert settings.app_name == "CustomApp"
        assert settings.model_temperature == 0.5

    def test_temperature_validation(self) -> None:
        """Test that temperature is validated."""
        # Valid temperature
        settings = Settings(google_api_key="test-key", model_temperature=0.5)
        assert settings.model_temperature == 0.5

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            Settings(google_api_key="test-key", model_temperature=3.0)

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            Settings(google_api_key="test-key", model_temperature=-0.1)

    def test_max_tokens_validation(self) -> None:
        """Test that max tokens is validated."""
        # Valid
        settings = Settings(google_api_key="test-key", model_max_tokens=1024)
        assert settings.model_max_tokens == 1024

        # Invalid (too high)
        with pytest.raises(ValidationError):
            Settings(google_api_key="test-key", model_max_tokens=10000)

        # Invalid (too low)
        with pytest.raises(ValidationError):
            Settings(google_api_key="test-key", model_max_tokens=0)

    def test_max_document_size_bytes(self) -> None:
        """Test calculation of max document size in bytes."""
        settings = Settings(
            google_api_key="test-key",
            max_document_size_mb=10,
        )

        expected_bytes = 10 * 1024 * 1024
        assert settings.get_max_document_size_bytes() == expected_bytes

    def test_supported_formats(self) -> None:
        """Test supported formats list."""
        settings = Settings(google_api_key="test-key")

        assert "pdf" in settings.supported_formats
        assert "docx" in settings.supported_formats

    def test_feature_flags(self) -> None:
        """Test feature flags."""
        settings = Settings(
            google_api_key="test-key",
            enable_reasoning_trace=False,
            enable_citation_mapping=False,
        )

        assert settings.enable_reasoning_trace is False
        assert settings.enable_citation_mapping is False
        assert settings.enable_plain_english_summary is True  # Default
