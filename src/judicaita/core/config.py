"""
Configuration management for Judicaita using Pydantic Settings.

This module provides type-safe configuration management with environment variable
support and validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Settings
    app_name: str = Field(default="Judicaita", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")

    # Google Tunix & Gemma Configuration
    google_api_key: str = Field(default="", description="Google API key for Tunix and Gemma")
    gemma_model_name: str = Field(default="gemma-3n", description="Gemma model name")
    tunix_endpoint: str = Field(
        default="https://api.tunix.google.com/v1", description="Tunix API endpoint"
    )

    # Model Configuration
    model_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Model temperature")
    model_max_tokens: int = Field(
        default=512, ge=1, le=1000, description="Maximum tokens for generation (competition limit: 1000)"
    )
    model_top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")

    # Document Processing
    max_document_size_mb: int = Field(
        default=50, ge=1, le=500, description="Maximum document size in MB"
    )
    supported_formats: list[str] = Field(
        default=["pdf", "docx", "doc", "txt"], description="Supported document formats"
    )
    ocr_enabled: bool = Field(default=False, description="Enable OCR for scanned documents")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, le=32, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Enable auto-reload in development")

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./judicaita.db", description="Database connection URL"
    )

    # Cache Configuration
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Citation Database
    citation_database_path: Path = Field(
        default=Path("./data/legal_citations.db"),
        description="Path to citation database",
    )
    legal_jurisdiction: str = Field(
        default="US", description="Legal jurisdiction (US, UK, EU, etc.)"
    )

    # Compliance & Audit
    audit_log_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_path: Path = Field(
        default=Path("./logs/audit.log"), description="Path to audit log file"
    )
    compliance_mode: Literal["STRICT", "MODERATE", "PERMISSIVE"] = Field(
        default="STRICT", description="Compliance mode"
    )
    retention_days: int = Field(
        default=2555, ge=0, description="Data retention period in days (7 years default)"
    )

    # Security
    secret_key: str = Field(default="change-me-in-production", description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT expiration in hours")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Maximum requests per window")
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, description="Rate limit window in seconds"
    )

    # Feature Flags
    enable_reasoning_trace: bool = Field(
        default=True, description="Enable reasoning trace generation"
    )
    enable_citation_mapping: bool = Field(default=True, description="Enable citation mapping")
    enable_plain_english_summary: bool = Field(
        default=True, description="Enable plain English summaries"
    )
    enable_compliance_audit: bool = Field(default=True, description="Enable compliance auditing")
    experimental_features: bool = Field(default=False, description="Enable experimental features")

    # GRPO Training Configuration
    grpo_checkpoint_path: str = Field(
        default="", description="Path to GRPO-tuned checkpoint for reasoning"
    )
    grpo_base_model: str = Field(
        default="google/gemma-2-2b-it", description="Base model for GRPO training"
    )
    grpo_learning_rate: float = Field(
        default=1e-5, ge=1e-7, le=1e-3, description="GRPO training learning rate"
    )
    grpo_num_epochs: int = Field(
        default=3, ge=1, le=20, description="Number of GRPO training epochs"
    )
    grpo_batch_size: int = Field(default=4, ge=1, le=32, description="GRPO training batch size")
    grpo_use_lora: bool = Field(
        default=True, description="Use LoRA for parameter-efficient GRPO training"
    )

    from pydantic import model_validator

    @model_validator(mode="after")
    def check_google_api_key(self):
        """Validate that API key is set in production."""
        if not self.google_api_key and not self.debug:
            raise ValueError("google_api_key must be set in production")
        return self

    @field_validator("audit_log_path", "citation_database_path")
    @classmethod
    def create_parent_dirs(cls, v: Path) -> Path:
        """Create parent directories if they don't exist."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    def get_max_document_size_bytes(self) -> int:
        """Get maximum document size in bytes."""
        return self.max_document_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()
