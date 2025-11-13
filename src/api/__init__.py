"""
API module for JudicAIta platform.

This module provides FastAPI endpoints for health checks and production services.
"""

from src.api.health import (
    DependencyHealth,
    DependencyType,
    HealthCheckResponse,
    HealthChecker,
    HealthStatus,
    create_health_checker,
)

__all__ = [
    "DependencyHealth",
    "DependencyType",
    "HealthCheckResponse",
    "HealthChecker",
    "HealthStatus",
    "create_health_checker",
]
