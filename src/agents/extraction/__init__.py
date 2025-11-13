"""Extraction agents for legal document analysis."""

from src.agents.extraction.entity_extractor import (
    EntityExtractionAgent,
    EntityExtractionConfig,
    Entity,
    EntityType,
)
from src.agents.extraction.clause_extractor import (
    ClauseExtractionAgent,
    ClauseExtractionConfig,
    Clause,
    ClauseType,
)

__all__ = [
    "EntityExtractionAgent",
    "EntityExtractionConfig",
    "Entity",
    "EntityType",
    "ClauseExtractionAgent",
    "ClauseExtractionConfig",
    "Clause",
    "ClauseType",
]
