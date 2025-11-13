"""Extraction agents for legal document analysis."""

from src.agents.extraction.entity_extractor import (
    EntityExtractionAgent,
    EntityExtractionConfig,
    Entity,
    EntityType,
)

__all__ = [
    "EntityExtractionAgent",
    "EntityExtractionConfig",
    "Entity",
    "EntityType",
]
