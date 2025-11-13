"""
RAG (Retrieval-Augmented Generation) module for legal knowledge retrieval.

This module provides semantic search and context retrieval using ChromaDB.
"""

from src.models.rag.hybrid_rag import (
    ChunkingStrategy,
    HybridRAGConfig,
    HybridRAGSystem,
    HybridSearchResult,
    QueryAnalytics,
    SearchStrategy,
    create_production_hybrid_rag,
)
from src.models.rag.legal_rag import (
    LegalRAGSystem,
    RAGConfig,
    SearchResult,
    create_legal_rag,
)

__all__ = [
    # Base RAG
    "LegalRAGSystem",
    "RAGConfig",
    "SearchResult",
    "create_legal_rag",
    # Hybrid RAG
    "ChunkingStrategy",
    "HybridRAGConfig",
    "HybridRAGSystem",
    "HybridSearchResult",
    "QueryAnalytics",
    "SearchStrategy",
    "create_production_hybrid_rag",
]
