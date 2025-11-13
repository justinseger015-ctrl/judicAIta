"""
RAG (Retrieval-Augmented Generation) module for legal knowledge retrieval.

This module provides semantic search and context retrieval using ChromaDB.
"""

from src.models.rag.legal_rag import (
    LegalRAGSystem,
    RAGConfig,
    SearchResult,
    create_legal_rag,
)

__all__ = [
    "LegalRAGSystem",
    "RAGConfig",
    "SearchResult",
    "create_legal_rag",
]
