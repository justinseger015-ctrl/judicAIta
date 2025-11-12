"""
Judicaita: An Explainable Legal AI Assistant

This package provides a comprehensive legal AI assistant built with Google Tunix
and Gemma 3n, designed for lawyers and paralegals participating in Kaggle hackathons.

Key Features:
- Document Input Processing (PDF, Word, and more)
- Reasoning Trace Generation with explainability
- Legal Citation Mapping and verification
- Plain-English Summaries of complex legal documents
- Compliance Audit Logs for transparency and accountability

Author: Judicaita Team
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Judicaita Team"
__license__ = "Apache 2.0"

from judicaita.core.config import Settings, get_settings
from judicaita.core.exceptions import (
    JudicaitaError,
    DocumentProcessingError,
    ModelInferenceError,
    CitationError,
    AuditError,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Settings",
    "get_settings",
    "JudicaitaError",
    "DocumentProcessingError",
    "ModelInferenceError",
    "CitationError",
    "AuditError",
]
