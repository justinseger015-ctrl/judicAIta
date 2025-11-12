"""
Document input module for Judicaita.
"""

from judicaita.document_input.base import (
    DocumentContent,
    DocumentMetadata,
    DocumentProcessor,
)
from judicaita.document_input.pdf_processor import PDFProcessor
from judicaita.document_input.service import DocumentInputService
from judicaita.document_input.word_processor import WordProcessor

__all__ = [
    "DocumentContent",
    "DocumentMetadata",
    "DocumentProcessor",
    "PDFProcessor",
    "WordProcessor",
    "DocumentInputService",
]
