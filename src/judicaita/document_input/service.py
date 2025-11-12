"""
Document input module for Judicaita.

This module provides a unified interface for processing various document formats
including PDF and Word documents.
"""

from pathlib import Path

from loguru import logger

from judicaita.core.exceptions import UnsupportedDocumentFormatError
from judicaita.document_input.base import DocumentContent, DocumentProcessor
from judicaita.document_input.pdf_processor import PDFProcessor
from judicaita.document_input.word_processor import WordProcessor


class DocumentInputService:
    """
    Service for processing documents from various formats.

    This service automatically selects the appropriate processor based on file type.
    """

    def __init__(self, max_size_bytes: int = 50 * 1024 * 1024) -> None:
        """
        Initialize the document input service.

        Args:
            max_size_bytes: Maximum allowed file size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self._processors: dict[str, DocumentProcessor] = {
            "pdf": PDFProcessor(max_size_bytes),
            "docx": WordProcessor(max_size_bytes),
            "doc": WordProcessor(max_size_bytes),
        }

    def register_processor(self, file_type: str, processor: DocumentProcessor) -> None:
        """
        Register a custom document processor.

        Args:
            file_type: File type/extension to handle
            processor: Processor instance
        """
        self._processors[file_type.lower().lstrip(".")] = processor
        logger.info(f"Registered processor for {file_type}")

    async def process_document(self, file_path: Path) -> DocumentContent:
        """
        Process a document and extract its content.

        Args:
            file_path: Path to the document file

        Returns:
            DocumentContent: Extracted document content

        Raises:
            UnsupportedDocumentFormatError: If file format is not supported
            DocumentProcessingError: If processing fails
        """
        # Determine file type
        file_extension = file_path.suffix.lower().lstrip(".")

        # Get appropriate processor
        processor = self._processors.get(file_extension)

        if not processor:
            raise UnsupportedDocumentFormatError(
                f"Unsupported document format: {file_extension}",
                details={
                    "file": str(file_path),
                    "extension": file_extension,
                    "supported_formats": list(self._processors.keys()),
                },
            )

        logger.info(f"Processing document: {file_path} using {processor.__class__.__name__}")

        # Process document
        return await processor.process(file_path)

    def supports_format(self, file_type: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            file_type: File type/extension to check

        Returns:
            bool: True if supported, False otherwise
        """
        return file_type.lower().lstrip(".") in self._processors

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported file formats.

        Returns:
            list: List of supported file extensions
        """
        return list(self._processors.keys())
