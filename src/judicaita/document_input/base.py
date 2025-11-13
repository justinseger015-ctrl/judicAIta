"""
Document input processing module for Judicaita.

This module handles document ingestion from various formats including PDF and Word documents.
It provides a unified interface for extracting text and metadata from legal documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""

    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Document file type")
    file_size_bytes: int = Field(..., description="File size in bytes")
    page_count: int | None = Field(None, description="Number of pages")
    author: str | None = Field(None, description="Document author")
    title: str | None = Field(None, description="Document title")
    subject: str | None = Field(None, description="Document subject")
    created_date: str | None = Field(None, description="Creation date")
    modified_date: str | None = Field(None, description="Last modified date")
    keywords: list[str] = Field(default_factory=list, description="Document keywords")
    custom_properties: dict[str, Any] = Field(
        default_factory=dict, description="Custom document properties"
    )


class DocumentContent(BaseModel):
    """Structured content extracted from a document."""

    text: str = Field(..., description="Extracted text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    sections: list[dict[str, Any]] = Field(default_factory=list, description="Document sections")
    tables: list[dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    images: list[dict[str, Any]] = Field(default_factory=list, description="Image references")
    footnotes: list[str] = Field(default_factory=list, description="Footnotes")
    citations: list[str] = Field(default_factory=list, description="Detected citations")


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    async def process(self, file_path: Path) -> DocumentContent:
        """
        Process a document and extract its content.

        Args:
            file_path: Path to the document file

        Returns:
            DocumentContent: Extracted document content

        Raises:
            DocumentProcessingError: If processing fails
        """
        pass

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """
        Check if this processor supports the given file type.

        Args:
            file_type: File type/extension to check

        Returns:
            bool: True if supported, False otherwise
        """
        pass

    def _validate_file(self, file_path: Path, max_size_bytes: int) -> None:
        """
        Validate that the file exists and is within size limits.

        Args:
            file_path: Path to the file
            max_size_bytes: Maximum allowed file size

        Raises:
            FileNotFoundError: If file doesn't exist
            DocumentTooLargeError: If file exceeds size limit
        """
        from judicaita.core.exceptions import DocumentTooLargeError

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            raise DocumentTooLargeError(
                f"File size ({file_size} bytes) exceeds maximum allowed size "
                f"({max_size_bytes} bytes)",
                details={"file_size": file_size, "max_size": max_size_bytes},
            )
