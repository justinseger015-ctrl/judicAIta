"""
PDF document processor using pdfplumber and pypdf.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from judicaita.core.exceptions import DocumentParsingError, DocumentProcessingError
from judicaita.document_input.base import (
    DocumentContent,
    DocumentMetadata,
    DocumentProcessor,
)


class PDFProcessor(DocumentProcessor):
    """PDF document processor using pdfplumber."""

    def __init__(self, max_size_bytes: int = 50 * 1024 * 1024) -> None:
        """
        Initialize the PDF processor.

        Args:
            max_size_bytes: Maximum allowed file size in bytes
        """
        self.max_size_bytes = max_size_bytes

    def supports(self, file_type: str) -> bool:
        """Check if PDF format is supported."""
        return file_type.lower() in ["pdf", ".pdf"]

    async def process(self, file_path: Path) -> DocumentContent:
        """
        Process a PDF document and extract its content.

        Args:
            file_path: Path to the PDF file

        Returns:
            DocumentContent: Extracted document content

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            import pdfplumber
            import pypdf

            # Validate file
            self._validate_file(file_path, self.max_size_bytes)

            logger.info(f"Processing PDF document: {file_path}")

            # Extract metadata using pypdf
            metadata = await self._extract_metadata(file_path)

            # Extract content using pdfplumber
            text_content: list[str] = []
            sections: list[dict[str, Any]] = []
            tables: list[dict[str, Any]] = []
            images: list[dict[str, Any]] = []

            with pdfplumber.open(file_path) as pdf:
                metadata.page_count = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_content.append(page_text)
                        sections.append(
                            {
                                "page": page_num,
                                "text": page_text,
                                "type": "page",
                            }
                        )

                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table:
                            tables.append(
                                {
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "data": table,
                                }
                            )

                    # Track images
                    if hasattr(page, "images") and page.images:
                        for img_idx, img in enumerate(page.images):
                            images.append(
                                {
                                    "page": page_num,
                                    "image_index": img_idx,
                                    "bbox": img.get("bbox", []),
                                }
                            )

            full_text = "\n\n".join(text_content)

            # Extract citations (basic pattern matching)
            citations = self._extract_citations(full_text)

            return DocumentContent(
                text=full_text,
                metadata=metadata,
                sections=sections,
                tables=tables,
                images=images,
                citations=citations,
            )

        except ImportError as e:
            raise DocumentProcessingError(
                f"Required PDF processing library not installed: {e}",
                error_code="MISSING_DEPENDENCY",
            )
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise DocumentParsingError(
                f"Failed to parse PDF document: {e}",
                details={"file": str(file_path), "error": str(e)},
            )

    async def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from PDF using pypdf."""
        import pypdf

        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                info = reader.metadata or {}

                return DocumentMetadata(
                    filename=file_path.name,
                    file_type="pdf",
                    file_size_bytes=file_path.stat().st_size,
                    page_count=len(reader.pages),
                    author=info.get("/Author"),
                    title=info.get("/Title"),
                    subject=info.get("/Subject"),
                    created_date=info.get("/CreationDate"),
                    modified_date=info.get("/ModDate"),
                    keywords=[k.strip() for k in info.get("/Keywords", "").split(",") if k.strip()],
                )
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            return DocumentMetadata(
                filename=file_path.name,
                file_type="pdf",
                file_size_bytes=file_path.stat().st_size,
            )

    def _extract_citations(self, text: str) -> list[str]:
        """
        Extract legal citations from text using regex patterns.

        This is a simplified implementation. In production, use a specialized
        legal citation parser.
        """
        import re

        citations: list[str] = []

        # Pattern for US case citations (e.g., "Brown v. Board of Education, 347 U.S. 483")
        case_pattern = r"\b\d+\s+[A-Z]\.[A-Za-z\.]+\s+\d+\b"
        citations.extend(re.findall(case_pattern, text))

        # Pattern for statute citations (e.g., "42 U.S.C. § 1983")
        statute_pattern = r"\b\d+\s+U\.S\.C\.\s+§\s*\d+\b"
        citations.extend(re.findall(statute_pattern, text))

        return list(set(citations))  # Remove duplicates
