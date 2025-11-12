"""
Citation parser for extracting and parsing legal citations.
"""

import re
from typing import List, Optional, Tuple

from loguru import logger

from judicaita.citation_mapping.models import (
    Citation,
    CitationMatch,
    CitationType,
    Jurisdiction,
)


class CitationParser:
    """
    Parser for extracting and parsing legal citations from text.

    This is a simplified implementation. For production use, consider
    integrating specialized legal citation parsers like eyecite or similar.
    """

    # US case citation patterns
    US_CASE_PATTERNS = [
        # e.g., "Brown v. Board of Education, 347 U.S. 483"
        r'([A-Z][A-Za-z\s\.]+\s+v\.\s+[A-Z][A-Za-z\s\.]+),\s*(\d+)\s+([A-Z]\.[A-Za-z\.]+)\s+(\d+)',
        # e.g., "123 F.3d 456"
        r'(\d+)\s+([A-Z]\.[A-Za-z\.\d]+)\s+(\d+)',
    ]

    # US statute citation patterns
    US_STATUTE_PATTERNS = [
        # e.g., "42 U.S.C. § 1983"
        r'(\d+)\s+U\.S\.C\.\s+§\s*(\d+[a-z]*)',
        # e.g., "18 U.S.C. 1001"
        r'(\d+)\s+U\.S\.C\.\s+(\d+[a-z]*)',
    ]

    def __init__(self, jurisdiction: str = "US") -> None:
        """
        Initialize the citation parser.

        Args:
            jurisdiction: Default jurisdiction to assume
        """
        self.jurisdiction = jurisdiction

    def extract_citations(self, text: str) -> List[CitationMatch]:
        """
        Extract all citations from text.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation matches with context
        """
        citations: List[CitationMatch] = []

        # Extract case citations
        citations.extend(self._extract_case_citations(text))

        # Extract statute citations
        citations.extend(self._extract_statute_citations(text))

        logger.info(f"Extracted {len(citations)} citations from text")
        return citations

    def _extract_case_citations(self, text: str) -> List[CitationMatch]:
        """Extract case citations from text."""
        citations: List[CitationMatch] = []

        for pattern in self.US_CASE_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    citation = self._parse_case_citation(match, text)
                    if citation:
                        citations.append(citation)
                except Exception as e:
                    logger.warning(f"Failed to parse case citation: {e}")

        return citations

    def _extract_statute_citations(self, text: str) -> List[CitationMatch]:
        """Extract statute citations from text."""
        citations: List[CitationMatch] = []

        for pattern in self.US_STATUTE_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    citation = self._parse_statute_citation(match, text)
                    if citation:
                        citations.append(citation)
                except Exception as e:
                    logger.warning(f"Failed to parse statute citation: {e}")

        return citations

    def _parse_case_citation(
        self, match: re.Match, text: str
    ) -> Optional[CitationMatch]:
        """Parse a case citation match."""
        raw_citation = match.group(0)
        groups = match.groups()

        # Extract case name if present (first group if it contains "v.")
        case_name = None
        volume = None
        reporter = None
        page = None

        if len(groups) >= 4 and " v. " in groups[0]:
            case_name, volume, reporter, page = groups[:4]
        elif len(groups) >= 3:
            volume, reporter, page = groups[:3]

        citation = Citation(
            raw_citation=raw_citation,
            citation_type=CitationType.CASE,
            jurisdiction=self._infer_jurisdiction(reporter or ""),
            case_name=case_name,
            volume=volume,
            reporter=reporter,
            page=page,
        )

        # Get surrounding context
        context = self._get_context(text, match.start(), match.end())

        return CitationMatch(
            citation=citation,
            context=context,
            start_pos=match.start(),
            end_pos=match.end(),
        )

    def _parse_statute_citation(
        self, match: re.Match, text: str
    ) -> Optional[CitationMatch]:
        """Parse a statute citation match."""
        raw_citation = match.group(0)
        groups = match.groups()

        title = groups[0] if len(groups) > 0 else None
        section = groups[1] if len(groups) > 1 else None

        citation = Citation(
            raw_citation=raw_citation,
            citation_type=CitationType.STATUTE,
            jurisdiction=Jurisdiction.US_FEDERAL,
            volume=title,
            statute_section=section,
        )

        # Get surrounding context
        context = self._get_context(text, match.start(), match.end())

        return CitationMatch(
            citation=citation,
            context=context,
            start_pos=match.start(),
            end_pos=match.end(),
        )

    def _infer_jurisdiction(self, reporter: str) -> Jurisdiction:
        """Infer jurisdiction from reporter abbreviation."""
        reporter_lower = reporter.lower()

        if "u.s." in reporter_lower or "sup. ct." in reporter_lower:
            return Jurisdiction.US_FEDERAL
        elif "f." in reporter_lower or "f.2d" in reporter_lower or "f.3d" in reporter_lower:
            return Jurisdiction.US_FEDERAL
        else:
            return Jurisdiction.US_STATE

    def _get_context(
        self, text: str, start: int, end: int, context_chars: int = 100
    ) -> str:
        """Get surrounding context for a citation."""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        context = text[context_start:context_end].strip()

        # Clean up context
        context = " ".join(context.split())

        return context

    def parse_single_citation(self, citation_str: str) -> Optional[Citation]:
        """
        Parse a single citation string.

        Args:
            citation_str: Citation string to parse

        Returns:
            Parsed Citation object or None if parsing fails
        """
        matches = self.extract_citations(citation_str)

        if matches:
            return matches[0].citation

        return None
