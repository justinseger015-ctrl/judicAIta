"""
Unit tests for citation parser.
"""

import pytest

from judicaita.citation_mapping.parser import CitationParser
from judicaita.citation_mapping.models import CitationType, Jurisdiction


class TestCitationParser:
    """Test suite for CitationParser."""

    def test_extract_us_case_citation(self, sample_citation_text: str) -> None:
        """Test extraction of US case citations."""
        parser = CitationParser()
        citations = parser.extract_citations(sample_citation_text)

        # Should find at least the case citations
        assert len(citations) >= 2

        # Check for Brown v. Board of Education
        brown_citations = [
            c for c in citations
            if "Brown v. Board of Education" in c.citation.raw_citation
        ]
        assert len(brown_citations) > 0
        assert brown_citations[0].citation.citation_type == CitationType.CASE

    def test_extract_statute_citation(self, sample_citation_text: str) -> None:
        """Test extraction of statute citations."""
        parser = CitationParser()
        citations = parser.extract_citations(sample_citation_text)

        # Check for statute citations
        statute_citations = [
            c for c in citations
            if c.citation.citation_type == CitationType.STATUTE
        ]
        assert len(statute_citations) >= 2

    def test_parse_single_citation(self) -> None:
        """Test parsing a single citation."""
        parser = CitationParser()
        citation = parser.parse_single_citation("347 U.S. 483")

        assert citation is not None
        assert citation.volume == "347"
        assert citation.reporter == "U.S."
        assert citation.page == "483"
        assert citation.citation_type == CitationType.CASE

    def test_context_extraction(self, sample_citation_text: str) -> None:
        """Test that context is extracted around citations."""
        parser = CitationParser()
        citations = parser.extract_citations(sample_citation_text)

        assert len(citations) > 0
        # Context should be non-empty
        assert len(citations[0].context) > 0
        # Context should contain surrounding text
        assert citations[0].citation.raw_citation in citations[0].context

    def test_no_citations_in_plain_text(self) -> None:
        """Test that plain text without citations returns empty list."""
        parser = CitationParser()
        text = "This is just plain text with no legal citations."
        citations = parser.extract_citations(text)

        assert len(citations) == 0

    def test_jurisdiction_inference(self) -> None:
        """Test jurisdiction inference from reporter."""
        parser = CitationParser()

        # US Supreme Court
        citation = parser.parse_single_citation("347 U.S. 483")
        assert citation.jurisdiction == Jurisdiction.US_FEDERAL

        # Federal courts
        citation = parser.parse_single_citation("123 F.3d 456")
        assert citation.jurisdiction == Jurisdiction.US_FEDERAL
