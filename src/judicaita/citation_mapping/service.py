"""
Citation mapping service for validating and enriching citations.
"""


from loguru import logger

from judicaita.citation_mapping.models import Citation, CitationGraph, CitationMatch
from judicaita.citation_mapping.parser import CitationParser
from judicaita.core.config import get_settings
from judicaita.core.exceptions import CitationError


class CitationMappingService:
    """
    Service for mapping, validating, and enriching legal citations.

    This service extracts citations from text, validates them against
    legal databases, and creates citation graphs showing relationships.
    """

    def __init__(self) -> None:
        """Initialize the citation mapping service."""
        self.settings = get_settings()
        self.parser = CitationParser(jurisdiction=self.settings.legal_jurisdiction)
        self._citation_cache: dict[str, Citation] = {}

    async def extract_and_map_citations(
        self, text: str, validate: bool = True
    ) -> list[CitationMatch]:
        """
        Extract and map citations from text.

        Args:
            text: Text containing legal citations
            validate: Whether to validate citations against databases

        Returns:
            List of citation matches with enriched data

        Raises:
            CitationError: If citation processing fails
        """
        try:
            logger.info("Extracting citations from text")

            # Extract citations
            citation_matches = self.parser.extract_citations(text)

            # Validate and enrich if requested
            if validate and self.settings.enable_citation_mapping:
                for match in citation_matches:
                    await self._validate_and_enrich_citation(match.citation)

            logger.info(f"Successfully mapped {len(citation_matches)} citations")
            return citation_matches

        except Exception as e:
            logger.error(f"Failed to extract and map citations: {e}")
            raise CitationError(
                f"Citation mapping failed: {e}",
                details={"error": str(e)},
            )

    async def _validate_and_enrich_citation(self, citation: Citation) -> None:
        """
        Validate and enrich a citation with additional information.

        Args:
            citation: Citation to validate and enrich
        """
        # Check cache first
        if citation.raw_citation in self._citation_cache:
            cached = self._citation_cache[citation.raw_citation]
            citation.is_valid = cached.is_valid
            citation.url = cached.url
            citation.metadata = cached.metadata
            return

        try:
            # TODO: Integrate with legal citation databases
            # For now, mark as valid if it matches expected patterns
            citation.is_valid = True

            # Add placeholder URL
            citation.url = f"https://example.com/citation/{citation.raw_citation}"

            # Cache the enriched citation
            self._citation_cache[citation.raw_citation] = citation

            logger.debug(f"Validated citation: {citation.raw_citation}")

        except Exception as e:
            logger.warning(f"Failed to validate citation {citation.raw_citation}: {e}")
            citation.is_valid = False

    async def build_citation_graph(self, citations: list[Citation]) -> CitationGraph:
        """
        Build a citation graph showing relationships between citations.

        Args:
            citations: List of citations to analyze

        Returns:
            CitationGraph: Graph of citation relationships
        """
        logger.info(f"Building citation graph for {len(citations)} citations")

        graph = CitationGraph()

        # Add all citations as nodes
        for citation in citations:
            graph.add_citation(citation)

        # TODO: Analyze and add relationships
        # This would involve:
        # - Finding citing/cited relationships
        # - Identifying overruled/distinguished cases
        # - Tracking statutory amendments

        logger.info(
            f"Built citation graph with {len(graph.nodes)} nodes and " f"{len(graph.edges)} edges"
        )
        return graph

    async def validate_citation(self, citation_str: str) -> Citation | None:
        """
        Validate a single citation string.

        Args:
            citation_str: Citation string to validate

        Returns:
            Validated Citation object or None if invalid
        """
        try:
            citation = self.parser.parse_single_citation(citation_str)

            if citation:
                await self._validate_and_enrich_citation(citation)
                return citation

            return None

        except Exception as e:
            logger.error(f"Failed to validate citation '{citation_str}': {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the citation cache."""
        self._citation_cache.clear()
        logger.info("Citation cache cleared")
