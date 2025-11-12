"""
Legal citation mapping and verification module for Judicaita.

This module provides functionality for extracting, parsing, validating,
and mapping legal citations to their sources.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CitationType(str, Enum):
    """Types of legal citations."""

    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL = "constitutional"
    TREATY = "treaty"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class Jurisdiction(str, Enum):
    """Legal jurisdictions."""

    US_FEDERAL = "us_federal"
    US_STATE = "us_state"
    UK = "uk"
    EU = "eu"
    INTERNATIONAL = "international"
    OTHER = "other"


class Citation(BaseModel):
    """Structured legal citation."""

    raw_citation: str = Field(..., description="Original citation text")
    citation_type: CitationType = Field(..., description="Type of citation")
    jurisdiction: Jurisdiction = Field(..., description="Legal jurisdiction")
    case_name: Optional[str] = Field(None, description="Case name (for case citations)")
    volume: Optional[str] = Field(None, description="Volume number")
    reporter: Optional[str] = Field(None, description="Reporter abbreviation")
    page: Optional[str] = Field(None, description="Page number")
    year: Optional[int] = Field(None, description="Year of decision/enactment")
    court: Optional[str] = Field(None, description="Court name")
    statute_section: Optional[str] = Field(None, description="Statute section")
    is_valid: bool = Field(default=False, description="Whether citation is validated")
    url: Optional[str] = Field(None, description="URL to full text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CitationMatch(BaseModel):
    """A citation match with context."""

    citation: Citation = Field(..., description="The citation object")
    context: str = Field(..., description="Surrounding text context")
    start_pos: int = Field(..., description="Start position in document")
    end_pos: int = Field(..., description="End position in document")
    relevance_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relevance to query"
    )


class CitationGraph(BaseModel):
    """Graph of citation relationships."""

    nodes: List[Citation] = Field(default_factory=list, description="Citation nodes")
    edges: List[Dict[str, Any]] = Field(
        default_factory=list, description="Citation relationships"
    )

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the graph."""
        if citation not in self.nodes:
            self.nodes.append(citation)

    def add_relationship(
        self,
        source: Citation,
        target: Citation,
        relationship_type: str,
    ) -> None:
        """Add a relationship between citations."""
        self.add_citation(source)
        self.add_citation(target)

        edge = {
            "source": source.raw_citation,
            "target": target.raw_citation,
            "type": relationship_type,
        }

        if edge not in self.edges:
            self.edges.append(edge)

    def get_related_citations(self, citation: Citation) -> List[Citation]:
        """Get all citations related to a given citation."""
        related = []
        citation_str = citation.raw_citation

        for edge in self.edges:
            if edge["source"] == citation_str:
                # Find target citation
                for node in self.nodes:
                    if node.raw_citation == edge["target"]:
                        related.append(node)
            elif edge["target"] == citation_str:
                # Find source citation
                for node in self.nodes:
                    if node.raw_citation == edge["source"]:
                        related.append(node)

        return related
