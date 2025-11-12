"""
Plain-English summary generation module for Judicaita.

This module generates accessible, plain-English summaries of complex legal
documents and analyses, making legal information more accessible.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SummaryLevel(str, Enum):
    """Levels of summary detail."""

    BRIEF = "brief"  # 1-2 sentences
    SHORT = "short"  # 1 paragraph
    MEDIUM = "medium"  # 2-3 paragraphs
    DETAILED = "detailed"  # Multiple paragraphs with sections


class ReadingLevel(str, Enum):
    """Target reading level for summaries."""

    ELEMENTARY = "elementary"  # Grade 5-6
    MIDDLE_SCHOOL = "middle_school"  # Grade 7-8
    HIGH_SCHOOL = "high_school"  # Grade 9-12
    COLLEGE = "college"  # College level
    PROFESSIONAL = "professional"  # Legal professional


class SummarySection(BaseModel):
    """A section of a summary."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    key_points: List[str] = Field(
        default_factory=list, description="Key points in this section"
    )


class LegalSummary(BaseModel):
    """A plain-English legal summary."""

    original_text: str = Field(..., description="Original legal text")
    summary: str = Field(..., description="Plain-English summary")
    summary_level: SummaryLevel = Field(..., description="Level of detail")
    reading_level: ReadingLevel = Field(..., description="Target reading level")
    sections: List[SummarySection] = Field(
        default_factory=list, description="Summary sections"
    )
    key_terms: Dict[str, str] = Field(
        default_factory=dict, description="Key legal terms and their definitions"
    )
    key_takeaways: List[str] = Field(
        default_factory=list, description="Main takeaways"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        lines = [
            "# Plain-English Summary",
            f"\n**Level:** {self.summary_level.value.title()}",
            f"**Reading Level:** {self.reading_level.value.replace('_', ' ').title()}",
            "\n## Summary\n",
            self.summary,
        ]

        if self.key_takeaways:
            lines.extend([
                "\n## Key Takeaways\n",
                "\n".join(f"- {takeaway}" for takeaway in self.key_takeaways),
            ])

        if self.sections:
            lines.append("\n## Detailed Breakdown\n")
            for section in self.sections:
                lines.extend([
                    f"### {section.title}\n",
                    section.content,
                ])
                if section.key_points:
                    lines.append("\n**Key Points:**")
                    lines.extend(f"- {point}" for point in section.key_points)
                lines.append("")

        if self.key_terms:
            lines.extend([
                "\n## Key Legal Terms\n",
                "\n".join(
                    f"**{term}:** {definition}"
                    for term, definition in self.key_terms.items()
                ),
            ])

        return "\n".join(lines)
