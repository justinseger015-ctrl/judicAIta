"""
Reasoning trace generation module for Judicaita.

This module generates explainable reasoning traces for legal analysis,
showing step-by-step how the AI arrives at its conclusions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReasoningStepType(str, Enum):
    """Types of reasoning steps."""

    ANALYSIS = "analysis"
    INFERENCE = "inference"
    CITATION_LOOKUP = "citation_lookup"
    COMPARISON = "comparison"
    CONCLUSION = "conclusion"
    CLARIFICATION = "clarification"


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace."""

    step_id: str = Field(..., description="Unique identifier for this step")
    step_type: ReasoningStepType = Field(..., description="Type of reasoning step")
    description: str = Field(..., description="Description of this reasoning step")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input data for this step")
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Output data from this step"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for this step"
    )
    sources: list[str] = Field(default_factory=list, description="Sources used in this step")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this step was executed"
    )
    parent_step_id: str | None = Field(
        None, description="ID of parent step (for nested reasoning)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReasoningTrace(BaseModel):
    """Complete reasoning trace for a legal analysis."""

    trace_id: str = Field(..., description="Unique identifier for this trace")
    query: str = Field(..., description="Original query or question")
    steps: list[ReasoningStep] = Field(
        default_factory=list, description="Ordered list of reasoning steps"
    )
    final_conclusion: str = Field(..., description="Final conclusion or answer")
    overall_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall confidence in conclusion"
    )
    citations_used: list[str] = Field(default_factory=list, description="All citations referenced")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When trace was created"
    )
    model_info: dict[str, Any] = Field(
        default_factory=dict, description="Information about the model used"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace."""
        self.steps.append(step)

    def get_step(self, step_id: str) -> ReasoningStep | None:
        """Get a specific step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: ReasoningStepType) -> list[ReasoningStep]:
        """Get all steps of a specific type."""
        return [step for step in self.steps if step.step_type == step_type]

    def to_markdown(self) -> str:
        """Convert the reasoning trace to markdown format."""
        lines = [
            f"# Reasoning Trace: {self.trace_id}",
            f"\n**Query:** {self.query}",
            f"\n**Created:** {self.created_at.isoformat()}",
            f"\n**Overall Confidence:** {self.overall_confidence:.2%}",
            "\n## Reasoning Steps\n",
        ]

        for idx, step in enumerate(self.steps, 1):
            lines.extend(
                [
                    f"### Step {idx}: {step.step_type.value.title()}",
                    f"\n**Description:** {step.description}",
                    f"\n**Confidence:** {step.confidence_score:.2%}",
                ]
            )

            if step.sources:
                lines.append(f"\n**Sources:** {', '.join(step.sources)}")

            if step.output_data:
                lines.append(f"\n**Output:** {step.output_data.get('summary', 'N/A')}")

            lines.append("\n")

        lines.extend(
            [
                "## Final Conclusion\n",
                self.final_conclusion,
            ]
        )

        if self.citations_used:
            lines.extend(
                [
                    "\n## Citations\n",
                    "\n".join(f"- {citation}" for citation in self.citations_used),
                ]
            )

        return "\n".join(lines)
