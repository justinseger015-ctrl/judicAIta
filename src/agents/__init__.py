"""Multi-agent system for legal document processing.

This module provides the base agent architecture and specialized agents for:
- Document ingestion (PDF, DOCX, OCR)
- Analysis (entity extraction, clause extraction, citation parsing)
- Reasoning (QA, summarization, compliance)
- Orchestration (LangGraph workflows)
"""

from src.agents.base import BaseAgent, AgentConfig, AgentResponse

__all__ = ["BaseAgent", "AgentConfig", "AgentResponse"]
