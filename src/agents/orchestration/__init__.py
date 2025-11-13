"""
Orchestration module for multi-agent workflows.

This module provides workflow orchestration capabilities using LangGraph.
"""

from src.agents.orchestration.workflow import (
    DocumentType,
    LegalDocumentWorkflow,
    WorkflowConfig,
    WorkflowState,
    create_case_law_workflow,
    create_contract_workflow,
)

__all__ = [
    "DocumentType",
    "LegalDocumentWorkflow",
    "WorkflowConfig",
    "WorkflowState",
    "create_contract_workflow",
    "create_case_law_workflow",
]
