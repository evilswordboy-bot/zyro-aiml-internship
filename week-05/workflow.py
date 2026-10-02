"""
Top-level workflow interface module.
Exposes WorkflowEngine, state transitions, and evaluation rules.
"""

from services.workflow import (
    WorkflowEngine,
    get_workflow_engine,
    ALLOWED_TRANSITIONS,
    CONFIDENCE_THRESHOLD,
)

__all__ = ["WorkflowEngine", "get_workflow_engine", "ALLOWED_TRANSITIONS", "CONFIDENCE_THRESHOLD"]
