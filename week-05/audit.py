"""
Top-level audit logging interface module.
Exposes AuditService, get_audit_service, and action constants.
"""

from services.audit import AuditService, get_audit_service, ALL_ACTIONS

__all__ = ["AuditService", "get_audit_service", "ALL_ACTIONS"]
