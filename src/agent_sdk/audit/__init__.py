"""Audit subpackage: pluggable, tamper-evident action provenance.

Re-exports the :class:`AuditEvent` model, the :class:`AuditSink` protocol,
and the dependency-free :class:`ConsoleAuditSink`. The durable SQL backend
(:class:`SqlAuditSink`) with its Alembic :data:`audit_metadata` symbol is
re-exported lazily via a module-level ``__getattr__`` — it requires the
optional ``sql`` extra, so ``import agent_sdk.audit`` itself does NOT pull
SQLAlchemy.

Accessing :data:`SqlAuditSink` or :data:`audit_metadata` without
``agent-sdk[sql]`` installed raises a clear :class:`ImportError` referencing
the extras line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_sdk.audit.console import ConsoleAuditSink
from agent_sdk.audit.protocol import AuditEvent, AuditSink

if TYPE_CHECKING:
    from agent_sdk.audit.sql import SqlAuditSink, audit_metadata

__all__ = [
    "AuditEvent",
    "AuditSink",
    "ConsoleAuditSink",
    "SqlAuditSink",
    "audit_metadata",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional-extra surface symbols on first access.

    Keeps the package's eager import surface free of SQLAlchemy. When the
    ``sql`` extra is not installed, importing the backing module (triggered
    here) raises a documented :class:`ImportError`.
    """
    if name in {"SqlAuditSink", "audit_metadata"}:
        from agent_sdk.audit import sql

        return getattr(sql, name)
    raise AttributeError(f"module 'agent_sdk.audit' has no attribute {name!r}")
