"""State subpackage: pluggable conversation persistence.

Re-exports the :class:`StateStore` protocol and the default in-memory
implementation :class:`MemoryStateStore`. The durable SQL backend
(:class:`SqlStateStore`) and the Alembic :data:`sql_metadata` symbol
are re-exported lazily via a module-level ``__getattr__`` — they require
the optional ``sql`` extra, so ``import agent_sdk.state`` itself does
not pull SQLAlchemy.

Accessing :data:`SqlStateStore` or :data:`sql_metadata` without
``agent-sdk[sql]`` installed raises a clear :class:`ImportError`
referencing the extras line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_sdk.state.memory import MemoryStateStore
from agent_sdk.state.protocol import StateStore

if TYPE_CHECKING:
    from agent_sdk.state.sql import SqlStateStore, sql_metadata

__all__ = ["MemoryStateStore", "SqlStateStore", "StateStore", "sql_metadata"]


def __getattr__(name: str) -> Any:
    """Lazily import SQL surface symbols on first access.

    Keeps the package's eager import surface free of SQLAlchemy. When
    the ``sql`` extra is not installed, importing
    :mod:`agent_sdk.state.sql` (triggered here) raises a documented
    :class:`ImportError`.
    """
    if name in {"SqlStateStore", "sql_metadata"}:
        from agent_sdk.state import sql

        return getattr(sql, name)
    raise AttributeError(f"module 'agent_sdk.state' has no attribute {name!r}")
