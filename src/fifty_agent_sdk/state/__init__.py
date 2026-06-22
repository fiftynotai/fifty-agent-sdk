"""State subpackage: pluggable conversation persistence.

Re-exports the :class:`StateStore` protocol and the default in-memory
implementation :class:`MemoryStateStore`. The durable SQL backend
(:class:`SqlStateStore`) with its Alembic :data:`sql_metadata` symbol, and
the Redis backend (:class:`RedisStateStore`), are re-exported lazily via a
module-level ``__getattr__`` — they require the optional ``sql`` / ``redis``
extras, so ``import fifty_agent_sdk.state`` itself pulls neither SQLAlchemy nor
redis-py.

Accessing :data:`SqlStateStore` or :data:`sql_metadata` without
``fifty-agent-sdk[sql]`` installed — or :data:`RedisStateStore` without
``fifty-agent-sdk[redis]`` installed — raises a clear :class:`ImportError`
referencing the relevant extras line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fifty_agent_sdk.state.memory import MemoryStateStore
from fifty_agent_sdk.state.protocol import StateStore

if TYPE_CHECKING:
    from fifty_agent_sdk.state.redis import RedisStateStore
    from fifty_agent_sdk.state.sql import SqlStateStore, sql_metadata

__all__ = [
    "MemoryStateStore",
    "RedisStateStore",
    "SqlStateStore",
    "StateStore",
    "sql_metadata",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional-extra surface symbols on first access.

    Keeps the package's eager import surface free of SQLAlchemy and
    redis-py. When the relevant extra is not installed, importing the
    backing module (triggered here) raises a documented
    :class:`ImportError`.
    """
    if name in {"SqlStateStore", "sql_metadata"}:
        from fifty_agent_sdk.state import sql

        return getattr(sql, name)
    if name == "RedisStateStore":
        from fifty_agent_sdk.state import redis

        return redis.RedisStateStore
    raise AttributeError(f"module 'fifty_agent_sdk.state' has no attribute {name!r}")
