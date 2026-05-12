"""In-memory dict implementation of :class:`agent_sdk.state.protocol.StateStore`.

:class:`MemoryStateStore` is the SDK's default conversation-state backend
when durability is not required (development, examples, tests, ephemeral
agents). It is purely process-local — all data is lost on process exit.

Concurrency model
    Each ``session_id`` gets its own :class:`asyncio.Lock`, allocated
    lazily on first access. A small registry lock guards lock-table
    mutations so concurrent first-access for the same session never
    races. Different sessions never block each other.

Memory characteristics
    Unbounded: every unique ``session_id`` ever seen retains a (small)
    lock entry until explicitly :meth:`MemoryStateStore.delete`'d. For
    long-running processes with many sessions, use a durable backend
    (BR-009 SQL / BR-010 Redis) which scopes locking to the backend
    layer. Callers may also periodically :meth:`delete` stale sessions.

The :meth:`get_messages` method returns a defensive copy of the internal
list, so callers mutating it cannot affect future reads.
"""

from __future__ import annotations

import asyncio
from typing import Final

import structlog

from agent_sdk.llm.types import ChatMessage

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Currently no events are emitted from the in-memory store (all operations
are O(1) dict ops and not worth log noise). Reserved for future use if a
TTL or eviction policy lands.
"""


class MemoryStateStore:
    """Pure in-memory implementation of :class:`agent_sdk.state.protocol.StateStore`.

    Thread- and asyncio-safe per session via :class:`asyncio.Lock`.

    Concurrency model:
        * One :class:`asyncio.Lock` per session, allocated lazily on
          first access through :meth:`_get_lock`.
        * A single registry lock (``self._registry_lock``) guards
          lock-table mutations to keep lock creation race-free.
        * Different sessions never block each other.
        * :meth:`delete` evicts both the message list and the per-session
          lock. A caller racing :meth:`append` and :meth:`delete` on the
          same session sees implementation-defined ordering on the data
          (last write wins); the lock eviction is best-effort.

    Memory characteristics:
        Unbounded: every unique ``session_id`` ever seen retains a small
        :class:`asyncio.Lock` entry. Use a durable backend for long-running
        processes with many distinct sessions.

    Failure model:
        Dict operations have no plausible backend failure mode, so this
        implementation does NOT raise
        :class:`agent_sdk.errors.StateStoreError`.
    """

    def __init__(self) -> None:
        """Construct an empty in-memory store."""
        self._messages: dict[str, list[ChatMessage]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session lock, creating it if necessary.

        Hot path: an already-existing lock is returned without acquiring
        ``self._registry_lock``. Cold path: under the registry lock we
        re-check existence (double-checked locking) and create if absent.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The :class:`asyncio.Lock` associated with ``session_id``.
        """
        existing = self._locks.get(session_id)
        if existing is not None:
            return existing
        async with self._registry_lock:
            # Re-check under the lock: a concurrent caller may have
            # created the lock between our initial read and acquiring
            # the registry lock.
            existing = self._locks.get(session_id)
            if existing is not None:
                return existing
            lock = asyncio.Lock()
            self._locks[session_id] = lock
            return lock

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Return the persisted messages for ``session_id``.

        Returns a defensive copy — callers mutating the returned list
        cannot affect future reads or :meth:`append` ordering. An unknown
        session yields ``[]`` (not an error).

        Args:
            session_id: Opaque session identifier.

        Returns:
            A new ``list[ChatMessage]`` in append order, possibly empty.
        """
        lock = await self._get_lock(session_id)
        async with lock:
            return list(self._messages.get(session_id, []))

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's message log.

        Atomic with respect to concurrent reads/writes on the same
        session; concurrent writes on DIFFERENT sessions do not block
        each other.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.
        """
        lock = await self._get_lock(session_id)
        async with lock:
            self._messages.setdefault(session_id, []).append(message)

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id``.

        Idempotent — calling on an unknown ``session_id`` is a silent
        no-op. Holds ONLY the registry lock (not the per-session lock)
        so it cannot deadlock with an in-flight :meth:`append`/
        :meth:`get_messages` on the same session; the resulting "last
        write wins on data, best-effort lock eviction" race is
        documented at the class level.

        Args:
            session_id: Opaque session identifier.
        """
        async with self._registry_lock:
            self._messages.pop(session_id, None)
            self._locks.pop(session_id, None)


__all__ = ["MemoryStateStore"]
