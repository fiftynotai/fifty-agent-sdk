"""In-memory implementation of :class:`fifty_agent_sdk.state.protocol.StateStore`.

:class:`MemoryStateStore` is the SDK's default conversation-state backend
when durability is not required (development, examples, tests, ephemeral
agents). It is purely process-local — all data is lost on process exit. It
is also the reference implementation of the BR-004 branching contract.

Concurrency model
    Each ``session_id`` gets its own :class:`asyncio.Lock`, allocated
    lazily on first access. A small registry lock guards lock-table
    mutations so concurrent first-access for the same session never
    races. Different sessions never block each other. Every operation
    (reads, appends, and the branch ops) runs under the per-session lock,
    so the active head and branch tree never observe a torn write.

Branching model
    A session holds a tree of :class:`_Branch` records keyed by branch id,
    an active-head pointer, and per-branch *own* message lists. A branch's
    materialized history is defined recursively as
    ``materialize(parent)[:fork_point] + own_messages`` — which naturally
    handles a branch forked from a point inside its parent's *inherited*
    history. The implicit first branch is ``"trunk"``
    (:data:`fifty_agent_sdk.state.protocol.TRUNK_BRANCH_ID`).

Memory characteristics
    Unbounded: every unique ``session_id`` ever seen retains a (small)
    lock entry until explicitly :meth:`MemoryStateStore.delete`'d. For
    long-running processes with many sessions, use a durable backend
    (BR-009 SQL / BR-010 Redis) which scopes locking to the backend
    layer. Callers may also periodically :meth:`delete` stale sessions.

:meth:`get_messages` returns a freshly-built list, so callers mutating it
cannot affect future reads.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

import structlog

from fifty_agent_sdk.llm.types import ChatMessage
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID, BranchInfo

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Currently no events are emitted from the in-memory store (all operations
are O(1)/O(history) in-process ops and not worth log noise). Reserved for
future use if a TTL or eviction policy lands.
"""


def _now() -> datetime:
    """Return the current timezone-aware UTC time (branch creation stamp)."""
    return datetime.now(UTC)


@dataclass
class _Branch:
    """One branch's own (non-inherited) messages plus its lineage metadata."""

    parent_branch_id: str | None
    forked_from_sequence: int | None
    created_at: datetime
    messages: list[ChatMessage] = field(default_factory=list)


@dataclass
class _Session:
    """A session's branch tree and active head."""

    active_branch_id: str
    created_at: datetime
    branches: dict[str, _Branch] = field(default_factory=dict)


class MemoryStateStore:
    """Pure in-memory implementation of :class:`StateStore` with branching.

    Satisfies :class:`fifty_agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).
    Thread- and asyncio-safe per session via :class:`asyncio.Lock`.

    Failure model:
        Dict operations have no plausible backend failure mode, so this
        implementation does NOT raise
        :class:`fifty_agent_sdk.errors.StateStoreError`. Programmer errors
        (unknown explicit ``branch_id``, out-of-range fork point) raise
        :class:`ValueError` per the protocol.
    """

    def __init__(self) -> None:
        """Construct an empty in-memory store."""
        self._sessions: dict[str, _Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session lock, creating it if necessary.

        Hot path: an already-existing lock is returned without acquiring
        ``self._registry_lock``. Cold path: under the registry lock we
        re-check existence (double-checked locking) and create if absent.
        """
        existing = self._locks.get(session_id)
        if existing is not None:
            return existing
        async with self._registry_lock:
            existing = self._locks.get(session_id)
            if existing is not None:
                return existing
            lock = asyncio.Lock()
            self._locks[session_id] = lock
            return lock

    @staticmethod
    def _materialize(session: _Session, branch_id: str) -> list[ChatMessage]:
        """Return the full materialized history of ``branch_id``.

        Defined recursively: a branch's history is its parent's history
        truncated at the fork point, followed by the branch's own messages.
        The trunk (no parent) materializes to just its own messages. Returns
        a fresh list every call (defensive-copy invariant).
        """
        branch = session.branches[branch_id]
        if branch.parent_branch_id is None:
            base: list[ChatMessage] = []
        else:
            parent_hist = MemoryStateStore._materialize(session, branch.parent_branch_id)
            # forked_from_sequence is a 1-based count of inherited messages.
            base = parent_hist[: branch.forked_from_sequence or 0]
        return base + list(branch.messages)

    def _ensure_session(self, session_id: str) -> _Session:
        """Return the session, creating it (with an empty trunk) if absent."""
        session = self._sessions.get(session_id)
        if session is None:
            now = _now()
            session = _Session(
                active_branch_id=TRUNK_BRANCH_ID,
                created_at=now,
                branches={TRUNK_BRANCH_ID: _Branch(None, None, now)},
            )
            self._sessions[session_id] = session
        return session

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        """Return the materialized messages for a branch of ``session_id``.

        ``branch_id=None`` reads the active branch. An unknown session with
        ``branch_id=None`` yields ``[]``; an explicit unknown ``branch_id``
        raises :class:`ValueError`.
        """
        lock = await self._get_lock(session_id)
        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                if branch_id is not None:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for unknown session {session_id!r}"
                    )
                return []
            target = branch_id if branch_id is not None else session.active_branch_id
            if target not in session.branches:
                raise ValueError(f"branch_id={target!r} does not exist for session {session_id!r}")
            return self._materialize(session, target)

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's active branch.

        Creates the session (on an empty ``"trunk"``) if it is new.
        """
        lock = await self._get_lock(session_id)
        async with lock:
            session = self._ensure_session(session_id)
            session.branches[session.active_branch_id].messages.append(message)

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id`` (every branch).

        Idempotent. Holds ONLY the registry lock (not the per-session lock)
        so it cannot deadlock with an in-flight op on the same session; the
        resulting "last write wins on data, best-effort lock eviction" race
        is acceptable for ephemeral in-memory state.
        """
        async with self._registry_lock:
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)

    async def fork(self, session_id: str, from_sequence: int) -> str:
        """Fork the active branch at ``from_sequence`` into a new branch.

        The new branch inherits the active branch's history up to
        ``from_sequence``; the original branch is untouched. Does NOT switch
        the active head.
        """
        lock = await self._get_lock(session_id)
        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"cannot fork unknown session {session_id!r}")
            active = session.active_branch_id
            head = len(self._materialize(session, active))
            if not 0 <= from_sequence <= head:
                raise ValueError(
                    f"from_sequence={from_sequence} out of range 0..{head} "
                    f"for active branch {active!r} of session {session_id!r}"
                )
            new_id = uuid.uuid4().hex
            session.branches[new_id] = _Branch(
                parent_branch_id=active,
                forked_from_sequence=from_sequence,
                created_at=_now(),
            )
            return new_id

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        """Enumerate all branches of ``session_id`` (trunk first, then by age)."""
        lock = await self._get_lock(session_id)
        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []

            def sort_key(item: tuple[str, _Branch]) -> tuple[int, datetime, str]:
                bid, branch = item
                # Trunk first, then by creation time, then id for determinism.
                return (0 if bid == TRUNK_BRANCH_ID else 1, branch.created_at, bid)

            return [
                BranchInfo(
                    branch_id=bid,
                    parent_branch_id=branch.parent_branch_id,
                    forked_from_sequence=branch.forked_from_sequence,
                    head_sequence=len(self._materialize(session, bid)),
                    created_at=branch.created_at,
                    is_active=(bid == session.active_branch_id),
                )
                for bid, branch in sorted(session.branches.items(), key=sort_key)
            ]

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        """Set the session's active head to ``branch_id``."""
        lock = await self._get_lock(session_id)
        async with lock:
            session = self._sessions.get(session_id)
            if session is None or branch_id not in session.branches:
                raise ValueError(
                    f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                )
            session.active_branch_id = branch_id


__all__ = ["MemoryStateStore"]
