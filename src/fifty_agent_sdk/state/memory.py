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
    history. That definition is recursive; the computation is **iterative**
    (TD-002), so lineage depth is not bounded by Python's recursion limit.
    The implicit first branch is ``"trunk"``
    (:data:`fifty_agent_sdk.state.protocol.TRUNK_BRANCH_ID`).

Memory characteristics
    Sessions are bounded by default: inactive sessions expire after one
    hour and the least-recently-used idle sessions are evicted above 1,000
    live sessions. Retention is whole-session, including every branch, and
    cleanup is lazy at operation boundaries. Reads refresh inactivity. Use
    ``MemoryStateStore(ttl_seconds=None, max_sessions=None)`` only when the
    former unbounded process-local behavior is intentional, or use a durable
    backend (BR-009 SQL / BR-010 Redis) when state must survive process exit.

:meth:`get_messages` returns a freshly-built list, so callers mutating it
cannot affect future reads.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

from fifty_agent_sdk.llm.types import ChatMessage
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID, BranchInfo

_DEFAULT_TTL_SECONDS: Final = 3600.0
_DEFAULT_MAX_SESSIONS: Final = 1_000


def _now() -> datetime:
    """Return the current timezone-aware UTC time (branch creation stamp)."""
    return datetime.now(UTC)


def _monotonic() -> float:
    """Return monotonic seconds for inactivity and LRU bookkeeping."""
    return time.monotonic()


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
    last_activity: float
    access_ordinal: int
    branches: dict[str, _Branch] = field(default_factory=dict)


class MemoryStateStore:
    """Bounded in-memory implementation of :class:`StateStore` with branching.

    Satisfies :class:`fifty_agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).
    Asyncio-safe per session via :class:`asyncio.Lock`.

    By default, a whole session expires after 3,600 seconds of inactivity and
    the store retains at most 1,000 sessions, evicting the least recently used
    idle session first. Every successful operation on an existing session,
    including a read, refreshes activity. Cleanup is lazy at operation
    boundaries; no background task is created. Sessions with active or queued
    operations are protected, so the cap may be exceeded transiently until an
    operation boundary makes an idle victim available.

    Disable either policy independently with ``None``. Construct with
    ``MemoryStateStore(ttl_seconds=None, max_sessions=None)`` for the former
    unbounded process-local behavior. This backend is never durable.

    Failure model:
        Dict operations have no plausible backend failure mode, so this
        implementation does NOT raise
        :class:`fifty_agent_sdk.errors.StateStoreError`. Programmer errors
        (unknown explicit ``branch_id``, out-of-range fork point) raise
        :class:`ValueError` per the protocol.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float | None = _DEFAULT_TTL_SECONDS,
        max_sessions: int | None = _DEFAULT_MAX_SESSIONS,
    ) -> None:
        """Construct an empty store with validated retention policy.

        Args:
            ttl_seconds: Positive finite inactivity window, or ``None`` to
                disable time-based expiration.
            max_sessions: Positive whole-session LRU cap, or ``None`` to
                disable capacity eviction.

        Raises:
            TypeError: A value has the wrong type.
            ValueError: A numeric value is not finite and strictly positive.
        """
        self._ttl_seconds = self._validate_ttl_seconds(ttl_seconds)
        self._max_sessions = self._validate_max_sessions(max_sessions)
        self._sessions: dict[str, _Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._reservations: dict[str, int] = {}
        self._registry_lock = asyncio.Lock()
        self._access_counter = 0

    @staticmethod
    def _validate_ttl_seconds(ttl_seconds: float | None) -> float | None:
        """Validate and normalize the inactivity window."""
        if ttl_seconds is None:
            return None
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("ttl_seconds must be a finite positive int or float, or None")
        try:
            normalized = float(ttl_seconds)
        except OverflowError as exc:
            raise ValueError("ttl_seconds must be finite and strictly greater than zero") from exc
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("ttl_seconds must be finite and strictly greater than zero")
        return normalized

    @staticmethod
    def _validate_max_sessions(max_sessions: int | None) -> int | None:
        """Validate the whole-session capacity."""
        if max_sessions is None:
            return None
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be a positive int or None")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be strictly greater than zero")
        return max_sessions

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session lock identity, creating it if necessary.

        Public operations use :meth:`_session_guard`, which reserves the lock
        before awaiting it. This identity helper remains for focused tests and
        diagnostics; callers must not use it as an operation guard.
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

    @asynccontextmanager
    async def _session_guard(self, session_id: str) -> AsyncIterator[None]:
        """Reserve and acquire one session lock, then perform lazy cleanup.

        Reservations include both active holders and queued waiters. Eviction
        can therefore never remove a lock after an operation has captured its
        identity but before that operation acquires it.
        """
        async with self._registry_lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            self._reservations[session_id] = self._reservations.get(session_id, 0) + 1

        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            async with self._registry_lock:
                reservations = self._reservations[session_id] - 1
                if reservations:
                    self._reservations[session_id] = reservations
                else:
                    self._reservations.pop(session_id, None)
                self._evict_idle_sessions(_monotonic())

    def _is_expired(self, session: _Session, now: float) -> bool:
        """Return whether ``session`` reached the inactivity boundary."""
        return self._ttl_seconds is not None and now - session.last_activity >= self._ttl_seconds

    def _expire_current_session(self, session_id: str, now: float) -> _Session | None:
        """Return a live session, dropping expired data under its own lock."""
        session = self._sessions.get(session_id)
        if session is not None and self._is_expired(session, now):
            self._sessions.pop(session_id)
            return None
        return session

    def _touch(self, session: _Session) -> None:
        """Refresh inactivity and deterministic LRU metadata."""
        self._access_counter += 1
        session.last_activity = _monotonic()
        session.access_ordinal = self._access_counter

    def _is_evictable(self, session_id: str) -> bool:
        """Return whether no active or queued operation protects a session."""
        lock = self._locks.get(session_id)
        return self._reservations.get(session_id, 0) == 0 and (lock is None or not lock.locked())

    def _drop_session(self, session_id: str) -> None:
        """Remove one idle session and all of its lock bookkeeping."""
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._reservations.pop(session_id, None)

    def _evict_idle_sessions(self, now: float) -> None:
        """Expire inactive sessions and enforce LRU capacity.

        Called only while ``self._registry_lock`` is held and only after the
        current operation released its per-session lock. It never waits for a
        protected session; a later operation boundary retries cleanup.
        """
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_expired(session, now) and self._is_evictable(session_id)
        ]
        for session_id in expired:
            self._drop_session(session_id)

        if self._max_sessions is not None and len(self._sessions) > self._max_sessions:
            candidates = sorted(
                (
                    (session.last_activity, session.access_ordinal, session_id)
                    for session_id, session in self._sessions.items()
                    if self._is_evictable(session_id)
                ),
            )
            overflow = len(self._sessions) - self._max_sessions
            for _, _, session_id in candidates[:overflow]:
                self._drop_session(session_id)

        orphaned_locks = [
            session_id
            for session_id in self._locks
            if session_id not in self._sessions and self._is_evictable(session_id)
        ]
        for session_id in orphaned_locks:
            self._locks.pop(session_id, None)
            self._reservations.pop(session_id, None)

    @staticmethod
    def _materialize(session: _Session, branch_id: str) -> list[ChatMessage]:
        """Return the full materialized history of ``branch_id``.

        Defined recursively: a branch's history is its parent's history
        truncated at the fork point, followed by the branch's own messages.
        The trunk (no parent) materializes to just its own messages. Returns
        a fresh list every call (defensive-copy invariant).

        **Computed iteratively** (TD-002): the definition recurses, the
        computation does not. Lineage depth is therefore bounded only by
        memory, not by Python's recursion limit — a deep linear fork chain
        previously raised :class:`RecursionError` from here (and so from
        :meth:`fork`, which calls this to bound ``from_sequence``).
        """
        # Phase 1: walk parent pointers leaf -> root. Pure pointer work.
        # No cycle guard, deliberately (TD-002): parent_branch_id is written
        # exactly once, in fork(), to the already-existing active branch, and
        # is never mutated afterwards, so every branch's parent strictly
        # precedes it in creation order and the lineage is acyclic by
        # construction. The honest asymmetry: an externally corrupted store
        # holding a cycle used to raise RecursionError and would now spin
        # forever. That is unreachable through any public API.
        chain: list[_Branch] = []
        bid: str | None = branch_id
        while bid is not None:
            branch = session.branches[bid]  # KeyError on a missing branch: preserved
            chain.append(branch)
            bid = branch.parent_branch_id
        chain.reverse()  # root .. leaf

        # Phase 2: fold root -> leaf. The base case and the step mirror the
        # recursive form literally (and stay separate) so this can be diffed
        # against it line for line.
        # `chain` is non-empty: branch_id is never None, so the walk ran once.
        history: list[ChatMessage] = list(chain[0].messages)  # base case: the trunk
        for branch in chain[1:]:  # step
            # forked_from_sequence is a count of inherited messages.
            #
            # The slice clamps naturally when an ancestor was truncated below
            # this fork point — a short `history` simply yields all of it. This
            # is the IMPLICIT form of the explicit `min` in SqlStateStore and
            # RedisStateStore `_materialized_len`, which cite this method as the
            # reference. Same clamp, expressed by Python's slice semantics
            # rather than by an arithmetic `min`; that equivalence is the seam
            # the BR-003 x BR-004 differential regressions guard.
            history = history[: branch.forked_from_sequence or 0] + list(branch.messages)
        return history

    def _ensure_session(self, session_id: str) -> _Session:
        """Return the session, creating it (with an empty trunk) if absent."""
        session = self._sessions.get(session_id)
        if session is None:
            now = _now()
            session = _Session(
                active_branch_id=TRUNK_BRANCH_ID,
                created_at=now,
                last_activity=_monotonic(),
                access_ordinal=0,
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
        async with self._session_guard(session_id):
            session = self._expire_current_session(session_id, _monotonic())
            if session is None:
                if branch_id is not None:
                    raise ValueError(
                        f"branch_id={branch_id!r} does not exist for unknown session {session_id!r}"
                    )
                return []
            target = branch_id if branch_id is not None else session.active_branch_id
            if target not in session.branches:
                raise ValueError(f"branch_id={target!r} does not exist for session {session_id!r}")
            messages = self._materialize(session, target)
            self._touch(session)
            return messages

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's active branch.

        Creates the session (on an empty ``"trunk"``) if it is new.
        """
        async with self._session_guard(session_id):
            self._expire_current_session(session_id, _monotonic())
            session = self._ensure_session(session_id)
            session.branches[session.active_branch_id].messages.append(message)
            self._touch(session)

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id`` (every branch).

        Idempotent. Serializes with every operation on the same session via the
        reservation-aware session guard. Data and lock bookkeeping are removed
        together when the guard exits.
        """
        async with self._session_guard(session_id):
            self._sessions.pop(session_id, None)

    async def fork(self, session_id: str, from_sequence: int) -> str:
        """Fork the active branch at ``from_sequence`` into a new branch.

        The new branch inherits the active branch's history up to
        ``from_sequence``; the original branch is untouched. Does NOT switch
        the active head.
        """
        async with self._session_guard(session_id):
            session = self._expire_current_session(session_id, _monotonic())
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
            self._touch(session)
            return new_id

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        """Enumerate all branches of ``session_id`` (trunk first, then by age)."""
        async with self._session_guard(session_id):
            session = self._expire_current_session(session_id, _monotonic())
            if session is None:
                return []

            def sort_key(item: tuple[str, _Branch]) -> tuple[int, datetime, str]:
                bid, branch = item
                # Trunk first, then by creation time, then id for determinism.
                return (0 if bid == TRUNK_BRANCH_ID else 1, branch.created_at, bid)

            branches = [
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
            self._touch(session)
            return branches

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        """Set the session's active head to ``branch_id``."""
        async with self._session_guard(session_id):
            session = self._expire_current_session(session_id, _monotonic())
            if session is None or branch_id not in session.branches:
                raise ValueError(
                    f"branch_id={branch_id!r} does not exist for session {session_id!r}"
                )
            session.active_branch_id = branch_id
            self._touch(session)

    async def truncate_after(
        self, session_id: str, sequence: int, *, branch_id: str | None = None
    ) -> None:
        """Destructively drop the target branch's own messages with
        ``sequence > N``.

        Only the branch's own messages are removed — a fork's inherited prefix
        is never touched. Idempotent; a no-op on an unknown session or branch.
        """
        async with self._session_guard(session_id):
            session = self._expire_current_session(session_id, _monotonic())
            if session is None:
                return
            target = branch_id if branch_id is not None else session.active_branch_id
            branch = session.branches.get(target)
            if branch is None:
                self._touch(session)
                return
            anchor = branch.forked_from_sequence or 0
            # Own message at index i has materialized sequence anchor + 1 + i;
            # keep those with sequence <= N (the first ``N - anchor``), drop the
            # rest. Slicing clamps naturally for out-of-range N.
            keep = sequence - anchor
            del branch.messages[max(keep, 0) :]
            self._touch(session)


__all__ = ["MemoryStateStore"]
