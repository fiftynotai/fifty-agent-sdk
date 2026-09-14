"""Deterministic retention and eviction regressions for ``MemoryStateStore``.

BR-012 makes the default ephemeral backend bounded without changing the
shared branching contract. Tests patch only the store's private monotonic
clock; no wall-clock sleeps or timing budgets participate in correctness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from fifty_agent_sdk import ChatMessage, MemoryStateStore
from fifty_agent_sdk.state import memory as memory_module
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID


class _FakeClock:
    """Explicit monotonic clock controlled by each test."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _AcquireGateLock(asyncio.Lock):
    """Pause an acquisition before the underlying lock becomes locked."""

    def __init__(self) -> None:
        super().__init__()
        self.acquire_started = asyncio.Event()
        self.allow_acquire = asyncio.Event()

    async def acquire(self) -> bool:
        self.acquire_started.set()
        await self.allow_acquire.wait()
        return await super().acquire()


class _ObservedLock(asyncio.Lock):
    """Signal when a second caller queues behind the current holder."""

    def __init__(self) -> None:
        super().__init__()
        self._acquire_count = 0
        self.second_acquire_started = asyncio.Event()

    async def acquire(self) -> bool:
        self._acquire_count += 1
        if self._acquire_count == 2:
            self.second_acquire_started.set()
        return await super().acquire()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    """Patch only BR-012's private monotonic boundary."""
    fake = _FakeClock()
    monkeypatch.setattr(memory_module, "_monotonic", fake)
    return fake


def _msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _contents(messages: list[ChatMessage]) -> list[str | None]:
    return [message.content for message in messages]


# ---------------------------------------------------------------------------
# Constructor and defaults
# ---------------------------------------------------------------------------


def test_constructor_retention_arguments_are_keyword_only() -> None:
    """BR-012 keeps zero-argument construction while rejecting positional policy."""
    with pytest.raises(TypeError):
        MemoryStateStore(10.0)  # type: ignore[misc]


@pytest.mark.parametrize("ttl_seconds", [True, "10", object()])
def test_constructor_rejects_non_numeric_ttl(ttl_seconds: object) -> None:
    """TTL accepts only non-boolean int/float values or ``None``."""
    with pytest.raises(TypeError, match="ttl_seconds"):
        MemoryStateStore(ttl_seconds=ttl_seconds)  # type: ignore[arg-type]


@pytest.mark.parametrize("ttl_seconds", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_constructor_rejects_non_positive_or_non_finite_ttl(ttl_seconds: float) -> None:
    """TTL rejects zero, negatives, NaN, and either infinity."""
    with pytest.raises(ValueError, match="ttl_seconds"):
        MemoryStateStore(ttl_seconds=ttl_seconds)


@pytest.mark.parametrize("max_sessions", [True, 1.5, "10", object()])
def test_constructor_rejects_non_integer_session_cap(max_sessions: object) -> None:
    """The LRU cap accepts only a non-boolean integer or ``None``."""
    with pytest.raises(TypeError, match="max_sessions"):
        MemoryStateStore(max_sessions=max_sessions)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_sessions", [0, -1])
def test_constructor_rejects_non_positive_session_cap(max_sessions: int) -> None:
    """The LRU cap must be strictly positive when enabled."""
    with pytest.raises(ValueError, match="max_sessions"):
        MemoryStateStore(max_sessions=max_sessions)


async def test_default_ttl_expires_at_one_hour(clock: _FakeClock) -> None:
    """Default construction really enables the documented 3,600-second TTL."""
    store = MemoryStateStore()
    await store.append("old", _msg("value"))

    clock.advance(3_600.0)

    assert await store.get_messages("old") == []


async def test_default_capacity_evicts_session_1001(clock: _FakeClock) -> None:
    """Default construction really enables the documented 1,000-session cap."""
    store = MemoryStateStore()
    for index in range(1_001):
        await store.append(f"session-{index}", _msg(str(index)))

    assert "session-0" not in store._sessions
    assert len(store._sessions) == 1_000
    assert _contents(await store.get_messages("session-1000")) == ["1000"]


# ---------------------------------------------------------------------------
# Inactivity TTL
# ---------------------------------------------------------------------------


async def test_ttl_boundary_and_successful_read_refresh(clock: _FakeClock) -> None:
    """A read refreshes inactivity, and elapsed == TTL expires the whole session."""
    store = MemoryStateStore(ttl_seconds=10.0, max_sessions=None)
    await store.append("s1", _msg("kept"))

    clock.advance(9.999)
    assert _contents(await store.get_messages("s1")) == ["kept"]

    clock.advance(9.999)
    assert _contents(await store.get_messages("s1")) == ["kept"]

    clock.advance(10.0)
    assert await store.get_messages("s1") == []


async def test_append_after_expiry_creates_clean_trunk(clock: _FakeClock) -> None:
    """Appending after BR-012 expiry cannot resurrect stale messages or forks."""
    store = MemoryStateStore(ttl_seconds=5.0, max_sessions=None)
    await store.append("s1", _msg("old"))
    old_branch = await store.fork("s1", from_sequence=1)

    clock.advance(5.0)
    await store.append("s1", _msg("new"))

    assert _contents(await store.get_messages("s1")) == ["new"]
    assert [branch.branch_id for branch in await store.list_branches("s1")] == [TRUNK_BRANCH_ID]
    with pytest.raises(ValueError, match="does not exist"):
        await store.get_messages("s1", branch_id=old_branch)


async def test_expiration_removes_whole_session_and_lock_bookkeeping(clock: _FakeClock) -> None:
    """TTL eviction removes branches, active head, lock, and reservations together."""
    store = MemoryStateStore(ttl_seconds=5.0, max_sessions=None)
    await store.append("s1", _msg("old"))
    await store.fork("s1", from_sequence=1)
    assert "s1" in store._locks

    clock.advance(5.0)
    assert await store.get_messages("s1") == []

    assert "s1" not in store._sessions
    assert "s1" not in store._locks
    assert "s1" not in store._reservations


async def test_expired_session_preserves_branch_operation_unknown_rules(clock: _FakeClock) -> None:
    """Expiry maps each branch operation to its established unknown-session outcome."""
    factories: list[
        tuple[
            str,
            Callable[[MemoryStateStore], Awaitable[object]],
            type[Exception] | None,
        ]
    ] = [
        ("list", lambda store: store.list_branches("s1"), None),
        ("fork", lambda store: store.fork("s1", from_sequence=0), ValueError),
        ("switch", lambda store: store.switch_branch("s1", TRUNK_BRANCH_ID), ValueError),
        ("truncate", lambda store: store.truncate_after("s1", 0), None),
        (
            "explicit-read",
            lambda store: store.get_messages("s1", branch_id=TRUNK_BRANCH_ID),
            ValueError,
        ),
    ]

    for name, operation, error_type in factories:
        store = MemoryStateStore(ttl_seconds=5.0, max_sessions=None)
        await store.append("s1", _msg(name))
        clock.advance(5.0)
        if error_type is None:
            result = await operation(store)
            if name == "list":
                assert result == []
        else:
            with pytest.raises(error_type):
                await operation(store)


async def test_every_successful_state_operation_refreshes_activity(clock: _FakeClock) -> None:
    """All non-delete operations participate in BR-012 inactivity bookkeeping."""
    store = MemoryStateStore(ttl_seconds=10.0, max_sessions=None)
    await store.append("s1", _msg("a"))

    clock.advance(9.0)
    assert _contents(await store.get_messages("s1")) == ["a"]
    clock.advance(9.0)
    branch = await store.fork("s1", from_sequence=1)
    clock.advance(9.0)
    assert len(await store.list_branches("s1")) == 2
    clock.advance(9.0)
    await store.switch_branch("s1", branch)
    clock.advance(9.0)
    await store.truncate_after("s1", 1)
    clock.advance(9.0)
    await store.append("s1", _msg("b"))

    assert _contents(await store.get_messages("s1")) == ["a", "b"]
    await store.delete("s1")
    assert "s1" not in store._sessions
    assert "s1" not in store._locks


# ---------------------------------------------------------------------------
# LRU and opt-outs
# ---------------------------------------------------------------------------


async def test_lru_capacity_evicts_least_recently_used_session_br012(
    clock: _FakeClock,
) -> None:
    """BR-012 mutation guard: capacity cleanup must remove the true LRU victim."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=2)
    await store.append("s1", _msg("one"))
    await store.append("s2", _msg("two"))
    assert _contents(await store.get_messages("s1")) == ["one"]

    await store.append("s3", _msg("three"))

    assert "s2" not in store._sessions
    assert "s2" not in store._locks
    assert await store.get_messages("s2") == []
    assert _contents(await store.get_messages("s1")) == ["one"]
    assert _contents(await store.get_messages("s3")) == ["three"]


async def test_lru_eviction_removes_every_branch_and_active_head(clock: _FakeClock) -> None:
    """Capacity eviction treats a branched conversation as one indivisible session."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=1)
    await store.append("old", _msg("trunk"))
    branch = await store.fork("old", from_sequence=1)
    await store.switch_branch("old", branch)
    await store.append("old", _msg("fork"))

    await store.append("new", _msg("new"))

    assert "old" not in store._sessions
    assert "old" not in store._locks
    assert "old" not in store._reservations
    assert await store.list_branches("old") == []


async def test_equal_clock_lru_uses_access_ordinal_tie_breaker(clock: _FakeClock) -> None:
    """Equal monotonic readings still produce deterministic access-order eviction."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=2)
    await store.append("first", _msg("first"))
    await store.append("second", _msg("second"))
    await store.get_messages("first")

    await store.append("third", _msg("third"))

    assert set(store._sessions) == {"first", "third"}


async def test_none_ttl_disables_time_based_expiry(clock: _FakeClock) -> None:
    """``ttl_seconds=None`` keeps a session live across arbitrary clock advances."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=1)
    await store.append("s1", _msg("kept"))
    clock.advance(1_000_000.0)
    assert _contents(await store.get_messages("s1")) == ["kept"]


async def test_none_capacity_allows_more_than_default_cap(clock: _FakeClock) -> None:
    """``max_sessions=None`` can intentionally retain more than 1,000 sessions."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=None)
    for index in range(1_001):
        await store.append(f"session-{index}", _msg(str(index)))
    assert len(store._sessions) == 1_001


async def test_both_none_restore_legacy_unbounded_behavior(clock: _FakeClock) -> None:
    """The documented two-``None`` opt-in survives both time and capacity pressure."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=None)
    for index in range(1_001):
        await store.append(f"session-{index}", _msg(str(index)))
    clock.advance(1_000_000.0)
    assert _contents(await store.get_messages("session-0")) == ["0"]
    assert len(store._sessions) == 1_001


# ---------------------------------------------------------------------------
# Reservation-aware concurrency
# ---------------------------------------------------------------------------


async def test_queued_append_reservation_prevents_stale_lock_resurrection_br012(
    clock: _FakeClock,
) -> None:
    """BR-012 mutation guard: a reserved waiter cannot lose old state before locking."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=1)
    await store.append("protected", _msg("old"))
    gate_lock = _AcquireGateLock()
    store._locks["protected"] = gate_lock

    queued_append = asyncio.create_task(store.append("protected", _msg("new")))
    await gate_lock.acquire_started.wait()

    await store.append("pressure", _msg("pressure"))
    assert "protected" in store._sessions
    assert store._reservations["protected"] == 1

    gate_lock.allow_acquire.set()
    await queued_append

    assert _contents(await store.get_messages("protected")) == ["old", "new"]


async def test_locked_session_is_not_an_lru_candidate_br012(clock: _FakeClock) -> None:
    """BR-012 mutation guard: a locked session is protected without a reservation."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=1)
    await store.append("protected", _msg("old"))
    lock = await store._get_lock("protected")
    await lock.acquire()
    try:
        await store.append("pressure", _msg("pressure"))
        assert "protected" in store._sessions
    finally:
        lock.release()

    assert _contents(await store.get_messages("protected")) == ["old"]


async def test_append_queued_on_held_session_lock_preserves_history_br012(
    clock: _FakeClock,
) -> None:
    """A queued same-session append survives concurrent LRU capacity pressure."""
    store = MemoryStateStore(ttl_seconds=None, max_sessions=1)
    await store.append("protected", _msg("old"))
    observed_lock = _ObservedLock()
    store._locks["protected"] = observed_lock
    await observed_lock.acquire()

    queued_append = asyncio.create_task(store.append("protected", _msg("new")))
    await observed_lock.second_acquire_started.wait()
    await store.append("pressure", _msg("pressure"))
    assert "protected" in store._sessions

    observed_lock.release()
    await queued_append

    assert _contents(await store.get_messages("protected")) == ["old", "new"]
    assert store._reservations.get("protected", 0) == 0
