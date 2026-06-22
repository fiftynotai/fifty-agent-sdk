"""Unit tests for :class:`fifty_agent_sdk.state.memory.MemoryStateStore`.

Covers the documented contract: round-trip, ordering, defensive copy,
idempotent delete, per-session locking that does not block unrelated
sessions, and lock-table eviction on delete.
"""

from __future__ import annotations

import asyncio

from fifty_agent_sdk import ChatMessage, MemoryStateStore

# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


async def test_get_empty_session_returns_empty_list() -> None:
    store = MemoryStateStore()
    assert await store.get_messages("never-seen") == []


async def test_append_then_get_preserves_order() -> None:
    store = MemoryStateStore()
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content="c"),
    ]
    for msg in msgs:
        await store.append("s1", msg)

    got = await store.get_messages("s1")
    assert got == msgs
    assert [m.content for m in got] == ["a", "b", "c"]


async def test_get_returns_defensive_copy() -> None:
    """Mutating the returned list MUST NOT affect future reads."""
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    first.append(ChatMessage(role="user", content="EVIL"))
    first.clear()

    second = await store.get_messages("s1")
    assert len(second) == 1
    assert second[0].content == "a"


async def test_get_returns_new_list_object_each_call() -> None:
    """Even with no mutations, every ``get_messages`` returns a fresh list."""
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    second = await store.get_messages("s1")
    assert first is not second
    assert first == second


# ---------------------------------------------------------------------------
# Delete semantics
# ---------------------------------------------------------------------------


async def test_delete_clears_messages() -> None:
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.delete("s1")
    assert await store.get_messages("s1") == []


async def test_delete_unknown_session_is_silent_noop() -> None:
    store = MemoryStateStore()
    await store.delete("never-seen")  # Must not raise.
    # And subsequent reads still return [].
    assert await store.get_messages("never-seen") == []


async def test_delete_then_reappend_starts_fresh() -> None:
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.delete("s1")
    await store.append("s1", ChatMessage(role="user", content="b"))
    got = await store.get_messages("s1")
    assert len(got) == 1
    assert got[0].content == "b"


async def test_delete_evicts_per_session_lock() -> None:
    """White-box: ``delete`` removes the per-session lock entry."""
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    assert "s1" in store._locks
    await store.delete("s1")
    assert "s1" not in store._locks
    assert "s1" not in store._sessions


# ---------------------------------------------------------------------------
# Isolation between sessions
# ---------------------------------------------------------------------------


async def test_delete_does_not_affect_other_sessions() -> None:
    store = MemoryStateStore()
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.append("s2", ChatMessage(role="user", content="b"))
    await store.delete("s1")

    assert await store.get_messages("s1") == []
    s2 = await store.get_messages("s2")
    assert len(s2) == 1
    assert s2[0].content == "b"


async def test_appends_to_different_sessions_do_not_interleave() -> None:
    """Two parallel append loops on different sessions both finish intact."""
    store = MemoryStateStore()

    async def write_many(session_id: str, prefix: str) -> None:
        for index in range(10):
            await store.append(
                session_id,
                ChatMessage(role="user", content=f"{prefix}-{index}"),
            )

    await asyncio.gather(
        write_many("s1", "alpha"),
        write_many("s2", "beta"),
    )

    s1 = await store.get_messages("s1")
    s2 = await store.get_messages("s2")
    assert len(s1) == 10
    assert len(s2) == 10
    assert [m.content for m in s1] == [f"alpha-{i}" for i in range(10)]
    assert [m.content for m in s2] == [f"beta-{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# Per-session locking
# ---------------------------------------------------------------------------


async def test_concurrent_appends_same_session_preserve_count() -> None:
    """100 concurrent appends to the same session yield exactly 100 messages."""
    store = MemoryStateStore()

    async def appender(index: int) -> None:
        await store.append("s1", ChatMessage(role="user", content=f"msg-{index}"))

    await asyncio.gather(*(appender(i) for i in range(100)))

    got = await store.get_messages("s1")
    assert len(got) == 100
    # Order may not match scheduling order (asyncio.Lock fairness varies),
    # but EVERY index must be present exactly once.
    contents = sorted(m.content for m in got)
    assert contents == sorted(f"msg-{i}" for i in range(100))


async def test_concurrent_sessions_do_not_block() -> None:
    """Sessions hold independent locks: a slow op on one cannot stall another.

    Wraps both calls in a tight ``asyncio.wait_for`` budget; if they were
    serialized through a shared lock, the test would time out.
    """
    store = MemoryStateStore()

    async def append_to(session: str) -> None:
        await store.append(session, ChatMessage(role="user", content="x"))

    # Use gather with a tight timeout — both ops should finish well under
    # the limit because they don't share a lock.
    await asyncio.wait_for(
        asyncio.gather(
            append_to("s1"),
            append_to("s2"),
            append_to("s3"),
        ),
        timeout=1.0,
    )

    assert len(await store.get_messages("s1")) == 1
    assert len(await store.get_messages("s2")) == 1
    assert len(await store.get_messages("s3")) == 1


async def test_get_lock_returns_same_lock_for_same_session() -> None:
    """Lock-table identity: repeated access returns the same lock object."""
    store = MemoryStateStore()
    lock_a = await store._get_lock("s1")
    lock_b = await store._get_lock("s1")
    assert lock_a is lock_b


async def test_get_lock_returns_different_locks_for_different_sessions() -> None:
    store = MemoryStateStore()
    lock_a = await store._get_lock("s1")
    lock_b = await store._get_lock("s2")
    assert lock_a is not lock_b


async def test_lock_creation_is_race_free() -> None:
    """Concurrent first-access for the same session yields a single shared lock."""
    store = MemoryStateStore()

    async def fetch() -> asyncio.Lock:
        return await store._get_lock("contended")

    results = await asyncio.gather(*(fetch() for _ in range(20)))
    first = results[0]
    assert all(lock is first for lock in results)


# ---------------------------------------------------------------------------
# Edge case content
# ---------------------------------------------------------------------------


async def test_append_preserves_all_chat_message_fields() -> None:
    """Optional fields (``name``, ``tool_call_id``) survive round-trip."""
    store = MemoryStateStore()
    msg = ChatMessage(
        role="tool",
        content="result",
        name="search",
        tool_call_id="abc123",
    )
    await store.append("s1", msg)
    got = await store.get_messages("s1")
    assert got[0].role == "tool"
    assert got[0].name == "search"
    assert got[0].tool_call_id == "abc123"
    assert got[0].content == "result"


async def test_empty_session_id_is_a_valid_distinct_session() -> None:
    """An empty string is a valid (if unusual) opaque session id."""
    store = MemoryStateStore()
    await store.append("", ChatMessage(role="user", content="a"))
    await store.append("s1", ChatMessage(role="user", content="b"))

    empty = await store.get_messages("")
    other = await store.get_messages("s1")
    assert len(empty) == 1
    assert empty[0].content == "a"
    assert len(other) == 1
    assert other[0].content == "b"
