"""Branching tests for :class:`SqlStateStore` (BR-004 M3).

Mirrors the MemoryStateStore reference scenarios against the SQL backend
(in-memory aiosqlite), and adds SQL-specific coverage for zero-migration
resilience: a pre-BR-004 session (message rows, no ``agent_branches`` record)
must read as an implicit trunk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import BranchInfo, ChatMessage, SqlStateStore
from fifty_agent_sdk.state import TRUNK_BRANCH_ID
from fifty_agent_sdk.state.sql import AgentMessage, AgentSession, sql_metadata


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def store(engine: AsyncEngine) -> AsyncIterator[SqlStateStore]:
    s = SqlStateStore(engine)
    try:
        yield s
    finally:
        await s.aclose()


def _msg(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


async def _seed(store: SqlStateStore, session: str, *contents: str) -> None:
    for c in contents:
        await store.append(session, _msg(c))


def _contents(messages: list[ChatMessage]) -> list[str]:
    return [m.content for m in messages]


# ---------------------------------------------------------------------------
# Trunk + fork semantics (mirrors the Memory reference)
# ---------------------------------------------------------------------------


async def test_new_session_is_on_trunk(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    branches = await store.list_branches("s1")
    assert len(branches) == 1
    assert branches[0].branch_id == TRUNK_BRANCH_ID
    assert branches[0].head_sequence == 2
    assert branches[0].is_active is True


async def test_fork_preserves_original_and_inherits_prefix(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=2)
    assert _contents(await store.get_messages("s1")) == ["a", "b", "c"]
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]

    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    assert _contents(await store.get_messages("s1")) == ["a", "b", "c2"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_fork_from_zero_starts_empty(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=0)
    assert await store.get_messages("s1", branch_id=branch) == []
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("fresh"))
    assert _contents(await store.get_messages("s1")) == ["fresh"]


async def test_switch_changes_active_head_and_append_target(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a")
    branch = await store.fork("s1", from_sequence=1)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("b-on-fork"))
    await store.switch_branch("s1", TRUNK_BRANCH_ID)
    await store.append("s1", _msg("b-on-trunk"))
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == [
        "a",
        "b-on-trunk",
    ]
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b-on-fork"]


async def test_nested_fork_materializes_through_lineage(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    b1 = await store.fork("s1", from_sequence=3)
    await store.switch_branch("s1", b1)
    await store.append("s1", _msg("d"))
    b2 = await store.fork("s1", from_sequence=4)
    await store.switch_branch("s1", b2)
    await store.append("s1", _msg("e"))
    assert _contents(await store.get_messages("s1", branch_id=b2)) == ["a", "b", "c", "d", "e"]


async def test_fork_from_inherited_portion_of_parent(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c", "d", "e")
    b1 = await store.fork("s1", from_sequence=3)
    await store.switch_branch("s1", b1)
    await store.append("s1", _msg("x"))  # b1: a,b,c,x
    b2 = await store.fork("s1", from_sequence=2)  # within inherited a,b
    await store.switch_branch("s1", b2)
    await store.append("s1", _msg("y"))
    assert _contents(await store.get_messages("s1", branch_id=b2)) == ["a", "b", "y"]
    assert _contents(await store.get_messages("s1", branch_id=b1)) == ["a", "b", "c", "x"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]


async def test_list_branches_reports_lineage_and_active(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=1)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c"))
    branches = {b.branch_id: b for b in await store.list_branches("s1")}
    assert set(branches) == {TRUNK_BRANCH_ID, branch}
    fork_info = branches[branch]
    assert isinstance(fork_info, BranchInfo)
    assert fork_info.parent_branch_id == TRUNK_BRANCH_ID
    assert fork_info.forked_from_sequence == 1
    assert fork_info.head_sequence == 2
    assert fork_info.is_active is True
    assert branches[TRUNK_BRANCH_ID].is_active is False


async def test_list_branches_trunk_first(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a")
    await store.fork("s1", from_sequence=1)
    await store.fork("s1", from_sequence=1)
    branches = await store.list_branches("s1")
    assert branches[0].branch_id == TRUNK_BRANCH_ID
    assert len(branches) == 3


# ---------------------------------------------------------------------------
# Validation / error contract
# ---------------------------------------------------------------------------


async def test_fork_unknown_session_raises(store: SqlStateStore) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        await store.fork("never", from_sequence=0)


async def test_fork_out_of_range_raises(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    with pytest.raises(ValueError, match="out of range"):
        await store.fork("s1", from_sequence=3)


async def test_switch_unknown_branch_raises(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a")
    with pytest.raises(ValueError, match="does not exist"):
        await store.switch_branch("s1", "no-such-branch")


async def test_get_messages_explicit_unknown_branch_raises(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a")
    with pytest.raises(ValueError, match="does not exist"):
        await store.get_messages("s1", branch_id="no-such-branch")


async def test_get_messages_unknown_session_none_branch_is_empty(store: SqlStateStore) -> None:
    assert await store.get_messages("never") == []
    assert await store.list_branches("never") == []


async def test_delete_removes_all_branches(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a")
    await store.fork("s1", from_sequence=1)
    await store.delete("s1")
    assert await store.list_branches("s1") == []
    assert await store.get_messages("s1") == []


# ---------------------------------------------------------------------------
# Zero-migration resilience: a pre-BR-004 session has no agent_branches row
# ---------------------------------------------------------------------------


async def test_legacy_session_without_branch_row_reads_as_trunk(
    store: SqlStateStore, engine: AsyncEngine
) -> None:
    """Insert a session + messages directly (no agent_branches row, no
    active_branch_id) to simulate pre-BR-004 data, then exercise the API."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session, session.begin():
        await session.execute(insert(AgentSession).values(session_id="legacy"))
        await session.execute(
            insert(AgentMessage).values(
                session_id="legacy",
                branch_id=TRUNK_BRANCH_ID,
                sequence=1,
                role="user",
                content="old",
            )
        )
        await session.execute(
            insert(AgentMessage).values(
                session_id="legacy",
                branch_id=TRUNK_BRANCH_ID,
                sequence=2,
                role="user",
                content="old2",
            )
        )

    # Reads synthesize the implicit trunk.
    assert _contents(await store.get_messages("legacy")) == ["old", "old2"]
    branches = await store.list_branches("legacy")
    assert len(branches) == 1
    assert branches[0].branch_id == TRUNK_BRANCH_ID
    assert branches[0].head_sequence == 2

    # And it can be forked / appended without a pre-existing branch row.
    branch = await store.fork("legacy", from_sequence=1)
    await store.switch_branch("legacy", branch)
    await store.append("legacy", _msg("new"))
    assert _contents(await store.get_messages("legacy", branch_id=branch)) == ["old", "new"]
    assert _contents(await store.get_messages("legacy", branch_id=TRUNK_BRANCH_ID)) == [
        "old",
        "old2",
    ]


# ---------------------------------------------------------------------------
# truncate_after (BR-003)
# ---------------------------------------------------------------------------


async def test_truncate_after_keeps_le_n_drops_gt_n(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c", "d")
    await store.truncate_after("s1", 2)
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_is_idempotent(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    await store.truncate_after("s1", 1)
    await store.truncate_after("s1", 1)
    assert _contents(await store.get_messages("s1")) == ["a"]


async def test_truncate_after_unknown_session_is_noop(store: SqlStateStore) -> None:
    await store.truncate_after("never", 0)
    assert await store.get_messages("never") == []


async def test_truncate_after_unknown_branch_is_noop(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 0, branch_id="no-such")
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_to_zero_empties_branch(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 0)
    assert await store.get_messages("s1") == []


async def test_truncate_after_at_or_beyond_head_is_noop(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 5)
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_on_fork_leaves_trunk_intact(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.append("s1", _msg("d2"))
    await store.truncate_after("s1", 3, branch_id=branch)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b", "c2"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_truncate_below_fork_point_keeps_inherited(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.truncate_after("s1", 1, branch_id=branch)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_truncate_after_targets_active_branch_by_default(store: SqlStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.truncate_after("s1", 2)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]


async def test_truncate_after_concurrent_with_append_is_safe(store: SqlStateStore) -> None:
    import asyncio

    await _seed(store, "s1", "a", "b", "c")
    await asyncio.gather(
        store.append("s1", _msg("d")),
        store.truncate_after("s1", 1),
    )
    msgs = _contents(await store.get_messages("s1"))
    assert msgs[0] == "a"
    assert len(msgs) in (1, 2)
