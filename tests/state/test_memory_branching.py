"""Branching tests for :class:`MemoryStateStore` (BR-004 reference impl).

MemoryStateStore is the reference implementation of the branching contract;
these tests pin the semantics that SQL (M3) and Redis (M4) must match.
"""

from __future__ import annotations

import asyncio

import pytest

from fifty_agent_sdk import BranchInfo, ChatMessage, MemoryStateStore
from fifty_agent_sdk.state import TRUNK_BRANCH_ID


def _msg(content: str, role: str = "user") -> ChatMessage:
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


async def _seed(store: MemoryStateStore, session: str, *contents: str) -> None:
    """Append a sequence of user messages to the active branch."""
    for c in contents:
        await store.append(session, _msg(c))


def _contents(messages: list[ChatMessage]) -> list[str]:
    return [m.content for m in messages]


@pytest.fixture
def store() -> MemoryStateStore:
    """A fresh in-memory store (used by the truncate_after tests below)."""
    return MemoryStateStore()


# ---------------------------------------------------------------------------
# Implicit trunk
# ---------------------------------------------------------------------------


async def test_new_session_is_on_trunk() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    branches = await store.list_branches("s1")
    assert len(branches) == 1
    assert branches[0].branch_id == TRUNK_BRANCH_ID
    assert branches[0].parent_branch_id is None
    assert branches[0].forked_from_sequence is None
    assert branches[0].head_sequence == 2
    assert branches[0].is_active is True


async def test_unknown_session_list_branches_is_empty() -> None:
    store = MemoryStateStore()
    assert await store.list_branches("never") == []


async def test_unknown_session_get_messages_none_branch_is_empty() -> None:
    store = MemoryStateStore()
    assert await store.get_messages("never") == []


# ---------------------------------------------------------------------------
# fork: non-destructive, old line preserved
# ---------------------------------------------------------------------------


async def test_fork_preserves_original_and_inherits_prefix() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b", "c")  # trunk: 1=a 2=b 3=c

    branch = await store.fork("s1", from_sequence=2)  # inherit a,b
    # fork does NOT switch the active head.
    assert _contents(await store.get_messages("s1")) == ["a", "b", "c"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]

    # The new branch has inherited a,b and nothing else yet.
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]

    # Switch and append: original trunk stays intact.
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    assert _contents(await store.get_messages("s1")) == ["a", "b", "c2"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_fork_from_zero_starts_empty() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=0)
    assert await store.get_messages("s1", branch_id=branch) == []
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("fresh"))
    assert _contents(await store.get_messages("s1")) == ["fresh"]


async def test_fork_from_head_inherits_everything() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=3)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# switch_branch / active head
# ---------------------------------------------------------------------------


async def test_switch_changes_active_head_and_append_target() -> None:
    store = MemoryStateStore()
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


# ---------------------------------------------------------------------------
# Nested forks + fork from inherited history
# ---------------------------------------------------------------------------


async def test_nested_fork_materializes_through_lineage() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b", "c")  # trunk
    b1 = await store.fork("s1", from_sequence=3)  # inherit a,b,c
    await store.switch_branch("s1", b1)
    await store.append("s1", _msg("d"))  # b1: a,b,c,d
    b2 = await store.fork("s1", from_sequence=4)  # from b1 head
    await store.switch_branch("s1", b2)
    await store.append("s1", _msg("e"))
    assert _contents(await store.get_messages("s1", branch_id=b2)) == ["a", "b", "c", "d", "e"]


async def test_fork_from_inherited_portion_of_parent() -> None:
    """Fork a child at a sequence that lies in the parent's *inherited* prefix.

    trunk: a,b,c,d,e ; b1 forks trunk@3 (a,b,c) + adds x ; b2 forks b1@2 — which
    is inside b1's inherited prefix, so b2 must materialize to a,b + its own.
    """
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b", "c", "d", "e")
    b1 = await store.fork("s1", from_sequence=3)
    await store.switch_branch("s1", b1)
    await store.append("s1", _msg("x"))  # b1: a,b,c,x
    b2 = await store.fork("s1", from_sequence=2)  # within inherited a,b
    await store.switch_branch("s1", b2)
    await store.append("s1", _msg("y"))
    assert _contents(await store.get_messages("s1", branch_id=b2)) == ["a", "b", "y"]
    # b1 and trunk untouched.
    assert _contents(await store.get_messages("s1", branch_id=b1)) == ["a", "b", "c", "x"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == [
        "a",
        "b",
        "c",
        "d",
        "e",
    ]


# ---------------------------------------------------------------------------
# list_branches metadata
# ---------------------------------------------------------------------------


async def test_list_branches_reports_lineage_and_active() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=1)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c"))

    branches = {b.branch_id: b for b in await store.list_branches("s1")}
    assert set(branches) == {TRUNK_BRANCH_ID, branch}
    assert branches[TRUNK_BRANCH_ID].is_active is False
    assert branches[TRUNK_BRANCH_ID].head_sequence == 2
    fork_info = branches[branch]
    assert isinstance(fork_info, BranchInfo)
    assert fork_info.parent_branch_id == TRUNK_BRANCH_ID
    assert fork_info.forked_from_sequence == 1
    assert fork_info.head_sequence == 2  # inherited 1 + own 1
    assert fork_info.is_active is True


async def test_list_branches_trunk_first() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a")
    await store.fork("s1", from_sequence=1)
    await store.fork("s1", from_sequence=1)
    branches = await store.list_branches("s1")
    assert branches[0].branch_id == TRUNK_BRANCH_ID


# ---------------------------------------------------------------------------
# Validation / error contract
# ---------------------------------------------------------------------------


async def test_fork_unknown_session_raises() -> None:
    store = MemoryStateStore()
    with pytest.raises(ValueError, match="unknown session"):
        await store.fork("never", from_sequence=0)


async def test_fork_out_of_range_raises() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    with pytest.raises(ValueError, match="out of range"):
        await store.fork("s1", from_sequence=3)
    with pytest.raises(ValueError, match="out of range"):
        await store.fork("s1", from_sequence=-1)


async def test_switch_unknown_branch_raises() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a")
    with pytest.raises(ValueError, match="does not exist"):
        await store.switch_branch("s1", "no-such-branch")


async def test_get_messages_explicit_unknown_branch_raises() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a")
    with pytest.raises(ValueError, match="does not exist"):
        await store.get_messages("s1", branch_id="no-such-branch")


async def test_get_messages_unknown_session_explicit_branch_raises() -> None:
    store = MemoryStateStore()
    with pytest.raises(ValueError, match="unknown session"):
        await store.get_messages("never", branch_id="trunk")


# ---------------------------------------------------------------------------
# delete + defensive copy
# ---------------------------------------------------------------------------


async def test_delete_removes_all_branches() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a")
    await store.fork("s1", from_sequence=1)
    await store.delete("s1")
    assert await store.list_branches("s1") == []
    assert await store.get_messages("s1") == []


async def test_branch_read_is_defensive_copy() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=2)
    snapshot = await store.get_messages("s1", branch_id=branch)
    snapshot.clear()
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_concurrent_appends_to_active_branch_are_serialized() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "seed")
    await asyncio.gather(*(store.append("s1", _msg(f"m{i}")) for i in range(50)))
    msgs = await store.get_messages("s1")
    assert len(msgs) == 51  # seed + 50


async def test_concurrent_forks_create_distinct_branches() -> None:
    store = MemoryStateStore()
    await _seed(store, "s1", "a", "b")
    branch_ids = await asyncio.gather(*(store.fork("s1", from_sequence=1) for _ in range(20)))
    assert len(set(branch_ids)) == 20
    # trunk + 20 forks
    assert len(await store.list_branches("s1")) == 21


# ---------------------------------------------------------------------------
# truncate_after (BR-003)
# ---------------------------------------------------------------------------


async def test_truncate_after_keeps_le_n_drops_gt_n(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c", "d")
    await store.truncate_after("s1", 2)
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_is_idempotent(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    await store.truncate_after("s1", 1)
    await store.truncate_after("s1", 1)
    assert _contents(await store.get_messages("s1")) == ["a"]


async def test_truncate_after_unknown_session_is_noop(store: MemoryStateStore) -> None:
    await store.truncate_after("never", 0)
    assert await store.get_messages("never") == []


async def test_truncate_after_unknown_branch_is_noop(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 0, branch_id="no-such")
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_to_zero_empties_branch(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 0)
    assert await store.get_messages("s1") == []


async def test_truncate_after_at_or_beyond_head_is_noop(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    await store.truncate_after("s1", 5)
    assert _contents(await store.get_messages("s1")) == ["a", "b"]


async def test_truncate_after_on_fork_leaves_trunk_intact(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.append("s1", _msg("d2"))
    await store.truncate_after("s1", 3, branch_id=branch)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b", "c2"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_truncate_below_fork_point_keeps_inherited(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b", "c")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.truncate_after("s1", 1, branch_id=branch)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]
    assert _contents(await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)) == ["a", "b", "c"]


async def test_truncate_after_targets_active_branch_by_default(store: MemoryStateStore) -> None:
    await _seed(store, "s1", "a", "b")
    branch = await store.fork("s1", from_sequence=2)
    await store.switch_branch("s1", branch)
    await store.append("s1", _msg("c2"))
    await store.truncate_after("s1", 2)
    assert _contents(await store.get_messages("s1", branch_id=branch)) == ["a", "b"]


async def test_truncate_after_concurrent_with_append_is_safe(store: MemoryStateStore) -> None:
    import asyncio

    await _seed(store, "s1", "a", "b", "c")
    await asyncio.gather(
        store.append("s1", _msg("d")),
        store.truncate_after("s1", 1),
    )
    msgs = _contents(await store.get_messages("s1"))
    assert msgs[0] == "a"
    assert len(msgs) in (1, 2)
