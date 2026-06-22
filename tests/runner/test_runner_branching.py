"""Runner wire-through for conversation branching (BR-004 M5).

The runner has NO branch-specific logic: it persists to and reads from the
store's *active* branch via the default ``get_messages`` / ``append`` calls.
These tests prove that a consumer can fork + switch a session out of band and
the runner transparently continues on the active head — the zero-change
wire-through the brief specifies (orchestration stays in the consumer).
"""

from __future__ import annotations

from fifty_agent_sdk import TRUNK_BRANCH_ID
from tests.loop.conftest import FakeLLMClient, make_response
from tests.runner.conftest import collect, final_json, make_runner


async def test_runner_follows_active_branch() -> None:
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("a1")),
            make_response(final_json("a2")),
        ]
    )
    runner, store = make_runner(llm=llm)

    await collect(runner.run("s1", "q1"))
    head = len(await store.get_messages("s1"))  # user q1 + assistant a1

    # Consumer forks at the current head and switches onto the new branch.
    branch = await store.fork("s1", from_sequence=head)
    await store.switch_branch("s1", branch)

    # The runner — unchanged — now reads and writes the active (forked) branch.
    await collect(runner.run("s1", "q2"))

    trunk = await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)
    active = await store.get_messages("s1")
    assert [m.content for m in trunk if m.role == "user"] == ["q1"]
    assert [m.content for m in active if m.role == "user"] == ["q1", "q2"]
    assert len(trunk) == head
    assert len(active) == head + 2  # inherited turn + (user q2, assistant a2)


async def test_runner_on_trunk_is_unaffected_by_an_idle_fork() -> None:
    """Forking without switching leaves the runner on the trunk."""
    llm = FakeLLMClient(
        replies=[
            make_response(final_json("a1")),
            make_response(final_json("a2")),
        ]
    )
    runner, store = make_runner(llm=llm)

    await collect(runner.run("s1", "q1"))
    await store.fork("s1", from_sequence=1)  # fork but do NOT switch
    await collect(runner.run("s1", "q2"))

    # Still on trunk: both turns landed on the trunk.
    trunk = await store.get_messages("s1", branch_id=TRUNK_BRANCH_ID)
    assert [m.content for m in trunk if m.role == "user"] == ["q1", "q2"]
    branches = await store.list_branches("s1")
    assert len(branches) == 2
    assert next(b for b in branches if b.is_active).branch_id == TRUNK_BRANCH_ID
