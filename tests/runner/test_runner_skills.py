"""Runner tests for a loop built with ``skills=`` (FR-004, amended AC D8; plan R1-R3).

The amended criterion: the catalog reaches every turn's request through
:class:`AgentRunner`, and nothing from a skill is persisted to the state
store. The catalog is part of the loop's structured prompt, which a Runner
never persists; a loaded skill body is a tool roundtrip, which a Runner never
persists either.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import (
    AgentLoop,
    AgentRunner,
    ChatMessage,
    MemoryStateStore,
    PromptSections,
    Registry,
    SafetyConfig,
    StateStore,
    ToolMode,
)
from fifty_agent_sdk.skills import Skill, SkillSet
from fifty_agent_sdk.state.sql import SqlStateStore, sql_metadata
from tests.loop.conftest import FakeLLMClient, make_response
from tests.runner.conftest import collect, final_json, tool_json

_KICKOFF = "You are the support desk."
_BODY = "SKILL-BODY: read the refund policy, then answer."
_DESCRIPTION = "Answer refund questions."


def _skills() -> SkillSet:
    return SkillSet([Skill(name="refunds", description=_DESCRIPTION, body=_BODY)])


@asynccontextmanager
async def _open_store(kind: str) -> AsyncIterator[StateStore]:
    """A fresh store: memory, or SQL on in-memory aiosqlite."""
    if kind == "memory":
        yield MemoryStateStore()
        return
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(sql_metadata.create_all)
    try:
        yield SqlStateStore(engine)
    finally:
        await engine.dispose()


async def _two_turns(store: StateStore) -> FakeLLMClient:
    """Turn 1 loads the skill then answers; turn 2 answers directly."""
    llm = FakeLLMClient(
        [
            make_response(tool_json("look it up", "load_skill", {"name": "refunds"})),
            make_response(final_json("Refunds take 5 days.")),
            make_response(final_json("Yes, by card.")),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        registry=Registry(),
        prompts=PromptSections(persona="Persona."),
        safety=SafetyConfig(),
        model="m",
        tool_mode=ToolMode.JSON,
        skills=_skills(),
    )
    runner = AgentRunner(loop=loop, state=store, system_prompt=_KICKOFF)
    await collect(runner.run("s1", "How long do refunds take?"))
    await collect(runner.run("s1", "Can I get it on my card?"))
    return llm


def _stored(messages: list[ChatMessage]) -> list[tuple[str, str | None]]:
    return [(message.role, message.content) for message in messages]


_EXPECTED_STORED = [
    ("system", _KICKOFF),
    ("user", "How long do refunds take?"),
    ("assistant", final_json("Refunds take 5 days.")),
    ("user", "Can I get it on my card?"),
    ("assistant", final_json("Yes, by card.")),
]
"""What the store holds: the kickoff, each user turn and each raw final completion."""


async def test_catalog_reaches_every_turn_and_nothing_from_a_skill_is_stored() -> None:
    """R1: each request starts with the loop's catalog; the store holds the five turn messages only."""
    async with _open_store("memory") as store:
        llm = await _two_turns(store)
        stored = await store.get_messages("s1")
    catalog = _skills().render_catalog()
    assert len(llm.calls) == 3
    for request in llm.calls:
        assert request.messages[0].role == "system"
        assert f"# Skills\n{catalog}" in str(request.messages[0].content)
    assert _stored(stored) == _EXPECTED_STORED
    stored_text = json.dumps([message.model_dump(mode="json") for message in stored])
    for fragment in (_DESCRIPTION, "# Skills", "load_skill", _BODY):
        assert fragment not in stored_text


async def test_a_body_loaded_on_turn_1_is_not_in_the_turn_2_request() -> None:
    """R2: tool roundtrips are not persisted, so turn 2 does not resend the body (the model reloads it)."""
    async with _open_store("memory") as store:
        llm = await _two_turns(store)
    assert _BODY in json.dumps([m.model_dump(mode="json") for m in llm.calls[1].messages])
    turn_2 = json.dumps([m.model_dump(mode="json") for m in llm.calls[2].messages])
    assert _BODY not in turn_2
    assert [message.role for message in llm.calls[2].messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.parametrize("kind", ["sql"])
async def test_a_durable_store_holds_the_same_rows(kind: str) -> None:
    """R3: ``SqlStateStore`` on aiosqlite stores exactly what the memory store stores."""
    async with _open_store(kind) as store:
        await _two_turns(store)
        stored = await store.get_messages("s1")
    assert _stored(stored) == _EXPECTED_STORED
