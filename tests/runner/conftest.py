"""Shared helpers for the runner integration tests.

Reuses :class:`tests.loop.conftest.FakeLLMClient` and
:class:`tests.loop.conftest.FakeTool` so the runner tests exercise the
SAME plumbing the loop integration tests do — keeping a single source of
truth for LLM/tool stand-ins.

The :func:`_make_runner` helper composes a full
``loop + state + runner`` stack mirroring the shape of
:func:`tests.loop.test_loop._make_loop`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent_sdk import (
    AgentEvent,
    AgentLoop,
    AgentRunner,
    JsonModeParser,
    MemoryStateStore,
    PromptSections,
    Registry,
    SafetyConfig,
)
from agent_sdk.state.protocol import StateStore
from tests.loop.conftest import FakeLLMClient


def final_json(answer: str, thought: str = "done") -> str:
    """Render a JSON envelope for a :class:`FinalAnswer` parse."""
    return json.dumps(
        {
            "thought": thought,
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def tool_json(
    thought: str, name: str, args: dict[str, Any] | None
) -> str:
    """Render a JSON envelope for a :class:`ThoughtAction` parse."""
    return json.dumps(
        {
            "thought": thought,
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def make_runner(
    *,
    llm: FakeLLMClient,
    registry: Registry | None = None,
    state: StateStore | None = None,
    safety: SafetyConfig | None = None,
    system_prompt: str | None = None,
    persona: str = "You are a helpful agent.",
) -> tuple[AgentRunner, StateStore]:
    """Build a ready-to-drive :class:`AgentRunner` and return it with its store.

    The store is returned alongside the runner so tests can assert on
    persisted history without reaching into private state.
    """
    loop = AgentLoop(
        llm=llm,
        registry=registry if registry is not None else Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=persona),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
    )
    store = state if state is not None else MemoryStateStore()
    runner = AgentRunner(
        loop=loop, state=store, system_prompt=system_prompt
    )
    return runner, store


async def collect(iterator: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    """Drain an :class:`AsyncIterator[AgentEvent]` to a list."""
    return [event async for event in iterator]
