"""Canonical end-to-end worked example for the fifty-agent-sdk.

This single test is the executable companion to the README Quickstart. It
walks the full happy path a consumer follows — define a tool, wire it into a
registry, build the loop and runner, drive one turn, and consume the typed
event stream — and asserts on every observable property of the run.

Unlike the focused integration suites under ``tests/loop`` and
``tests/runner``, this file is deliberately written to read as documentation:
each step is heavily commented so a newcomer can follow the SDK's shape from
top to bottom. It reuses the established test fakes (``FakeLLMClient`` and the
JSON-envelope helpers) rather than inventing new ones, so it stays in lockstep
with the rest of the suite. The LLM is scripted, so the test is fully
deterministic — no network, no flakiness.
"""

from __future__ import annotations

import re
from typing import Any

import fifty_agent_sdk
from fifty_agent_sdk import (
    ActionEvent,
    AgentLoop,
    AgentRunner,
    FinalEvent,
    JsonModeParser,
    MemoryStateStore,
    ObservationEvent,
    PromptSections,
    Registry,
    SafetyConfig,
    ThoughtEvent,
    ToolStartedEvent,
    tool,
)
from tests.loop.conftest import FakeLLMClient, make_response
from tests.runner.conftest import final_json, tool_json


async def test_e2e_define_tool_wire_registry_run_stream() -> None:
    """Define a tool, wire a runner, drive one turn, and verify the stream.

    Mirrors the README Quickstart end to end: a single tool-using turn that
    produces the canonical ReACT event sequence and a clean two-message
    conversation history. Every assertion below pins one observable property
    of the run so this test doubles as a regression fence for the public API.
    """
    # ── Step 0: the SDK exposes a well-formed version string ──────────────
    # Assert __version__ exists and is PEP 440-shaped rather than pinning a
    # frozen literal: the old `== "0.1.0"` silently rotted across the
    # 0.1.0 -> 1.0.0 release bump (TD-026). importlib.metadata.version is
    # deliberately NOT used as the oracle here — the editable install's dist
    # metadata is stale (reports 0.0.1), so it would diverge from __version__.
    assert isinstance(fifty_agent_sdk.__version__, str) and fifty_agent_sdk.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", fifty_agent_sdk.__version__)

    # ── Step 1: define a real tool with the @tool decorator ───────────────
    # @tool derives the JSON Schema for `city` from the type annotation and
    # uses the docstring as the tool description. The function must be async.
    @tool()
    async def get_weather(city: str) -> dict[str, Any]:
        """Return the current weather for a city."""
        return {"city": city, "temp_c": 21}

    # ── Step 2: register the tool into a Registry ─────────────────────────
    # The Registry is the dispatch table the ReACT loop talks to; it keys
    # tools by name (here, "get_weather", the function name).
    registry = Registry()
    registry.register(get_weather)

    # ── Step 3: script the LLM and build the loop + runner ────────────────
    # The fake LLM replays two scripted JSON envelopes in order:
    #   reply 1 — a tool call asking for get_weather(city="Paris")
    #   reply 2 — a final answer
    # `tool_json` / `final_json` render the exact JSON the JsonModeParser
    # expects, so the loop parses them into a ThoughtAction then a
    # FinalAnswer respectively.
    llm = FakeLLMClient(
        replies=[
            make_response(tool_json("checking weather", "get_weather", {"city": "Paris"})),
            make_response(final_json("It is 21°C in Paris.")),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        parser=JsonModeParser(),
        prompts=PromptSections(persona="You are a helpful weather assistant."),
        safety=SafetyConfig(),
        model="test-model",
    )
    # The runner wraps the loop with conversation-state persistence. A
    # MemoryStateStore is enough for an in-process run; durable backends
    # (SqlStateStore / RedisStateStore) sit behind optional extras.
    store = MemoryStateStore()
    runner = AgentRunner(loop=loop, state=store)

    # ── Step 4: drive one turn and drain the event stream ─────────────────
    events = [event async for event in runner.run("session-e2e", "What's the weather in Paris?")]

    # ── Step 5: assert on every observable property of the run ────────────
    # 5a. The event-type sequence is the canonical tool-using ReACT cycle:
    #     a thought, the chosen action, the tool dispatch, the observation,
    #     a second thought, then the terminal final answer.
    assert [type(e) for e in events] == [
        ThoughtEvent,
        ActionEvent,
        ToolStartedEvent,
        ObservationEvent,
        ThoughtEvent,
        FinalEvent,
    ]

    # 5b. Sequence numbers are monotonic and dense, starting at 0 — the
    #     contract that lets a consumer detect a dropped or reordered event.
    assert [e.sequence for e in events] == list(range(len(events)))

    # 5c. The run ends with exactly one FinalEvent carrying the answer text.
    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "It is 21°C in Paris."

    # 5d. The ActionEvent carries the tool name and args the model chose.
    action_event = events[1]
    assert isinstance(action_event, ActionEvent)
    assert action_event.tool_name == "get_weather"
    assert action_event.args == {"city": "Paris"}

    # 5e. The ObservationEvent carries the tool's real return value, wrapped
    #     in a ToolResult.
    observation_event = events[3]
    assert isinstance(observation_event, ObservationEvent)
    assert observation_event.result.output == {"city": "Paris", "temp_c": 21}

    # 5f. The LLM was called exactly twice — one tool turn, one final turn.
    assert len(llm.calls) == 2

    # 5g. State round-trip: the runner persisted exactly the durable
    #     conversation — the user message and the final assistant answer.
    #     Tool roundtrips live only in the loop's working list, never state.
    history = await store.get_messages("session-e2e")
    assert [m.role for m in history] == ["user", "assistant"]
