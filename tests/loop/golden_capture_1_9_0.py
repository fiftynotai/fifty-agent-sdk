"""Golden capture of the 1.9.0 request bodies AND event stream (FR-003 AC-4).

FR-003 adds ``AgentLoop(interventions=...)``. Its compatibility promise is that
a loop built WITHOUT ``interventions`` sends the same request bodies (same
keys, values and JSON types) and emits the same event stream as 1.9.0 (since
1.10.2 claimed only for the runs :mod:`fifty_agent_sdk.loop` scopes it to:
"Non-ASCII text (BR-020)", "Error-path final text (BR-021)" and
"Tool-argument nesting (BR-019)"). The two
earlier fixtures (``legacy_1_7_0.json``, ``requests_1_8_0.json``) pin request
bodies only, and none of their scenarios drives the branches FR-003 edits:

* a native multi-call batch with ``ToolNotFound`` and ``ToolTimeout`` members;
* single-call ``is_error`` and ``ToolTimeout`` observations in the PROSE,
  JSON-``user`` and legacy ``"tool"`` roles;
* 1.9.0's own ``reasoning_effort`` / ``temperature`` request shapes.

FR-003 also edits the event-emitting dispatch blocks themselves (the
``call_id`` hoist, ``before_tool`` placement, ``after_tool`` after the
terminal yield), so each scenario here records the event stream as well as
the bodies: ``{"bodies": [...], "events": [...]}``.

* :data:`SCENARIOS_1_9_0` maps a stable key to a builder. Each builder makes a
  fresh registry and fresh fakes, and none passes ``interventions``: the
  kwarg did not exist when the fixture was captured.
* Bodies go through the REAL wire translator
  (:func:`tests.loop.golden_capture.wire_bodies`, which calls
  ``OpenAICompatibleClient._build_body`` and does no I/O), so the system
  prompt is pinned as the first message of every body.
* Each event is ``model_dump(mode="json")`` without ``timestamp``.
* :func:`_normalise_scenario` rewrites every call id to ``<id-N>`` with ONE
  first-appearance mapping over the bodies, then the events. Pairing
  structure survives (assistant ``tool_calls[].id``, ``role="tool"``
  ``tool_call_id`` and event ``call_id`` share a placeholder when they share
  an id), and the result does not depend on how many times ``uuid4`` runs.

It does NOT pin non-ASCII text or U+007F in the JSON the SDK writes for the
model, which 1.10.2 changed (BR-020): no scenario's tool results, tool schemas
or replayed arguments hold any. Nor does it pin any error path: no scenario
ends on an ``ErrorEvent``, so the error-path final text, message and context
that 1.10.2 changed (BR-021) are outside it. Nor any text the 64-level
nesting check refuses (BR-019): it refuses no text in any scenario, whose
tool arguments nest at most 1 level.

Running ``python -m tests.loop.golden_capture_1_9_0`` from the repo root writes
``tests/loop/golden/requests_1_9_0.json``. The fixture in the tree was written
from the UNMODIFIED 1.9.0 source (commit ``50560a3``) before any FR-003 edit
under ``src/``; its size, sha256, capture time and PyPI-wheel cross-check are
recorded in the FR-003 evidence file. It must never be regenerated to make a
failing diff pass: a diff means a no-intervention request or event moved.
Never run the older writers either (``python -m tests.loop.golden_capture``,
``python -m tests.loop.golden_capture_1_8_0``): they would rewrite the 1.7.0
and 1.8.0 fixtures from newer source and destroy their provenance.

This file is deliberately NOT named ``test_*.py`` so pytest does not collect
it; :mod:`tests.loop.test_golden_1_9_0` imports it, and the FR-003 differential
tests import :func:`serialise_events` and :func:`_normalise_scenario` from it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentEvent,
    AgentLoop,
    ChatMessage,
    JsonModeParser,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    ToolResult,
)
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)
from tests.loop.golden_capture import render, wire_bodies

GOLDEN_1_9_0_PATH = Path(__file__).parent / "golden" / "requests_1_9_0.json"
"""Where the fixture lives. Written once, from 1.9.0, by ``__main__`` below."""

_PERSONA = "You are a careful research assistant."
_MODEL = "golden-model"
_USER_INPUT = [ChatMessage(role="user", content="Find the FR-003 design notes.")]

LoopScenario = tuple[AgentLoop, FakeLLMClient, list[ChatMessage]]
"""A built loop, the fake it records requests on, and the caller's input messages."""

# The four observation kinds every "observation_kinds" scenario walks through:
# a success, an is_error result, an unregistered name and a timeout.
_KINDS: list[tuple[str, dict[str, Any]]] = [
    ("ok", {"q": "rows"}),
    ("bad", {"q": "fail"}),
    ("missing", {"q": "ghost"}),
    ("slow", {"q": "wait"}),
]


# ---------------------------------------------------------------------------
# Deterministic registry, safety and scripted replies
# ---------------------------------------------------------------------------


def _registry() -> Registry:
    """``ok`` succeeds, ``bad`` returns ``is_error``, ``slow`` outlives the timeout.

    ``missing`` is deliberately NOT registered, so it takes the ToolNotFound path.
    """
    registry = Registry()
    registry.register(FakeTool("ok", result=ToolResult(output={"rows": [{"id": 1}, {"id": 2}]})))
    registry.register(
        FakeTool("bad", result=ToolResult(is_error=True, error="upstream returned 503"))
    )
    registry.register(FakeTool("slow", sleep_seconds=1.0))
    return registry


def _safety(**extra: Any) -> SafetyConfig:
    """A 0.05 s tool timeout, so ``slow`` (1.0 s) always ends in ToolTimeout."""
    return SafetyConfig(tool_timeout_seconds=0.05, **extra)


def _json_tool(name: str, args: dict[str, Any]) -> str:
    return json.dumps(
        {
            "thought": f"calling {name}",
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def _json_final(answer: str) -> str:
    return json.dumps(
        {
            "thought": "done",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def _prose_tool(name: str, args: dict[str, Any]) -> str:
    return f"Thought: calling {name}\nAction: {name}\nAction Input: {json.dumps(args)}"


def _prose_final(answer: str) -> str:
    return f"Thought: done\nFinal Answer: {answer}"


def _mode_loop(llm: FakeLLMClient, tool_mode: ToolMode, **extra: Any) -> AgentLoop:
    """A loop with an explicit ``tool_mode`` and no ``interventions``."""
    safety = extra.pop("safety", None)
    return AgentLoop(
        llm=llm,
        registry=_registry(),
        prompts=PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else _safety(),
        model=_MODEL,
        tool_mode=tool_mode,
        **extra,
    )


def _legacy_loop(llm: FakeLLMClient, **extra: Any) -> AgentLoop:
    """A loop with the legacy call shape (``parser=``, no ``tool_mode``, no ``interventions``)."""
    safety = extra.pop("safety", None)
    return AgentLoop(
        llm=llm,
        registry=_registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else _safety(),
        model=_MODEL,
        output_format=JSON_MODE_OUTPUT_FORMAT,
        **extra,
    )


def _json_kinds_llm() -> FakeLLMClient:
    """ok -> bad -> missing -> slow -> final, as JSON envelopes."""
    replies: list[Any] = [make_response(_json_tool(name, args)) for name, args in _KINDS]
    replies.append(make_response(_json_final("all four seen")))
    return FakeLLMClient(replies)


# ---------------------------------------------------------------------------
# Scenarios: 1.9.0 call shapes only, no interventions
# ---------------------------------------------------------------------------


def _json_observation_kinds() -> LoopScenario:
    llm = _json_kinds_llm()
    return _mode_loop(llm, ToolMode.JSON), llm, list(_USER_INPUT)


def _json_user_role_observation_kinds() -> LoopScenario:
    llm = _json_kinds_llm()
    return _mode_loop(llm, ToolMode.JSON, tool_message_role="user"), llm, list(_USER_INPUT)


def _prose_observation_kinds() -> LoopScenario:
    replies: list[Any] = [make_response(_prose_tool(name, args)) for name, args in _KINDS]
    replies.append(make_response(_prose_final("all four seen")))
    llm = FakeLLMClient(replies)
    return _mode_loop(llm, ToolMode.PROSE), llm, list(_USER_INPUT)


def _legacy_tool_role_observation_kinds() -> LoopScenario:
    llm = _json_kinds_llm()
    return _legacy_loop(llm), llm, list(_USER_INPUT)


def _legacy_assistant_role_observation_kinds() -> LoopScenario:
    llm = _json_kinds_llm()
    return _legacy_loop(llm, tool_message_role="assistant"), llm, list(_USER_INPUT)


def _native_single_observation_kinds() -> LoopScenario:
    replies: list[Any] = [make_multi_tool_response([(name, args)]) for name, args in _KINDS]
    replies.append(make_response("all four seen"))
    llm = FakeLLMClient(replies)
    return _mode_loop(llm, ToolMode.NATIVE), llm, list(_USER_INPUT)


def _native_batch_observation_kinds() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response(list(_KINDS), content="checking all four"),
            make_response("batch seen"),
        ]
    )
    safety = _safety(max_concurrent_tool_calls=4)
    return _mode_loop(llm, ToolMode.NATIVE, safety=safety), llm, list(_USER_INPUT)


def _legacy_native_flag_batch_observation_kinds() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response(list(_KINDS)),
            make_response(_json_final("batch seen")),
        ]
    )
    safety = _safety(native_tools_enabled=True, max_concurrent_tool_calls=4)
    return _legacy_loop(llm, safety=safety), llm, list(_USER_INPUT)


def _json_stream_observation_kinds() -> LoopScenario:
    final = _json_final("streamed after two tools")
    llm = FakeLLMClient(
        [
            make_stream_chunks([_json_tool("ok", {"q": "rows"})]),
            make_stream_chunks([_json_tool("bad", {"q": "fail"})]),
            make_stream_chunks([final[:25], final[25:]]),
        ]
    )
    return _mode_loop(llm, ToolMode.JSON, stream=True), llm, list(_USER_INPUT)


def _json_request_options() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("ok", {"q": "rows"})),
            make_response(_json_final("with options")),
        ]
    )
    loop = _mode_loop(llm, ToolMode.JSON, reasoning_effort="medium", temperature=0.4)
    return loop, llm, list(_USER_INPUT)


def _native_request_options() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("ok", {"q": "rows"})]),
            make_response("with options"),
        ]
    )
    loop = _mode_loop(llm, ToolMode.NATIVE, reasoning_effort="low", temperature=None)
    return loop, llm, list(_USER_INPUT)


SCENARIOS_1_9_0: dict[str, Callable[[], LoopScenario]] = {
    "json_observation_kinds": _json_observation_kinds,
    "json_user_role_observation_kinds": _json_user_role_observation_kinds,
    "prose_observation_kinds": _prose_observation_kinds,
    "legacy_tool_role_observation_kinds": _legacy_tool_role_observation_kinds,
    "legacy_assistant_role_observation_kinds": _legacy_assistant_role_observation_kinds,
    "native_single_observation_kinds": _native_single_observation_kinds,
    "native_batch_observation_kinds": _native_batch_observation_kinds,
    "legacy_native_flag_batch_observation_kinds": _legacy_native_flag_batch_observation_kinds,
    "json_stream_observation_kinds": _json_stream_observation_kinds,
    "json_request_options": _json_request_options,
    "native_request_options": _native_request_options,
}
"""Scenario key -> builder. These are the fixture's top-level keys."""


# ---------------------------------------------------------------------------
# Serialisation and id normalisation
# ---------------------------------------------------------------------------


def serialise_events(events: list[AgentEvent]) -> list[dict[str, Any]]:
    """``model_dump(mode="json")`` of each event, without the wall-clock ``timestamp``."""
    return [event.model_dump(mode="json", exclude={"timestamp"}) for event in events]


def _normalise_scenario(
    bodies: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Replace every call id with ``<id-N>``, first appearance over ``bodies`` then ``events``.

    ONE mapping covers ``tool_calls[].id`` and ``tool_call_id`` in the bodies
    and ``call_id`` in the events, so an event whose ``call_id`` equals a wire
    ``tool_call_id`` gets the same placeholder. The inputs are not mutated.
    """
    mapping: dict[str, str] = {}

    def _placeholder(raw: str) -> str:
        if raw not in mapping:
            mapping[raw] = f"<id-{len(mapping) + 1}>"
        return mapping[raw]

    new_bodies: list[dict[str, Any]] = json.loads(json.dumps(bodies))
    for body in new_bodies:
        for message in body["messages"]:
            for entry in message.get("tool_calls") or []:
                if entry.get("id") is not None:
                    entry["id"] = _placeholder(entry["id"])
            if message.get("tool_call_id") is not None:
                message["tool_call_id"] = _placeholder(message["tool_call_id"])
    new_events: list[dict[str, Any]] = json.loads(json.dumps(events))
    for event in new_events:
        if event.get("call_id") is not None:
            event["call_id"] = _placeholder(event["call_id"])
    return {"bodies": new_bodies, "events": new_events}


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def capture_scenario_1_9_0(key: str) -> dict[str, list[dict[str, Any]]]:
    """Run scenario ``key`` and return its id-normalised wire bodies and events."""
    loop, llm, messages = SCENARIOS_1_9_0[key]()
    events = [event async for event in loop.run(messages)]
    bodies = await wire_bodies(llm, stream=loop._stream)
    return _normalise_scenario(bodies, serialise_events(events))


async def capture_all_1_9_0() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Capture every scenario, keyed as in :data:`SCENARIOS_1_9_0`."""
    return {key: await capture_scenario_1_9_0(key) for key in sorted(SCENARIOS_1_9_0)}


if __name__ == "__main__":
    GOLDEN_1_9_0_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_1_9_0_PATH.write_text(render(asyncio.run(capture_all_1_9_0())), encoding="utf-8")
    print(f"wrote {GOLDEN_1_9_0_PATH}")
