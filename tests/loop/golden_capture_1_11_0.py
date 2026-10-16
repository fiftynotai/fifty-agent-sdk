"""Golden capture of the 1.11.0 request bodies AND event stream (FR-004, plan D14a).

FR-004 adds ``AgentLoop(skills=...)``: a ``# Skills`` system-prompt section
and a ``load_skill`` tool. Its compatibility promise is that a loop built
WITHOUT ``skills`` (or with an empty ``SkillSet``) sends the same request
bodies (same keys, values and JSON types), the same system prompt and the
same event stream as 1.11.0 (timestamps and minted ids aside). FR-004 edits
the prompt renderer's section order and the loop's tool snapshot (used for
the prompt-side tool block and the NATIVE ``tools`` list) and its two
dispatch sites. The three earlier fixtures (``legacy_1_7_0.json``,
``requests_1_8_0.json``, ``requests_1_9_0.json``) leave three of those paths
uncovered:

* no scenario sets ``PromptSections.additional_context``, the slot after the
  new one (grep found 0 occurrences of ``# Additional Context`` in
  ``tests/loop/golden/``);
* no scenario configures ``interventions``, so the intervention dispatch path
  (an argument replacement, an ``after_tool`` note, a denied batch member)
  rests on unit tests only;
* ``ToolMode.NATIVE`` with an empty registry is pinned for bodies only
  (``requests_1_8_0.json``), not for events.

So each scenario here records ``{"bodies": [...], "events": [...]}``, as the
1.9.0 harness does:

* :data:`SCENARIOS_1_11_0` maps a stable key to a builder. Each builder makes
  a fresh registry and fresh fakes, and none passes ``skills``: the kwarg did
  not exist when the fixture was captured.
* Bodies go through the REAL wire translator
  (:func:`tests.loop.golden_capture.wire_bodies`, which calls
  ``OpenAICompatibleClient._build_body`` and does no I/O), so the system
  prompt is pinned as the first message of every body.
* Events and ids go through
  :func:`tests.loop.golden_capture_1_9_0.serialise_events` and
  :func:`tests.loop.golden_capture_1_9_0._normalise_scenario` (ONE
  first-appearance id mapping over the bodies, then the events).

Because the baseline IS 1.11.0, it needs none of the scope clauses the 1.7.0,
1.8.0 and 1.9.0 harnesses carry for the BR-019 to BR-028 batch. It still
does not pin: any error path (no scenario ends on an ``ErrorEvent``); key
order (sorted on both sides); HTTP bytes; non-ASCII text (every string in
the scenarios is ASCII apart from the constant parser-retry reminders, which
no scenario triggers); log lines; or any skills path.

Running ``python -m tests.loop.golden_capture_1_11_0`` from the repo root
writes ``tests/loop/golden/requests_1_11_0.json``. The fixture in the tree was
written from the UNMODIFIED 1.11.0 source (commit ``4d4fda6``, the
``v1.11.0`` tag) before any FR-004 edit under ``src/``, and cross-checked
against a capture run on the published 1.11.0 PyPI wheel. It must never be
regenerated to make a failing diff pass: a diff means a no-skills request or
event moved. Never run the older writers either
(``python -m tests.loop.golden_capture``, ``..._1_8_0``, ``..._1_9_0``): they
would rewrite the older fixtures from newer source and destroy their
provenance.

This file is deliberately NOT named ``test_*.py`` so pytest does not collect
it; :mod:`tests.loop.test_golden_1_11_0` imports it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    ChatMessage,
    DenyToolCall,
    Interventions,
    JsonModeParser,
    PromptSections,
    Registry,
    ReplaceToolArgs,
    SafetyConfig,
    ToolMode,
    ToolResult,
    ToolSchema,
)
from tests.loop.conftest import FakeLLMClient, make_multi_tool_response, make_response
from tests.loop.golden_capture import render, wire_bodies
from tests.loop.golden_capture_1_9_0 import _normalise_scenario, serialise_events

GOLDEN_1_11_0_PATH = Path(__file__).parent / "golden" / "requests_1_11_0.json"
"""Where the fixture lives. Written once, from 1.11.0, by ``__main__`` below."""

_PERSONA = "You are a careful support assistant."
_ADDITIONAL_CONTEXT = "Ticket context: the customer is on the Pro plan, region eu-west."
_MODEL = "golden-model"
_USER_INPUT = [ChatMessage(role="user", content="What is the status of ticket T-7?")]

LoopScenario = tuple[AgentLoop, FakeLLMClient, list[ChatMessage]]
"""A built loop, the fake it records requests on, and the caller's input messages."""


# ---------------------------------------------------------------------------
# Deterministic tools, registry and scripted replies
# ---------------------------------------------------------------------------


class _LookupTool:
    """A tool with a real schema (properties and ``required``) and a fixed result.

    ``ticket`` is echoed in the result, so an argument replacement by
    ``before_tool`` is visible in the observation.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.schema = ToolSchema(
            properties={"ticket": {"type": "string", "description": "The ticket id."}},
            required=["ticket"],
        )

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output={"ticket": args.get("ticket"), "status": "open"})


def _registry() -> Registry:
    """``lookup_ticket`` and ``lookup_customer``, both succeeding."""
    registry = Registry()
    registry.register(_LookupTool("lookup_ticket", "Look up a support ticket by id."))
    registry.register(_LookupTool("lookup_customer", "Look up the customer of a ticket."))
    return registry


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


def _prompts() -> PromptSections:
    """Persona plus ``additional_context``; the loop fills the tool and format slots."""
    return PromptSections(persona=_PERSONA, additional_context=_ADDITIONAL_CONTEXT)


def _mode_loop(
    llm: FakeLLMClient,
    tool_mode: ToolMode,
    *,
    registry: Registry | None = None,
    **extra: Any,
) -> AgentLoop:
    """A loop with an explicit ``tool_mode``, all prompt slots the caller owns, no skills."""
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _registry(),
        prompts=_prompts(),
        safety=extra.pop("safety", SafetyConfig()),
        model=_MODEL,
        tool_mode=tool_mode,
        **extra,
    )


# ---------------------------------------------------------------------------
# Scenarios: 1.11.0 call shapes only, no skills
# ---------------------------------------------------------------------------


def _all_slots_json() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("lookup_ticket", {"ticket": "T-7"})),
            make_response(_json_final("T-7 is open.")),
        ]
    )
    return _mode_loop(llm, ToolMode.JSON), llm, list(_USER_INPUT)


def _all_slots_prose() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_prose_tool("lookup_ticket", {"ticket": "T-7"})),
            make_response(_prose_final("T-7 is open.")),
        ]
    )
    return _mode_loop(llm, ToolMode.PROSE), llm, list(_USER_INPUT)


def _all_slots_native() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("lookup_ticket", {"ticket": "T-7"})]),
            make_response("T-7 is open."),
        ]
    )
    return _mode_loop(llm, ToolMode.NATIVE), llm, list(_USER_INPUT)


def _legacy_caller_tool_descriptions() -> LoopScenario:
    """The legacy path, where the caller's ``tool_descriptions`` is replaced by the loop's."""
    llm = FakeLLMClient(
        [
            make_response(_json_tool("lookup_ticket", {"ticket": "T-7"})),
            make_response(_json_final("T-7 is open.")),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        registry=_registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(
            persona=_PERSONA,
            tool_descriptions="IGNORED",
            additional_context=_ADDITIONAL_CONTEXT,
        ),
        safety=SafetyConfig(),
        model=_MODEL,
        output_format=JSON_MODE_OUTPUT_FORMAT,
    )
    return loop, llm, list(_USER_INPUT)


def _native_empty_registry() -> LoopScenario:
    llm = FakeLLMClient([make_response("No tools are needed.")])
    return _mode_loop(llm, ToolMode.NATIVE, registry=Registry()), llm, list(_USER_INPUT)


def _interventions_json_replace_and_note() -> LoopScenario:
    def before_tool(
        _session_id: str | None, _call_id: str, _name: str, args: dict[str, Any]
    ) -> ReplaceToolArgs:
        return ReplaceToolArgs(args={**args, "ticket": "T-8"})

    def after_tool(
        _session_id: str | None,
        _call_id: str,
        _name: str,
        _args: dict[str, Any],
        _result: ToolResult,
    ) -> str:
        return "Note: ticket ids are normalised by policy."

    llm = FakeLLMClient(
        [
            make_response(_json_tool("lookup_ticket", {"ticket": "T-7"})),
            make_response(_json_final("T-8 is open.")),
        ]
    )
    loop = _mode_loop(
        llm,
        ToolMode.JSON,
        interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
    )
    return loop, llm, list(_USER_INPUT)


def _interventions_native_batch_deny() -> LoopScenario:
    def before_tool(
        _session_id: str | None, _call_id: str, name: str, _args: dict[str, Any]
    ) -> DenyToolCall | None:
        return (
            DenyToolCall(reason="customer data needs approval")
            if name == "lookup_customer"
            else None
        )

    llm = FakeLLMClient(
        [
            make_multi_tool_response(
                [("lookup_ticket", {"ticket": "T-7"}), ("lookup_customer", {"ticket": "T-7"})],
                content="checking both",
            ),
            make_response("T-7 is open; customer lookup was denied."),
        ]
    )
    loop = _mode_loop(
        llm,
        ToolMode.NATIVE,
        safety=SafetyConfig(max_concurrent_tool_calls=2),
        interventions=Interventions(before_tool=before_tool),
    )
    return loop, llm, list(_USER_INPUT)


SCENARIOS_1_11_0: dict[str, Callable[[], LoopScenario]] = {
    "all_slots_json": _all_slots_json,
    "all_slots_prose": _all_slots_prose,
    "all_slots_native": _all_slots_native,
    "legacy_caller_tool_descriptions": _legacy_caller_tool_descriptions,
    "native_empty_registry": _native_empty_registry,
    "interventions_json_replace_and_note": _interventions_json_replace_and_note,
    "interventions_native_batch_deny": _interventions_native_batch_deny,
}
"""Scenario key -> builder. These are the fixture's top-level keys."""


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def capture_scenario_1_11_0(key: str) -> dict[str, list[dict[str, Any]]]:
    """Run scenario ``key`` and return its id-normalised wire bodies and events."""
    loop, llm, messages = SCENARIOS_1_11_0[key]()
    events = [event async for event in loop.run(messages)]
    bodies = await wire_bodies(llm, stream=loop._stream)
    return _normalise_scenario(bodies, serialise_events(events))


async def capture_all_1_11_0() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Capture every scenario, keyed as in :data:`SCENARIOS_1_11_0`."""
    return {key: await capture_scenario_1_11_0(key) for key in sorted(SCENARIOS_1_11_0)}


if __name__ == "__main__":
    GOLDEN_1_11_0_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_1_11_0_PATH.write_text(render(asyncio.run(capture_all_1_11_0())), encoding="utf-8")
    print(f"wrote {GOLDEN_1_11_0_PATH}")
