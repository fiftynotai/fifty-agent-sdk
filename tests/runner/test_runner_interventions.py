"""Runner integration tests for the intervention hooks (FR-003).

``Interventions`` are wired on :class:`fifty_agent_sdk.loop.AgentLoop` only;
the Runner passes its ``session_id`` down and correlates events as before.
What these pin:

* AC-6 (D7): a note lives in the loop's working list for the rest of the
  run and is never persisted: the history after a turn is the user message
  and the raw final answer, and the next turn's first request carries no
  note. Three tool-result roles: legacy ``"tool"``, ``JSON`` ``"assistant"``
  and ``NATIVE``.
* D5: a ``before_tool`` replacement reaches ``on_tool_start`` and the
  ``tool_invocation`` audit payload (the Runner reads ``ActionEvent.args``),
  and a denied batch member keeps the Runner's FIFO per-call correlation,
  which depends on the denied call still emitting ``ToolStartedEvent``.
* G1 under a Runner: ``after_tool`` runs after the consumer handled the
  terminal event and after ``on_tool_end`` fired.
"""

from __future__ import annotations

from typing import Any

import pytest

from fifty_agent_sdk import (
    AgentLoop,
    AgentRunner,
    AuditEvent,
    DenyToolCall,
    FinalEvent,
    Hooks,
    Interventions,
    MemoryStateStore,
    ObservationEvent,
    PromptSections,
    Registry,
    ReplaceToolArgs,
    SafetyConfig,
    StateStore,
    ToolFailedEvent,
    ToolMode,
    ToolResult,
)
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response
from tests.runner.conftest import collect, final_json, make_runner, tool_json

_NOTE = "The user can already see these records in the UI."


class _SpyAuditSink:
    """An :class:`AuditSink` that records every event (file-local)."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def _constant(value: object) -> Any:
    def hook(*_args: object) -> object:
        return value

    return hook


def _registry(*tools: FakeTool) -> Registry:
    registry = Registry()
    for tool in tools:
        registry.register(tool)
    return registry


def _runner_for(
    mode: str, llm: FakeLLMClient, interventions: Interventions
) -> tuple[AgentRunner, StateStore]:
    """A runner whose loop sends tool results in ``mode``'s role."""
    registry = _registry(FakeTool("search", result=ToolResult(output={"rows": [1, 2]})))
    if mode == "legacy_tool":
        return make_runner(llm=llm, registry=registry, interventions=interventions)
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        prompts=PromptSections(persona="You are helpful."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=ToolMode.JSON if mode == "json_assistant" else ToolMode.NATIVE,
        interventions=interventions,
    )
    store = MemoryStateStore()
    return AgentRunner(loop=loop, state=store), store


def _turn_replies(mode: str) -> tuple[list[Any], str]:
    """Turn 1 (a tool call, then a final) plus turn 2 (a final), and turn 1's raw final."""
    if mode == "native":
        first_final = "found two records"
        return [
            make_multi_tool_response([("search", {"q": "open"})]),
            make_response(first_final),
            make_response("second turn answer"),
        ], first_final
    first_final = final_json("found two records")
    return [
        make_response(tool_json("look up", "search", {"q": "open"})),
        make_response(first_final),
        make_response(final_json("second turn answer")),
    ], first_final


@pytest.mark.parametrize("mode", ["legacy_tool", "json_assistant", "native"])
async def test_intervention_text_is_never_persisted(mode: str) -> None:
    """A note reaches the same run's follow-up request once and is never stored; the next turn never sees it (FR-003 AC-6)."""
    replies, first_final = _turn_replies(mode)
    llm = FakeLLMClient(replies)
    runner, store = _runner_for(mode, llm, Interventions(after_tool=_constant(_NOTE)))

    await collect(runner.run("s1", "Show me the open records."))

    history = await store.get_messages("s1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].content == first_final
    assert all(_NOTE not in m.content for m in history)
    follow_up = llm.calls[1].messages
    assert sum(m.content.count(_NOTE) for m in follow_up) == 1

    await collect(runner.run("s1", "And the closed ones?"))

    assert len(llm.calls) == 3
    assert all(_NOTE not in m.content for m in llm.calls[2].messages)
    assert all(_NOTE not in m.content for m in await store.get_messages("s1"))


async def test_replaced_args_reach_on_tool_start_and_the_audit_payload() -> None:
    """``on_tool_start`` and the audit ``args`` summary show the replacement that ran (FR-003 D5)."""
    tool = FakeTool("search", result=ToolResult(output="ok"))
    llm = FakeLLMClient(
        [
            make_response(tool_json("look up", "search", {"q": "everything"})),
            make_response(final_json("done")),
        ]
    )
    starts: list[tuple[Any, ...]] = []
    spy = _SpyAuditSink()
    replacement = {"q": "scoped", "tenant": "t-1"}
    runner, _store = make_runner(
        llm=llm,
        registry=_registry(tool),
        audit=spy,
        hooks=Hooks(on_tool_start=lambda *args: starts.append(args)),
        interventions=Interventions(before_tool=_constant(ReplaceToolArgs(args=replacement))),
    )

    await collect(runner.run("s1", "Hi"))

    assert tool.last_args == replacement
    assert starts == [("s1", "search", replacement)]
    invocations = [e for e in spy.events if e.event_type == "tool_invocation"]
    assert len(invocations) == 1
    assert invocations[0].payload["args"] == {
        "q": {"type": "str", "len": 6},
        "tenant": {"type": "str", "len": 3},
    }
    assert invocations[0].payload["outcome"] == "ok"


async def test_denied_batch_member_keeps_runner_correlation_per_call() -> None:
    """A denied batch member still pairs its own args, denial text and audit outcome per call (FR-003 D5)."""
    alpha = FakeTool("alpha", result=ToolResult(output="A-result"))
    beta = FakeTool("beta", result=ToolResult(output="B-result"))
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("alpha", {"n": 1}), ("beta", {"n": 2, "m": 3})]),
            make_response(final_json("done")),
        ]
    )
    starts: list[tuple[Any, ...]] = []
    ends: list[tuple[Any, ...]] = []
    spy = _SpyAuditSink()

    def before_tool(_sid: object, _cid: object, name: str, _args: object) -> DenyToolCall | None:
        return DenyToolCall(reason="alpha is paused") if name == "alpha" else None

    runner, _store = make_runner(
        llm=llm,
        registry=_registry(alpha, beta),
        safety=SafetyConfig(native_tools_enabled=True, max_concurrent_tool_calls=2),
        audit=spy,
        hooks=Hooks(
            on_tool_start=lambda *args: starts.append(args),
            on_tool_end=lambda *args: ends.append(args),
        ),
        interventions=Interventions(before_tool=before_tool),
    )

    events = await collect(runner.run("s1", "Hi"))

    assert alpha.call_count == 0
    assert beta.last_args == {"n": 2, "m": 3}
    assert starts == [("s1", "alpha", {"n": 1}), ("s1", "beta", {"n": 2, "m": 3})]
    assert [(e[1], e[2]) for e in ends] == [
        ("alpha", "Tool call denied: alpha is paused"),
        ("beta", "B-result"),
    ]
    invocations = [e.payload for e in spy.events if e.event_type == "tool_invocation"]
    assert [(p["tool_name"], p["outcome"]) for p in invocations] == [
        ("alpha", "failed"),
        ("beta", "ok"),
    ]
    assert [p["args"] for p in invocations] == [
        {"n": {"type": "int", "len": None}},
        {"m": {"type": "int", "len": None}, "n": {"type": "int", "len": None}},
    ]
    assert isinstance(events[-1], FinalEvent)


async def test_after_tool_runs_after_on_tool_end_and_after_the_consumer_saw_the_event() -> None:
    """Under a Runner, each call's order is: consumer gets the terminal event, ``on_tool_end``, then ``after_tool`` (FR-003 G1)."""
    llm = FakeLLMClient(
        [
            make_response(tool_json("first", "search", {"q": "a"})),
            make_response(tool_json("second", "search", {"q": "b"})),
            make_response(final_json("done")),
        ]
    )
    order: list[tuple[str, str]] = []

    def after_tool(_sid: object, call_id: str, *_rest: object) -> None:
        order.append(("after_tool", call_id))

    def on_tool_end(_sid: object, _name: object, _result: object, _ms: object) -> None:
        order.append(("on_tool_end", ""))

    runner, _store = make_runner(
        llm=llm,
        registry=_registry(FakeTool("search", result=ToolResult(output="rows"))),
        hooks=Hooks(on_tool_end=on_tool_end),
        interventions=Interventions(after_tool=after_tool),
    )

    call_ids: list[str] = []
    async for event in runner.run("s1", "Hi"):
        if isinstance(event, ObservationEvent | ToolFailedEvent):
            call_ids.append(event.call_id)
            order.append(("consumer", event.call_id))

    assert len(call_ids) == 2
    expected: list[tuple[str, str]] = []
    for call_id in call_ids:
        expected += [("consumer", call_id), ("on_tool_end", ""), ("after_tool", call_id)]
    assert order == expected
