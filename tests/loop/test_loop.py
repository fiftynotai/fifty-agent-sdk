"""Integration tests for ``agent_sdk.loop.AgentLoop``.

Each test wires the loop with :class:`tests.loop.conftest.FakeLLMClient` and
:class:`tests.loop.conftest.FakeTool` doubles, drives a single ``run()``,
collects the event stream, and asserts on shape, sequence, and side
effects.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest

from agent_sdk import (
    ActionEvent,
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    Hooks,
    JsonModeParser,
    ObservationEvent,
    PromptSections,
    Registry,
    SafetyConfig,
    ThoughtEvent,
    TokenEvent,
    ToolFailedEvent,
    ToolResult,
    ToolStartedEvent,
)
from agent_sdk.errors import LLMError
from agent_sdk.streaming import ToolProgressEvent
from tests.loop.conftest import (
    DriftsOnceFakeLLM,
    FakeLLMClient,
    FakeTool,
    make_response,
    make_stream_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_json(answer: str) -> str:
    return json.dumps(
        {
            "thought": "done",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def _tool_json(thought: str, name: str, args: dict[str, Any] | None) -> str:
    return json.dumps(
        {
            "thought": thought,
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def _make_loop(
    *,
    llm: FakeLLMClient,
    registry: Registry | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    persona: str = "You are a helpful agent.",
    tool_message_role: Literal["tool", "user", "assistant"] = "tool",
) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=persona),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
        stream=stream,
        tool_message_role=tool_message_role,
    )


async def _collect(
    iterator: AsyncIterator[AgentEvent],
) -> list[AgentEvent]:
    return [event async for event in iterator]


# ---------------------------------------------------------------------------
# Happy single-step
# ---------------------------------------------------------------------------


async def test_happy_single_step_emits_thought_then_final() -> None:
    llm = FakeLLMClient(replies=[make_response(_final_json("42"))])
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    types = [type(e) for e in events]
    assert types == [ThoughtEvent, FinalEvent]
    assert [e.sequence for e in events] == [0, 1]
    final_event = events[1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "42"
    assert len(llm.calls) == 1


async def test_happy_single_step_thought_text_propagated() -> None:
    completion = json.dumps(
        {
            "thought": "deep reasoning",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": "ans",
        }
    )
    llm = FakeLLMClient(replies=[make_response(completion)])
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    thought_event = events[0]
    assert isinstance(thought_event, ThoughtEvent)
    assert thought_event.text == "deep reasoning"


# ---------------------------------------------------------------------------
# Happy multi-step
# ---------------------------------------------------------------------------


async def test_happy_multi_step_with_tool_then_final() -> None:
    tool = FakeTool("search", result=ToolResult(output={"results": ["a", "b"]}))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("look it up", "search", {"q": "x"})),
            make_response(_final_json("answer is a,b")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    types = [type(e) for e in events]
    assert types == [
        ThoughtEvent,
        ActionEvent,
        ToolStartedEvent,
        ObservationEvent,
        ThoughtEvent,
        FinalEvent,
    ]
    assert [e.sequence for e in events] == [0, 1, 2, 3, 4, 5]

    started_event = events[2]
    observation_event = events[3]
    assert isinstance(started_event, ToolStartedEvent)
    assert isinstance(observation_event, ObservationEvent)
    assert started_event.call_id == observation_event.call_id
    # uuid4().hex is a 32-char lowercase hex string
    assert len(started_event.call_id) == 32
    assert all(c in "0123456789abcdef" for c in started_event.call_id)

    # second LLM call sees the tool reply with role="tool"
    assert len(llm.calls) == 2
    second_request = llm.calls[1]
    tool_msgs = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "search"
    assert tool_msgs[0].tool_call_id == started_event.call_id


async def test_happy_multi_step_action_event_carries_args() -> None:
    tool = FakeTool("search", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "search", {"k": "v"})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    action_event = events[1]
    assert isinstance(action_event, ActionEvent)
    assert action_event.tool_name == "search"
    assert action_event.args == {"k": "v"}
    assert tool.last_args == {"k": "v"}


# ---------------------------------------------------------------------------
# Tool failure: ToolResult(is_error=True)
# ---------------------------------------------------------------------------


async def test_tool_returns_is_error_emits_tool_failed_and_continues() -> None:
    tool = FakeTool(
        "search",
        result=ToolResult(output=None, is_error=True, error="boom"),
    )
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "search", {})),
            make_response(_final_json("recovered")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    types = [type(e) for e in events]
    assert types == [
        ThoughtEvent,
        ActionEvent,
        ToolStartedEvent,
        ToolFailedEvent,
        ThoughtEvent,
        FinalEvent,
    ]
    failed_event = events[3]
    assert isinstance(failed_event, ToolFailedEvent)
    assert failed_event.error == "boom"

    # The second LLM call should see a tool message starting with "Tool error:"
    second_request = llm.calls[1]
    tool_msgs = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content.startswith("Tool error:")


async def test_tool_returns_is_error_without_error_message() -> None:
    """ToolResult(is_error=True, error=None) gets a default error description."""
    tool = FakeTool("t", result=ToolResult(output=None, is_error=True, error=None))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("th", "t", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    failed_event = next(e for e in events if isinstance(e, ToolFailedEvent))
    assert failed_event.error == "tool reported error with no message"


# ---------------------------------------------------------------------------
# Tool not found
# ---------------------------------------------------------------------------


async def test_tool_not_found_emits_tool_failed_and_continues() -> None:
    registry = Registry()  # no tools registered
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "ghost", {})),
            make_response(_final_json("gave up")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    failed_event = next(e for e in events if isinstance(e, ToolFailedEvent))
    assert failed_event.error.startswith("ToolNotFound:")

    second_request = llm.calls[1]
    tool_msgs = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "ToolNotFound" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# Tool timeout
# ---------------------------------------------------------------------------


async def test_tool_timeout_emits_tool_failed_and_continues() -> None:
    tool = FakeTool("slow", sleep_seconds=0.5)
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=4, tool_timeout_seconds=0.05)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "slow", {})),
            make_response(_final_json("recovered")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    failed_event = next(e for e in events if isinstance(e, ToolFailedEvent))
    assert failed_event.error.startswith("ToolTimeout:")
    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "recovered"


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------


async def test_iteration_cap_emits_error_and_fallback_final() -> None:
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=2)
    # LLM emits tool calls forever; loop will hit the cap after 2 iterations.
    tool_call = make_response(_tool_json("loop", "t", {}))
    llm = FakeLLMClient(replies=[tool_call, tool_call, tool_call])
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = events[-2]
    final_event = events[-1]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.error_type == "MaxIterationsExceeded"
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == safety.fallback_message
    assert len(llm.calls) == 2


async def test_max_iterations_exceeded_context_includes_counters() -> None:
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=2)
    tool_call = make_response(_tool_json("t", "t", {}))
    llm = FakeLLMClient(replies=[tool_call, tool_call])
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = next(e for e in events if isinstance(e, ErrorEvent))
    assert error_event.context == {"max_iterations": 2, "iteration_count": 2}


async def test_iteration_cap_does_not_raise() -> None:
    """Iteration-cap termination must NOT propagate an exception out of the generator."""
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=1)
    tool_call = make_response(_tool_json("t", "t", {}))
    llm = FakeLLMClient(replies=[tool_call])
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    # If this raised, pytest would fail the test outright.
    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    assert isinstance(events[-1], FinalEvent)


# ---------------------------------------------------------------------------
# Parser error
# ---------------------------------------------------------------------------


async def test_parser_error_emits_error_event_and_fallback_final() -> None:
    # BR-018: the loop performs a one-shot parser-error retry by default, so
    # we need TWO prose replies to drive the run through to the terminal
    # ParserError path (drift → retry → drift again → exhausted).
    llm = FakeLLMClient(
        replies=[
            make_response("not valid json at all"),
            make_response("still not valid json"),
        ]
    )
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = events[-2]
    final_event = events[-1]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.error_type == "ParserError"
    assert error_event.context["parser"] == "JsonModeParser"
    assert "error_phase" in error_event.context
    assert isinstance(final_event, FinalEvent)


async def test_parser_error_does_not_propagate() -> None:
    # BR-018: see above — two prose replies to exhaust the retry budget.
    llm = FakeLLMClient(replies=[make_response("garbage"), make_response("more garbage")])
    loop = _make_loop(llm=llm)
    # Must not raise:
    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    assert isinstance(events[-1], FinalEvent)


# ---------------------------------------------------------------------------
# Parser-error retry (BR-018)
# ---------------------------------------------------------------------------


async def test_parser_error_one_shot_retry_recovers() -> None:
    """The default loop self-heals a single parse drift via the BR-018 retry.

    Models the exact Gemini meta-query failure mode: the model emits a
    Markdown list outside the envelope on call 1, then on the
    format-reminder retry returns a clean envelope on call 2. The loop
    must NOT surface an :class:`ErrorEvent` and must reach the parsed
    final answer.
    """
    llm = DriftsOnceFakeLLM(
        prose_reply="# Here are my tools:\n- echo",
        json_reply=_final_json("ok"),
    )
    loop = AgentLoop(
        llm=llm,
        registry=Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona="P"),
        safety=SafetyConfig(),
        model="test-model",
    )

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # No ErrorEvent in the stream — the retry absorbs the drift.
    assert not any(isinstance(e, ErrorEvent) for e in events)
    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "ok"

    # Two LLM calls: the drift attempt plus the post-reminder success.
    assert len(llm.calls) == 2

    # The reminder text was injected into the second call's messages.
    second_request = llm.calls[1]
    reminder = SafetyConfig().parser_retry_reminder
    assert any(
        message.role == "user" and message.content == reminder
        for message in second_request.messages
    ), "BR-018 reminder text must appear as a user-role message on the retry"


async def test_parser_error_retry_exhausted_terminates() -> None:
    """Two prose replies in a row exhaust the one-shot retry budget.

    Locks the terminal ``ErrorEvent`` + fallback ``FinalEvent`` shape on
    the retry-exhausted path and asserts the LLM was called exactly twice
    — once for the drift, once for the retry that also drifted.
    """
    llm = FakeLLMClient(replies=[make_response("drift one"), make_response("drift two")])
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = events[-2]
    final_event = events[-1]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.error_type == "ParserError"
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == SafetyConfig().fallback_message
    # The retry attempt fires a second LLM call before exhausting the budget.
    assert len(llm.calls) == 2


async def test_parser_error_retry_disabled_via_safety() -> None:
    """``SafetyConfig(parser_retry_enabled=False)`` is the kill-switch.

    A single prose reply terminates the run on the first ParserError, the
    second reply is never consumed, and the LLM call count is exactly one.
    """
    llm = FakeLLMClient(replies=[make_response("drift only")])
    safety = SafetyConfig(parser_retry_enabled=False)
    loop = _make_loop(llm=llm, safety=safety)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = events[-2]
    final_event = events[-1]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.error_type == "ParserError"
    assert isinstance(final_event, FinalEvent)
    # NO retry was attempted — the kill-switch held.
    assert len(llm.calls) == 1


async def test_parser_retry_does_not_increment_iteration_hook() -> None:
    """The BR-018 retry is sub-iteration: ``on_iteration`` fires once, not twice.

    Pins the hook-firing contract: a single outer ReACT iteration that
    happens to retry once internally fires ``on_iteration`` exactly ONCE
    (per outer iteration) and ``on_llm_call`` exactly TWICE (per inner
    LLM call). If the retry ever leaks into the outer counter this test
    catches the regression.
    """
    iteration_calls: list[int] = []
    llm_call_records: list[tuple[ChatRequest, ChatResponse, float]] = []

    def on_iteration(session_id: str | None, iteration_n: int) -> None:
        iteration_calls.append(iteration_n)

    def on_llm_call(
        session_id: str | None,
        request: ChatRequest,
        response: ChatResponse,
        duration_ms: float,
    ) -> None:
        llm_call_records.append((request, response, duration_ms))

    hooks = Hooks(on_iteration=on_iteration, on_llm_call=on_llm_call)
    llm = DriftsOnceFakeLLM(
        prose_reply="# drift",
        json_reply=_final_json("done"),
    )
    loop = AgentLoop(
        llm=llm,
        registry=Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona="P"),
        safety=SafetyConfig(),
        model="test-model",
        hooks=hooks,
    )

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # Exactly one outer iteration despite the sub-iteration retry.
    assert iteration_calls == [1]
    # Two LLM calls: the drift + the post-reminder success.
    assert len(llm_call_records) == 2


# ---------------------------------------------------------------------------
# Require-tool-before-final force-reconsider (BR-036)
#
# POLARITY NOTE: this guard fires on a NEGATIVE condition — "NO tool ran this
# run". Every scenario below pairs a FIRE case (guard nudges) with a NO-FIRE
# case (guard stays out of the way), so a flipped boolean is caught.
# ---------------------------------------------------------------------------


def _require_tool_safety() -> SafetyConfig:
    """A SafetyConfig with the BR-036 force-reconsider opted in (custom reminder)."""
    return SafetyConfig(
        require_tool_before_final=True,
        tool_required_reminder="REMINDER: call a tool first or re-answer.",
    )


async def test_require_tool_forces_reconsider_then_search() -> None:
    """FIRE then recover: a no-tool final is nudged, the model searches, then grounds.

    Call 1 = a no-tool ``final`` (the model tried to answer a policy
    question without searching). The guard MUST NOT emit that final;
    instead it injects the reminder and re-prompts. Call 2 = a
    ``policy_search`` tool call (satisfying "a tool ran"). Call 3 = a
    grounded ``final`` that now passes unguarded.
    """
    tool = FakeTool("policy_search", result=ToolResult(output="passage about leave"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_final_json("Annual leave is 30 days.")),  # no-tool final → nudged
            make_response(_tool_json("search", "policy_search", {"q": "leave"})),  # forced search
            make_response(_final_json("Annual leave is 30 days [leave-policy].")),  # grounded
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=_require_tool_safety())

    events = await _collect(loop.run([ChatMessage(role="user", content="leave days?")]))

    # An ActionEvent for policy_search appears (the forced search ran).
    action_events = [e for e in events if isinstance(e, ActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].tool_name == "policy_search"

    # Exactly ONE terminal FinalEvent, and it is the grounded answer — the
    # call-1 no-tool final was suppressed (not emitted).
    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert final_events[0].text == "Annual leave is 30 days [leave-policy]."

    # Three LLM calls: nudged drift + forced search + grounded final.
    assert len(llm.calls) == 3

    # The reminder text landed as a user-role message on the SECOND call.
    reminder = _require_tool_safety().tool_required_reminder
    assert any(
        message.role == "user" and message.content == reminder for message in llm.calls[1].messages
    ), "BR-036 reminder must appear as a user-role message after the no-tool final"


async def test_require_tool_reanchors_original_question_after_reminder() -> None:
    """The forced reconsideration re-anchors the ORIGINAL question last.

    Locks the UX fix for the meta-acknowledgment failure mode: after the
    reminder is injected, the loop re-appends the current turn's original user
    message so the model's LATEST message is the question (not the reminder).
    Asserts, on the second LLM call's messages:

    * the reminder is present as a user turn, AND
    * the original question is re-appended as a user turn AFTER it, AND
    * the very last message is the original question (so the model answers
      THAT, not the reminder).
    """
    original = "What is the leave policy?"
    tool = FakeTool("policy_search", result=ToolResult(output="passage about leave"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_final_json("Annual leave is 30 days.")),  # no-tool final → nudged
            make_response(_tool_json("search", "policy_search", {"q": "leave"})),  # forced search
            make_response(_final_json("Annual leave is 30 days [leave-policy].")),  # grounded
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=_require_tool_safety())

    await _collect(loop.run([ChatMessage(role="user", content=original)]))

    second_call_messages = llm.calls[1].messages
    reminder = _require_tool_safety().tool_required_reminder
    reminder_indices = [
        i for i, m in enumerate(second_call_messages) if m.role == "user" and m.content == reminder
    ]
    reanchor_indices = [
        i for i, m in enumerate(second_call_messages) if m.role == "user" and m.content == original
    ]
    assert reminder_indices, "reminder must be a user turn on the forced reconsideration"
    # The original question appears twice now: the genuine first turn AND the
    # re-anchored copy after the reminder. The re-anchored copy is the one that
    # follows the reminder.
    assert any(ri > reminder_indices[-1] for ri in reanchor_indices), (
        "original question must be re-appended AFTER the reminder"
    )
    # The model's latest message is the original question, not the reminder.
    last = second_call_messages[-1]
    assert last.role == "user"
    assert last.content == original


async def test_require_tool_greeting_passes_after_one_nudge() -> None:
    """NO-FIRE-ish: a greeting is nudged once, re-confirmed, and allowed through.

    The reminder is PERMISSIVE, so a genuine greeting is NOT coerced into a
    search. Call 1 = a no-tool greeting ``final``; call 2 = the SAME no-tool
    ``final`` (model re-confirms it is just a greeting). The loop accepts the
    second final — exactly one nudge, no tool ever runs.
    """
    greeting = "Hi! I'm the MOCA policy assistant."
    llm = FakeLLMClient(
        replies=[
            make_response(_final_json(greeting)),
            make_response(_final_json(greeting)),
        ]
    )
    loop = _make_loop(llm=llm, safety=_require_tool_safety())

    events = await _collect(loop.run([ChatMessage(role="user", content="hi")]))

    # The greeting passes through — never forced to search.
    assert not any(isinstance(e, ActionEvent) for e in events)
    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert final_events[0].text == greeting
    # Exactly one nudge: two LLM calls, no more.
    assert len(llm.calls) == 2


async def test_require_tool_bounded_one_shot() -> None:
    """BOUNDED: the force is strictly one-shot — it never loops forever.

    Call 1 and call 2 are BOTH no-tool ``final``s. The loop forces exactly
    ONCE, then ACCEPTS the second no-tool final at the loop layer (the hard
    decline is the app backstop's job, asserted separately). It must not
    nudge again and must not spin.
    """
    llm = FakeLLMClient(
        replies=[
            make_response(_final_json("first ungrounded answer")),
            make_response(_final_json("second ungrounded answer")),
        ]
    )
    loop = _make_loop(llm=llm, safety=_require_tool_safety())

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # Exactly two calls — one forced reconsideration, then acceptance.
    assert len(llm.calls) == 2
    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    # The second (accepted) final is emitted; the loop did not spin.
    assert final_events[0].text == "second ungrounded answer"
    assert not any(isinstance(e, ActionEvent) for e in events)


async def test_require_tool_off_by_default_unchanged() -> None:
    """NO-FIRE: default SafetyConfig (flag OFF) accepts a no-tool final immediately.

    Locks backward-compat for every other agent/test: with
    ``require_tool_before_final=False`` (the default), a no-tool final on
    call 1 is accepted with NO nudge and exactly ONE LLM call.
    """
    llm = FakeLLMClient(replies=[make_response(_final_json("immediate answer"))])
    loop = _make_loop(llm=llm)  # default SafetyConfig → flag OFF

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    assert len(llm.calls) == 1
    types = [type(e) for e in events]
    assert types == [ThoughtEvent, FinalEvent]
    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "immediate answer"


async def test_require_tool_does_not_fire_when_tool_already_ran() -> None:
    """NO-FIRE: a tool ran this run, so a later final is accepted unguarded.

    Call 1 = a ``policy_search`` tool call (sets "a tool ran"); call 2 = a
    grounded ``final``. The guard targets ONLY no-tool finals, so no nudge
    happens and the final is accepted normally.
    """
    tool = FakeTool("policy_search", result=ToolResult(output="passage"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("search", "policy_search", {"q": "x"})),
            make_response(_final_json("Grounded answer [policy].")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=_require_tool_safety())

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # Exactly two calls — no forced reconsideration was inserted.
    assert len(llm.calls) == 2
    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert final_events[0].text == "Grounded answer [policy]."
    # The reminder text never appears (guard did not fire).
    reminder = _require_tool_safety().tool_required_reminder
    assert not any(
        message.role == "user" and message.content == reminder
        for call in llm.calls
        for message in call.messages
    )


async def test_require_tool_resets_per_run_across_reused_loop() -> None:
    """INVARIANT: ``tool_invoked_this_run`` resets each ``run()`` on a reused loop.

    Drives the SAME ``AgentLoop`` instance for TWO turns:

    * Turn 1 = a ``policy_search`` tool call then a grounded ``final`` —
      accepted (a tool ran this run), NO nudge.
    * Turn 2 = a no-tool ``final`` — it MUST STILL be nudged, proving turn 2
      did not inherit turn 1's "a tool ran" state.

    Locks the core per-run scoping against a future refactor that lifts the
    flag to instance state (which would wrongly exempt turn 2). The scripted
    ``FakeLLMClient`` consumes its replies sequentially ACROSS both ``run()``
    calls, so the reply order is: [t1 tool, t1 final, t2 nudged final, t2
    re-confirmed final].
    """
    tool = FakeTool("policy_search", result=ToolResult(output="passage"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            # Turn 1: search → grounded final (accepted, no nudge).
            make_response(_tool_json("search", "policy_search", {"q": "leave"})),
            make_response(_final_json("Turn 1 grounded [leave-policy].")),
            # Turn 2: no-tool final → nudged → re-confirmed → accepted.
            make_response(_final_json("Turn 2 ungrounded answer")),
            make_response(_final_json("Turn 2 ungrounded answer")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=_require_tool_safety())
    reminder = _require_tool_safety().tool_required_reminder

    # ----- Turn 1: a tool ran, so the final is accepted WITHOUT a nudge. -----
    turn1 = await _collect(loop.run([ChatMessage(role="user", content="leave days?")]))
    assert len(llm.calls) == 2  # tool call + grounded final, no force
    t1_finals = [e for e in turn1 if isinstance(e, FinalEvent)]
    assert len(t1_finals) == 1
    assert t1_finals[0].text == "Turn 1 grounded [leave-policy]."
    # No reminder was injected on turn 1 (a tool ran).
    assert not any(
        m.role == "user" and m.content == reminder for call in llm.calls for m in call.messages
    )

    # ----- Turn 2: SAME loop, no tool — the guard MUST still fire. -----
    turn2 = await _collect(loop.run([ChatMessage(role="user", content="hi")]))
    # Two MORE calls on turn 2 (nudged final + re-confirmed), total 4.
    assert len(llm.calls) == 4
    t2_finals = [e for e in turn2 if isinstance(e, FinalEvent)]
    assert len(t2_finals) == 1
    assert t2_finals[0].text == "Turn 2 ungrounded answer"
    assert not any(isinstance(e, ActionEvent) for e in turn2)
    # The reminder WAS injected on turn 2 — turn 2 did NOT inherit turn 1's
    # "a tool ran" state. This is the invariant under test.
    assert any(m.role == "user" and m.content == reminder for m in llm.calls[3].messages), (
        "turn 2 must be nudged: tool_invoked_this_run must reset per run()"
    )


async def test_require_tool_coexists_with_parser_retry() -> None:
    """REGRESSION: the BR-036 force and the BR-018 parser-retry fire independently.

    Guard ON. Call 1 = prose drift (triggers the per-iteration parser
    retry); call 2 = a no-tool ``final`` (triggers the per-run force);
    call 3 = a ``policy_search`` tool call; call 4 = a grounded ``final``.
    Both mechanisms fire once each and the run reaches the grounded answer.
    """
    tool = FakeTool("policy_search", result=ToolResult(output="passage"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response("prose drift, not json"),  # parser retry
            make_response(_final_json("ungrounded answer")),  # force-reconsider
            make_response(_tool_json("search", "policy_search", {"q": "x"})),  # forced search
            make_response(_final_json("Grounded answer [policy].")),  # grounded final
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, safety=_require_tool_safety())

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # No ErrorEvent — the parser retry absorbed the drift.
    assert not any(isinstance(e, ErrorEvent) for e in events)
    # All four calls consumed: drift + ungrounded + search + grounded.
    assert len(llm.calls) == 4
    action_events = [e for e in events if isinstance(e, ActionEvent)]
    assert len(action_events) == 1
    assert action_events[0].tool_name == "policy_search"
    final_events = [e for e in events if isinstance(e, FinalEvent)]
    assert len(final_events) == 1
    assert final_events[0].text == "Grounded answer [policy]."


# ---------------------------------------------------------------------------
# LLM error
# ---------------------------------------------------------------------------


async def test_llm_error_emits_error_event_and_fallback_final() -> None:
    llm = FakeLLMClient(replies=[LLMError("provider down", context={"model": "test-model"})])
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    error_event = events[-2]
    final_event = events[-1]
    assert isinstance(error_event, ErrorEvent)
    assert error_event.error_type == "LLMError"
    assert error_event.message == "provider down"
    assert error_event.context == {"model": "test-model"}
    assert isinstance(final_event, FinalEvent)


# ---------------------------------------------------------------------------
# Stream mode
# ---------------------------------------------------------------------------


async def test_stream_mode_emits_token_events_on_final_answer() -> None:
    final_completion = _final_json("42")
    # Split into 3 non-empty chunks.
    chunk_size = max(1, len(final_completion) // 3)
    parts = [
        final_completion[i : i + chunk_size] for i in range(0, len(final_completion), chunk_size)
    ]
    llm = FakeLLMClient(replies=[make_stream_chunks(parts)])
    loop = _make_loop(llm=llm, stream=True)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert len(token_events) == len(parts)
    # Token text concatenation should equal the original completion text.
    assert "".join(t.text for t in token_events) == final_completion
    # Final event payload is the parsed answer field, not the raw JSON.
    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "42"

    # Order: Thought before Tokens before Final.
    thought_idx = next(i for i, e in enumerate(events) if isinstance(e, ThoughtEvent))
    token_indices = [i for i, e in enumerate(events) if isinstance(e, TokenEvent)]
    final_idx = next(i for i, e in enumerate(events) if isinstance(e, FinalEvent))
    assert thought_idx < token_indices[0]
    assert token_indices[-1] < final_idx


async def test_stream_mode_does_not_emit_token_events_on_tool_call() -> None:
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    tool_completion = _tool_json("th", "t", {})
    final_completion = _final_json("done")
    llm = FakeLLMClient(
        replies=[
            make_stream_chunks([tool_completion]),
            make_stream_chunks([final_completion]),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, stream=True)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # Iteration 1: tool call — no TokenEvents in that iteration.
    # Iteration 2: final answer — TokenEvents allowed.
    tool_started_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolStartedEvent))
    iter1_events = events[: tool_started_idx + 1]
    assert not any(isinstance(e, TokenEvent) for e in iter1_events)


async def test_stream_mode_skips_empty_deltas() -> None:
    """Empty content chunks must NOT produce zero-length TokenEvents."""
    final_completion = _final_json("ok")
    chunks_with_empty = [
        final_completion[: len(final_completion) // 2],
        "",
        final_completion[len(final_completion) // 2 :],
    ]
    llm = FakeLLMClient(replies=[make_stream_chunks(chunks_with_empty)])
    loop = _make_loop(llm=llm, stream=True)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))

    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert all(e.text for e in token_events)
    # We expect 2 (the empty one was filtered).
    assert len(token_events) == 2


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def test_cancellation_propagates() -> None:
    """Cancelling the consuming task must surface as CancelledError."""
    tool = FakeTool("slow", sleep_seconds=1.0)
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "slow", {})),
            make_response(_final_json("done")),
        ]
    )
    safety = SafetyConfig(max_iterations=3, tool_timeout_seconds=None)
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    async def consume() -> list[AgentEvent]:
        return await _collect(loop.run([ChatMessage(role="user", content="q")]))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancellation_does_not_call_more_tools() -> None:
    """After cancellation, the tool is called at most once and the loop unwinds."""
    tool = FakeTool("slow", sleep_seconds=0.5)
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "slow", {})),
            make_response(_final_json("done")),
        ]
    )
    safety = SafetyConfig(max_iterations=3, tool_timeout_seconds=None)
    loop = _make_loop(llm=llm, registry=registry, safety=safety)

    async def consume() -> list[AgentEvent]:
        return await _collect(loop.run([ChatMessage(role="user", content="q")]))

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.call_count == 1


# ---------------------------------------------------------------------------
# Sequencing invariants
# ---------------------------------------------------------------------------


async def test_event_sequences_are_monotonic_and_dense() -> None:
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "t", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    assert [e.sequence for e in events] == list(range(len(events)))


@pytest.mark.parametrize(
    "scenario",
    ["happy", "parser_error", "iteration_cap", "llm_error"],
)
async def test_run_always_terminates_with_final_event(scenario: str) -> None:
    if scenario == "happy":
        llm = FakeLLMClient(replies=[make_response(_final_json("ok"))])
        loop = _make_loop(llm=llm)
    elif scenario == "parser_error":
        # BR-018: two prose replies needed to exhaust the one-shot retry.
        llm = FakeLLMClient(replies=[make_response("garbage"), make_response("garbage")])
        loop = _make_loop(llm=llm)
    elif scenario == "iteration_cap":
        tool = FakeTool("t", result=ToolResult(output="ok"))
        registry = Registry()
        registry.register(tool)
        tool_call = make_response(_tool_json("t", "t", {}))
        llm = FakeLLMClient(replies=[tool_call, tool_call])
        loop = _make_loop(llm=llm, registry=registry, safety=SafetyConfig(max_iterations=2))
    else:  # llm_error
        llm = FakeLLMClient(replies=[LLMError("fail")])
        loop = _make_loop(llm=llm)

    events = await _collect(loop.run([ChatMessage(role="user", content="q")]))
    assert isinstance(events[-1], FinalEvent)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


async def test_system_prompt_contains_tool_descriptions() -> None:
    registry = Registry()
    registry.register(FakeTool("alpha"))
    registry.register(FakeTool("beta"))
    llm = FakeLLMClient(replies=[make_response(_final_json("ok"))])
    loop = _make_loop(llm=llm, registry=registry)

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    first_request = llm.calls[0]
    system_msg = first_request.messages[0]
    assert system_msg.role == "system"
    assert "alpha" in system_msg.content
    assert "beta" in system_msg.content
    assert "# Tools" in system_msg.content


async def test_empty_registry_omits_tool_section() -> None:
    llm = FakeLLMClient(replies=[make_response(_final_json("ok"))])
    loop = _make_loop(llm=llm, registry=Registry())

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    system_msg = llm.calls[0].messages[0]
    assert "# Tools" not in system_msg.content


async def test_registry_snapshot_does_not_change_per_iteration() -> None:
    """Tools registered AFTER AgentLoop construction must NOT appear in the prompt."""
    registry = Registry()
    registry.register(FakeTool("alpha", result=ToolResult(output="ok")))
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "alpha", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)
    # Register a new tool AFTER constructing the loop:
    registry.register(FakeTool("beta"))

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    # Both LLM calls must share the same system prompt (no beta).
    assert len(llm.calls) == 2
    sys_msg_1 = llm.calls[0].messages[0].content
    sys_msg_2 = llm.calls[1].messages[0].content
    assert "alpha" in sys_msg_1
    assert "beta" not in sys_msg_1
    assert sys_msg_1 == sys_msg_2


async def test_output_format_argument_appears_in_system_prompt() -> None:
    llm = FakeLLMClient(replies=[make_response(_final_json("ok"))])
    loop = AgentLoop(
        llm=llm,
        registry=Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona="P"),
        safety=SafetyConfig(),
        model="test-model",
        output_format="USE JSON",
    )

    await _collect(loop.run([ChatMessage(role="user", content="q")]))
    system_msg = llm.calls[0].messages[0].content
    assert "USE JSON" in system_msg


# ---------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------


async def test_run_does_not_mutate_input_messages() -> None:
    tool = FakeTool("t", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "t", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)
    user_messages: list[ChatMessage] = [ChatMessage(role="user", content="hello")]
    snapshot_id = id(user_messages)
    snapshot_contents = list(user_messages)

    await _collect(loop.run(user_messages))

    # Input list identity is unchanged AND its contents are unmodified.
    assert id(user_messages) == snapshot_id
    assert user_messages == snapshot_contents


# ---------------------------------------------------------------------------
# Empty messages input
# ---------------------------------------------------------------------------


async def test_run_with_empty_messages_still_works() -> None:
    llm = FakeLLMClient(replies=[make_response(_final_json("hello"))])
    loop = _make_loop(llm=llm)

    events = await _collect(loop.run([]))

    final_event = events[-1]
    assert isinstance(final_event, FinalEvent)
    assert final_event.text == "hello"
    # First request has just the system message.
    assert len(llm.calls[0].messages) == 1
    assert llm.calls[0].messages[0].role == "system"


# ---------------------------------------------------------------------------
# _serialize_tool_output repr fallback
# ---------------------------------------------------------------------------


def test_serialize_tool_output_falls_back_to_repr_when_dumps_raises() -> None:
    """Both ``TypeError`` and ``ValueError`` from ``json.dumps(..., default=str)``
    must funnel through to the ``repr()`` fallback branch.

    json.dumps calls ``default=str`` on non-serializable objects; if ``str(obj)``
    itself raises, the exception propagates out of ``json.dumps``. We construct
    objects whose ``__str__`` raises (one TypeError, one ValueError) to cover
    both arms of the ``except`` clause.
    """
    from agent_sdk.loop import _serialize_tool_output

    class StrRaisesValueError:
        def __repr__(self) -> str:
            return "REPR_VALUE_ERR"

        def __str__(self) -> str:
            raise ValueError("str blew up")

    class StrRaisesTypeError:
        def __repr__(self) -> str:
            return "REPR_TYPE_ERR"

        def __str__(self) -> str:
            raise TypeError("str blew up with type error")

    value_err_obj = StrRaisesValueError()
    type_err_obj = StrRaisesTypeError()

    assert _serialize_tool_output(value_err_obj) == repr(value_err_obj)
    assert _serialize_tool_output(value_err_obj) == "REPR_VALUE_ERR"
    assert _serialize_tool_output(type_err_obj) == repr(type_err_obj)
    assert _serialize_tool_output(type_err_obj) == "REPR_TYPE_ERR"


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


def test_top_level_exports_loop_surface() -> None:
    from agent_sdk import (
        ActionEvent as _ActionEvent,
    )
    from agent_sdk import (
        AgentEvent as _AgentEvent,
    )
    from agent_sdk import (
        AgentLoop as _AgentLoop,
    )
    from agent_sdk import (
        ErrorEvent as _ErrorEvent,
    )
    from agent_sdk import (
        FinalEvent as _FinalEvent,
    )
    from agent_sdk import (
        ObservationEvent as _ObservationEvent,
    )
    from agent_sdk import (
        SafetyConfig as _SafetyConfig,
    )
    from agent_sdk import (
        ThoughtEvent as _ThoughtEvent,
    )
    from agent_sdk import (
        TokenEvent as _TokenEvent,
    )
    from agent_sdk import (
        ToolFailedEvent as _ToolFailedEvent,
    )
    from agent_sdk import (
        ToolProgressEvent as _ToolProgressEvent,
    )
    from agent_sdk import (
        ToolStartedEvent as _ToolStartedEvent,
    )

    assert _AgentLoop is AgentLoop
    assert _AgentEvent is AgentEvent
    assert _SafetyConfig is SafetyConfig
    assert _ThoughtEvent is ThoughtEvent
    assert _ActionEvent is ActionEvent
    assert _ToolStartedEvent is ToolStartedEvent
    assert _ObservationEvent is ObservationEvent
    assert _ToolFailedEvent is ToolFailedEvent
    assert _ToolProgressEvent is ToolProgressEvent
    assert _TokenEvent is TokenEvent
    assert _FinalEvent is FinalEvent
    assert _ErrorEvent is ErrorEvent


# ---------------------------------------------------------------------------
# Tool-message-role override (BR-017)
# ---------------------------------------------------------------------------


async def test_tool_message_role_user_emits_user_role_message() -> None:
    """Happy path: ``tool_message_role="user"`` collapses the synthetic
    post-tool message to ``role="user"`` carrying the tool name inline.

    This is the wire-format the Kalvad GDC proxy at ``gemini.kalvad.cloud``
    accepts — that provider rejects ``role="tool"`` with HTTP 500.
    """
    tool = FakeTool("search", result=ToolResult(output={"results": ["a", "b"]}))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("look it up", "search", {"q": "x"})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, tool_message_role="user")

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    # No role="tool" message must exist anywhere in the conversation.
    assert not any(m.role == "tool" for m in second_request.messages)
    # The synthesized message is a user-role message carrying the tool
    # name inline and the serialized output.
    synthesized = [
        m
        for m in second_request.messages
        if m.role == "user" and m.content.startswith("Tool search returned:")
    ]
    assert len(synthesized) == 1
    serialized = json.dumps({"results": ["a", "b"]})
    assert serialized in synthesized[0].content
    # tool_call_id / name fields are dropped in the collapsed envelope.
    assert synthesized[0].tool_call_id is None
    assert synthesized[0].name is None


async def test_tool_message_role_user_for_tool_error_branch() -> None:
    """``ToolResult(is_error=True)`` under ``tool_message_role="user"``
    surfaces as a user-role message prefixed ``"Tool {name} failed:"``."""
    tool = FakeTool(
        "search",
        result=ToolResult(output=None, is_error=True, error="boom"),
    )
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "search", {})),
            make_response(_final_json("recovered")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, tool_message_role="user")

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    assert not any(m.role == "tool" for m in second_request.messages)
    synthesized = [
        m
        for m in second_request.messages
        if m.role == "user" and m.content.startswith("Tool search failed:")
    ]
    assert len(synthesized) == 1
    assert synthesized[0].content == "Tool search failed: boom"


async def test_tool_message_role_user_for_tool_not_found_branch() -> None:
    """A ``ToolNotFound`` under ``tool_message_role="user"`` surfaces as a
    user-role message prefixed ``"Tool {name} failed: ToolNotFound:"``."""
    registry = Registry()  # no tools registered
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "ghost", {})),
            make_response(_final_json("gave up")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, tool_message_role="user")

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    assert not any(m.role == "tool" for m in second_request.messages)
    synthesized = [
        m
        for m in second_request.messages
        if m.role == "user" and m.content.startswith("Tool ghost failed: ToolNotFound:")
    ]
    assert len(synthesized) == 1


async def test_tool_message_role_user_for_tool_timeout_branch() -> None:
    """A ``ToolTimeout`` under ``tool_message_role="user"`` surfaces as a
    user-role message prefixed ``"Tool {name} failed: ToolTimeout:"``."""
    tool = FakeTool("slow", sleep_seconds=0.5)
    registry = Registry()
    registry.register(tool)
    safety = SafetyConfig(max_iterations=4, tool_timeout_seconds=0.05)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "slow", {})),
            make_response(_final_json("recovered")),
        ]
    )
    loop = _make_loop(
        llm=llm,
        registry=registry,
        safety=safety,
        tool_message_role="user",
    )

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    assert not any(m.role == "tool" for m in second_request.messages)
    synthesized = [
        m
        for m in second_request.messages
        if m.role == "user" and m.content.startswith("Tool slow failed: ToolTimeout:")
    ]
    assert len(synthesized) == 1


async def test_tool_message_role_assistant_emits_assistant_role_message() -> None:
    """Smoke test for ``tool_message_role="assistant"`` — the synthetic
    post-tool message uses ``role="assistant"`` with the same inline
    name-carrying content shape as the user-role variant."""
    tool = FakeTool("search", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "search", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry, tool_message_role="assistant")

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    assert not any(m.role == "tool" for m in second_request.messages)
    # The model's assistant turn ALSO has role="assistant"; we look for our
    # synthesized one by its content prefix.
    synthesized = [
        m
        for m in second_request.messages
        if m.role == "assistant" and m.content.startswith("Tool search returned:")
    ]
    assert len(synthesized) == 1
    assert "ok" in synthesized[0].content
    assert synthesized[0].tool_call_id is None
    assert synthesized[0].name is None


async def test_tool_message_role_default_is_tool_unchanged() -> None:
    """Lock the default: omitting ``tool_message_role`` keeps the OpenAI
    tool-role envelope with ``tool_call_id`` and ``name`` populated."""
    tool = FakeTool("search", result=ToolResult(output="ok"))
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("t", "search", {})),
            make_response(_final_json("done")),
        ]
    )
    loop = _make_loop(llm=llm, registry=registry)  # default

    await _collect(loop.run([ChatMessage(role="user", content="q")]))

    second_request = llm.calls[1]
    tool_msgs = [m for m in second_request.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "search"
    assert tool_msgs[0].tool_call_id is not None
    # Content must remain BYTE-IDENTICAL to the pre-BR-017 shape — the
    # serialized output with no "Tool ... returned:" prefix.
    assert tool_msgs[0].content == "ok"
