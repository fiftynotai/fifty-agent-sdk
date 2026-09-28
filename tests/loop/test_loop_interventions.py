"""Loop integration tests for the intervention hooks (FR-003).

Each test drives :class:`fifty_agent_sdk.loop.AgentLoop` with scripted fakes and
asserts on what reaches the wire (``OpenAICompatibleClient._build_body``
through :func:`tests.loop.golden_capture.wire_bodies`), the event stream and
the fake tools. Differential tests run the SAME script twice, with and without
the hook, on fresh fakes, and compare id-normalised bodies and events
(normaliser shared with ``golden_capture_1_9_0``).

What these pin:

* AC-1: an ``after_tool`` note lands, as ``observation + "\\n\\n" + note``, in
  exactly one message, in every (tool mode, tool-result role) pair, for both
  ToolResult outcomes, and in streamed turns; the ``ToolResult`` (identity and
  value, for success and ``is_error`` results, on the single path and in a
  batch) and the event stream are unchanged.
* AC-2: in a concurrent batch whose completion order differs from call order,
  each note lands on its own call's message, and ``before_tool`` decisions are
  per call.
* G1: ``after_tool`` runs after that call's terminal event was delivered, one
  call at a time in call order.
* AC-3: a raising ``after_tool`` leaves the observation unaugmented and the
  run continues; ``before_tool`` failures follow the configured fallback under
  both values; ``CancelledError`` propagates from either hook.
* AC-7: deny keeps the ToolNotFound event shape and skips dispatch; replace
  dispatches the replacement without rewriting the model's turn.
* AC-4 companion: configured-but-passive interventions change nothing, and
  without a hook no deep copy is made.
* Argument flow: ``after_tool`` sees each call's dispatched args (the
  replacement or the model's), never a tool's later top-level edits; hook
  edits at any depth never leak; uncopyable args take the one-level fallback
  without ever being treated as a hook failure.
* Dispatch log lines: ``tool_invoked`` per call as in 1.9.0, ``tool_denied``
  for a denied call.

What these do NOT pin: how a real model reads a note (in the ``"assistant"``
versus ``"user"`` role), or whether it stops reading rows back. That is the
consumer's live check; this suite makes no network call.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import pytest
import structlog
from pydantic import PrivateAttr

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    ActionEvent,
    AgentEvent,
    AgentLoop,
    BeforeToolFallback,
    ChatMessage,
    ChatResponse,
    DenyToolCall,
    FinalEvent,
    Hooks,
    Interventions,
    JsonModeParser,
    ObservationEvent,
    PromptSections,
    Registry,
    ReplaceToolArgs,
    SafetyConfig,
    ThoughtEvent,
    ToolFailedEvent,
    ToolMode,
    ToolResult,
    ToolStartedEvent,
)
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)
from tests.loop.golden_capture import wire_bodies
from tests.loop.golden_capture_1_9_0 import _normalise_scenario, serialise_events

_MODEL = "intervention-model"
_USER = [ChatMessage(role="user", content="Show me the open records.")]
_NOTE = "The user can already see these records in the UI; do not list them again."
_SDK_DENIAL = "Tool call denied: this call was not approved, so it was not run."
_SECRET = "SECRET-before-tool-args-DO-NOT-LOG"

Role = Literal["tool", "user", "assistant"]
Scenario = dict[str, list[dict[str, Any]]]


# --- Rows: every (tool mode, tool-result role) pair ----------------------------------


class _Row(NamedTuple):
    """One loop configuration and the role its tool observation must go out in."""

    tool_mode: ToolMode | None  # None: the legacy path (JsonModeParser, no tool_mode)
    role: Role | None  # the tool_message_role kwarg; None: not passed
    native_flag: bool  # legacy SafetyConfig(native_tools_enabled=True)
    expected_role: str


_JSON = _Row(ToolMode.JSON, None, False, "assistant")
_JSON_USER = _Row(ToolMode.JSON, "user", False, "user")
_PROSE = _Row(ToolMode.PROSE, None, False, "assistant")
_PROSE_USER = _Row(ToolMode.PROSE, "user", False, "user")
_NATIVE = _Row(ToolMode.NATIVE, None, False, "tool")
_LEGACY = _Row(None, None, False, "tool")
_LEGACY_ASSISTANT = _Row(None, "assistant", False, "assistant")
_LEGACY_USER = _Row(None, "user", False, "user")
_LEGACY_NATIVE_FLAG = _Row(None, None, True, "tool")

_EVERY_ROW = [
    pytest.param(_JSON, id="json"),
    pytest.param(_JSON_USER, id="json_user"),
    pytest.param(_PROSE, id="prose"),
    pytest.param(_PROSE_USER, id="prose_user"),
    pytest.param(_NATIVE, id="native"),
    pytest.param(_LEGACY, id="legacy_tool"),
    pytest.param(_LEGACY_ASSISTANT, id="legacy_assistant"),
    pytest.param(_LEGACY_USER, id="legacy_user"),
    pytest.param(_LEGACY_NATIVE_FLAG, id="legacy_native_flag"),
]


# --- Builders -----------------------------------------------------------------------


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


def _is_native(row: _Row) -> bool:
    return row.tool_mode is ToolMode.NATIVE or row.native_flag


def _tool_text(row: _Row, name: str, args: dict[str, Any]) -> str:
    return _prose_tool(name, args) if row.tool_mode is ToolMode.PROSE else _json_tool(name, args)


def _tool_turn(row: _Row, name: str, args: dict[str, Any]) -> ChatResponse:
    """One model turn calling ``name`` in the row's protocol."""
    if _is_native(row):
        return make_multi_tool_response([(name, args)])
    return make_response(_tool_text(row, name, args))


def _final_text(row: _Row, answer: str = "done") -> str:
    if row.tool_mode is ToolMode.NATIVE:
        return answer
    if row.tool_mode is ToolMode.PROSE:
        return _prose_final(answer)
    return _json_final(answer)


def _final_turn(row: _Row, answer: str = "done") -> ChatResponse:
    return make_response(_final_text(row, answer))


def _tools() -> dict[str, FakeTool]:
    """``ok`` succeeds, ``bad`` returns ``is_error``, ``slow`` outlives the 0.05 s timeout."""
    return {
        "ok": FakeTool("ok", result=ToolResult(output={"rows": [{"id": 1}, {"id": 2}]})),
        "bad": FakeTool("bad", result=ToolResult(is_error=True, error="upstream returned 503")),
        "slow": FakeTool("slow", sleep_seconds=1.0),
    }


class _CompletionRecordingTool(FakeTool):
    """A :class:`FakeTool` that appends its name to ``completed`` when ``invoke`` finishes.

    With ``wait_for`` it first waits for that event, and ``on_complete`` runs
    right after it records its completion, so a test can force the completion
    order without a wall-clock sleep.
    """

    def __init__(
        self,
        name: str,
        completed: list[str],
        *,
        wait_for: asyncio.Event | None = None,
        on_complete: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._completed = completed
        self._wait_for = wait_for
        self._on_complete = on_complete

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        if self._wait_for is not None:
            await self._wait_for.wait()
        result = await super().invoke(args)
        self._completed.append(self.name)
        if self._on_complete is not None:
            self._on_complete()
        return result


class _TopLevelArgEditingTool(FakeTool):
    """A :class:`FakeTool` that edits its own args dict at the top level after running."""

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        result = await super().invoke(args)
        args["q"] = "edited-by-tool"
        args["tool_added"] = "TOOL-TOP"
        return result


class _Uncopyable:
    """An argument value ``copy.deepcopy`` rejects, as a custom parser might produce."""

    def __deepcopy__(self, memo: dict[int, Any]) -> _Uncopyable:
        raise TypeError("this handle cannot be copied")


class _WriteRecordingToolResult(ToolResult):
    """A :class:`ToolResult` that records every attribute write made to it after construction.

    Pins "the SDK never writes to the tool's ``ToolResult``" (FR-003 review
    round 1) for writes of every kind: a changed value, a same-value write
    that only adds to ``model_fields_set``, an equal replacement, and a write
    undone later. Construction does not go through ``__setattr__``, so a fresh
    instance records nothing. It cannot see a change made INSIDE a value
    (``result.output["k"] = v``); the ``model_dump()`` snapshots cover those.
    """

    _writes: list[tuple[str, Any]] = PrivateAttr(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_writes":
            self._writes.append((name, value))
        super().__setattr__(name, value)


_NESTED_ARGS: dict[str, Any] = {"q": "open", "filters": {"tenant": "t-1", "tags": ["a", "b"]}}
"""Model arguments with nested values; every script gets a fresh deep copy of them."""


def _edit_at_every_depth(args: dict[str, Any]) -> None:
    """What a careless hook does to its ``args``: edit the top level and nested values in place."""
    args["q"] = "changed"
    args["filters"]["tenant"] = "t-2"
    args["filters"]["tags"].append("injected")


def _registry(tools: dict[str, FakeTool]) -> Registry:
    registry = Registry()
    for tool in tools.values():
        registry.register(tool)
    return registry


def _loop(
    llm: FakeLLMClient,
    row: _Row,
    tools: dict[str, FakeTool],
    *,
    interventions: Interventions | None = None,
    stream: bool = False,
    **safety_kwargs: Any,
) -> AgentLoop:
    """A loop for ``row``; ``interventions=None`` is the 1.9.0 construction."""
    safety_kwargs.setdefault("tool_timeout_seconds", 0.05)
    if row.native_flag:
        safety_kwargs["native_tools_enabled"] = True
    kwargs: dict[str, Any] = {}
    if row.role is not None:
        kwargs["tool_message_role"] = row.role
    if row.tool_mode is None:
        kwargs["parser"] = JsonModeParser()
        kwargs["output_format"] = JSON_MODE_OUTPUT_FORMAT
    else:
        kwargs["tool_mode"] = row.tool_mode
    return AgentLoop(
        llm=llm,
        registry=_registry(tools),
        prompts=PromptSections(persona="You are a careful records assistant."),
        safety=SafetyConfig(**safety_kwargs),
        model=_MODEL,
        stream=stream,
        interventions=interventions,
        **kwargs,
    )


def _nested_args_loop(
    batch: bool, interventions: Interventions | None
) -> tuple[AgentLoop, FakeLLMClient, dict[str, FakeTool], list[str]]:
    """A NATIVE loop whose model calls ``ok`` (and ``ok2``, as a batch) with :data:`_NESTED_ARGS`."""
    tools = {
        name: FakeTool(name, result=ToolResult(output=f"out-{name}")) for name in ("ok", "ok2")
    }
    names = ["ok", "ok2"] if batch else ["ok"]
    llm = FakeLLMClient(
        [
            make_multi_tool_response([(name, copy.deepcopy(_NESTED_ARGS)) for name in names]),
            make_response("done"),
        ]
    )
    loop = _loop(llm, _NATIVE, tools, interventions=interventions, max_concurrent_tool_calls=2)
    return loop, llm, tools, names


async def _drive(
    loop: AgentLoop,
    llm: FakeLLMClient,
    *,
    session_id: str | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
) -> tuple[list[AgentEvent], Scenario]:
    """Run to completion; return the events and the id-normalised bodies + events.

    ``on_event`` is called with each event as it arrives, inline, before the
    loop resumes: what a consumer's ``async for`` body sees.
    """
    events: list[AgentEvent] = []
    async for event in loop.run(list(_USER), session_id=session_id):
        events.append(event)
        if on_event is not None:
            on_event(event)
    bodies = await wire_bodies(llm, stream=loop._stream)
    return events, _normalise_scenario(bodies, serialise_events(events))


def _constant(value: object) -> Callable[..., object]:
    """A hook that ignores its arguments and returns ``value``."""

    def hook(*_args: object) -> object:
        return value

    return hook


def _of(events: list[AgentEvent], kind: type[Any]) -> list[Any]:
    return [event for event in events if isinstance(event, kind)]


def _warnings(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in logs
        if str(entry.get("event", "")).startswith("intervention.")
        and entry.get("log_level") == "warning"
    ]


def _assert_only_the_observation_gained_the_note(
    base: Scenario, hooked: Scenario, *, body: int, note: str
) -> dict[str, Any]:
    """Exactly one message of body ``body`` differs, and only by ``"\\n\\n" + note``.

    Every other body, every other message, and every other key of the changed
    message (role, ``tool_call_id``, ``name``) must be equal. Returns the
    changed message.
    """
    assert len(hooked["bodies"]) == len(base["bodies"])
    for index, (b, h) in enumerate(zip(base["bodies"], hooked["bodies"], strict=True)):
        if index != body:
            assert h == b
    before = base["bodies"][body]["messages"]
    after = hooked["bodies"][body]["messages"]
    assert len(after) == len(before)
    changed = [i for i, (b, h) in enumerate(zip(before, after, strict=True)) if b != h]
    assert len(changed) == 1
    i = changed[0]
    assert after[i]["content"] == before[i]["content"] + "\n\n" + note
    assert {k: v for k, v in after[i].items() if k != "content"} == {
        k: v for k, v in before[i].items() if k != "content"
    }
    rest = {k: v for k, v in hooked["bodies"][body].items() if k != "messages"}
    assert rest == {k: v for k, v in base["bodies"][body].items() if k != "messages"}
    changed_message: dict[str, Any] = after[i]
    return changed_message


# --- after_tool: AC-1 -------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["ok", "bad"])
@pytest.mark.parametrize("row", _EVERY_ROW)
async def test_after_tool_note_is_appended_to_the_observation_in_every_mode_and_role(
    row: _Row, outcome: str
) -> None:
    """The note reaches exactly one message, as ``observation + "\\n\\n" + note``, in every mode and role (FR-003 AC-1).

    ``bad`` is the ``is_error`` outcome. The changed message keeps its role,
    ``tool_call_id`` and ``name``; request 0 and every other key are equal.
    """

    async def run(interventions: Interventions | None) -> tuple[FakeLLMClient, Scenario]:
        llm = FakeLLMClient([_tool_turn(row, outcome, {"q": "open"}), _final_turn(row)])
        loop = _loop(llm, row, _tools(), interventions=interventions)
        _events, scenario = await _drive(loop, llm)
        return llm, scenario

    base_llm, base = await run(None)
    hook_llm, hooked = await run(Interventions(after_tool=_constant(_NOTE)))

    assert len(base_llm.calls) == len(hook_llm.calls) == 2
    message = _assert_only_the_observation_gained_the_note(base, hooked, body=1, note=_NOTE)
    assert message is hooked["bodies"][1]["messages"][-1]
    assert message["role"] == row.expected_role
    if row.expected_role == "tool":
        assert message["name"] == outcome
        assistant = hooked["bodies"][1]["messages"][-2]
        if _is_native(row):
            assert [c["id"] for c in assistant["tool_calls"]] == [message["tool_call_id"]]
    else:
        assert "tool_call_id" not in message
        assert "name" not in message
    assert hooked["events"] == base["events"]


@pytest.mark.parametrize("mode", [ToolMode.JSON, ToolMode.NATIVE], ids=["json", "native_batch"])
async def test_after_tool_leaves_tool_result_and_events_untouched(mode: ToolMode) -> None:
    """The ``ToolResult`` reaches the event and the hook BY IDENTITY and the SDK never writes to it; events equal the no-hook run (FR-003 AC-1).

    JSON exercises the single-call path, NATIVE a two-call batch (the two
    success ``after_tool`` sites). What is pinned: no attribute write at all
    during the run, with or without the hook (a recording ``ToolResult``
    subclass), and a ``model_dump()`` equal to the tool's when the consumer
    receives the call's terminal event, when the hook is called, and after the
    run. Not pinned: a change INSIDE a value that the SDK makes and undoes
    entirely between those three points.
    """

    def build(interventions: Interventions | None) -> tuple[AgentLoop, FakeLLMClient, list[Any]]:
        results = [
            _WriteRecordingToolResult(output={"rows": [{"id": 1}, {"id": 2}]}),
            _WriteRecordingToolResult(output=["x"]),
        ]
        tools = {
            "ok": FakeTool("ok", result=results[0]),
            "ok2": FakeTool("ok2", result=results[1]),
        }
        if mode is ToolMode.NATIVE:
            replies = [
                make_multi_tool_response([("ok", {"q": "a"}), ("ok2", {"q": "b"})]),
                make_response("done"),
            ]
        else:
            replies = [
                make_response(_json_tool("ok", {"q": "a"})),
                make_response(_json_tool("ok2", {"q": "b"})),
                make_response(_json_final("done")),
            ]
        llm = FakeLLMClient(replies)
        row = _NATIVE if mode is ToolMode.NATIVE else _JSON
        loop = _loop(llm, row, tools, interventions=interventions, max_concurrent_tool_calls=2)
        return loop, llm, results

    received: list[ToolResult] = []
    seen_by_hook: list[dict[str, Any]] = []
    seen_by_consumer: list[dict[str, Any]] = []

    def after_tool(*args: Any) -> str:
        received.append(args[4])
        seen_by_hook.append(args[4].model_dump())
        return _NOTE

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, ObservationEvent):
            seen_by_consumer.append(event.result.model_dump())

    loop, llm, results = build(Interventions(after_tool=after_tool))
    dumps = [result.model_dump() for result in results]
    events, hooked = await _drive(loop, llm, on_event=on_event)

    observations = _of(events, ObservationEvent)
    assert len(observations) == len(received) == 2
    for observation, got, result in zip(observations, received, results, strict=True):
        assert observation.result is result
        assert got is result
    assert seen_by_consumer == dumps
    assert seen_by_hook == dumps
    assert [result.model_dump() for result in results] == dumps
    assert [result._writes for result in results] == [[], []]
    assert results[0].output == {"rows": [{"id": 1}, {"id": 2}]}

    base_loop, base_llm, base_results = build(None)
    _events, base = await _drive(base_loop, base_llm)
    assert hooked["events"] == base["events"]
    assert [result._writes for result in base_results] == [[], []]


@pytest.mark.parametrize(
    "raw_error",
    [pytest.param("upstream returned 503", id="error_text"), pytest.param(None, id="error_none")],
)
@pytest.mark.parametrize("mode", [ToolMode.JSON, ToolMode.NATIVE], ids=["json", "native_batch"])
async def test_after_tool_leaves_an_is_error_tool_result_and_its_events_untouched(
    mode: ToolMode, raw_error: str | None
) -> None:
    """An ``is_error`` ``ToolResult`` reaches the hook BY IDENTITY and the SDK never writes to it; the event and the observation keep the 1.9.0 error text; the note still reaches the model (FR-003 AC-1).

    JSON exercises the single-call path (two calls in a row), NATIVE a
    concurrent two-call batch (the two ``is_error`` ``after_tool`` sites).
    ``error_none`` is a tool that reports failure with ``error=None``:
    ``ToolFailedEvent`` and the observation then carry the SDK's fallback text,
    as 1.9.0 already did, while the tool's own object must keep ``error=None``.
    Both calls return the SAME ``ToolResult`` object, as a tool returning a
    cached or constant error would, so an SDK write to it also reaches the
    second call's event and observation. What is pinned: no attribute write at
    all during the run, with or without the hook (a recording ``ToolResult``
    subclass), and a ``model_dump()`` equal to the tool's when the consumer
    receives each terminal event, when the hook is called, and after the run.
    Not pinned: a change INSIDE a value that the SDK makes and undoes entirely
    between those points.
    """
    shown_error = raw_error if raw_error is not None else "tool reported error with no message"

    def build(interventions: Interventions | None) -> tuple[AgentLoop, FakeLLMClient, ToolResult]:
        result = _WriteRecordingToolResult(is_error=True, error=raw_error)
        tools = {"bad": FakeTool("bad", result=result)}
        if mode is ToolMode.NATIVE:
            replies = [
                make_multi_tool_response([("bad", {"q": "a"}), ("bad", {"q": "b"})]),
                make_response("done"),
            ]
        else:
            replies = [
                make_response(_json_tool("bad", {"q": "a"})),
                make_response(_json_tool("bad", {"q": "b"})),
                make_response(_json_final("done")),
            ]
        llm = FakeLLMClient(replies)
        row = _NATIVE if mode is ToolMode.NATIVE else _JSON
        loop = _loop(llm, row, tools, interventions=interventions, max_concurrent_tool_calls=2)
        return loop, llm, result

    received: list[ToolResult] = []
    seen_by_hook: list[dict[str, Any]] = []
    seen_by_consumer: list[dict[str, Any]] = []

    def after_tool(*args: Any) -> str:
        received.append(args[4])
        seen_by_hook.append(args[4].model_dump())
        return _NOTE

    loop, llm, result = build(Interventions(after_tool=after_tool))

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, ToolFailedEvent):
            seen_by_consumer.append(result.model_dump())

    dump = result.model_dump()
    events, hooked = await _drive(loop, llm, on_event=on_event)

    assert len(received) == 2
    assert all(got is result for got in received)
    assert seen_by_consumer == [dump, dump]
    assert seen_by_hook == [dump, dump]
    assert result.model_dump() == dump
    assert result._writes == []
    assert result.error == raw_error
    assert [event.error for event in _of(events, ToolFailedEvent)] == [shown_error, shown_error]
    if mode is ToolMode.NATIVE:
        observations = [m.content for m in llm.calls[-1].messages if m.role == "tool"]
        expected = f"Tool error: {shown_error}\n\n{_NOTE}"
    else:
        observations = [
            m.content for m in llm.calls[-1].messages if m.content.startswith("Tool bad failed:")
        ]
        expected = f"Tool bad failed: {shown_error}\n\n{_NOTE}"
    assert observations == [expected, expected]

    base_loop, base_llm, base_result = build(None)
    _events, base = await _drive(base_loop, base_llm)
    assert hooked["events"] == base["events"]
    assert base_result._writes == []


@pytest.mark.parametrize("row", [pytest.param(_JSON, id="json"), pytest.param(_PROSE, id="prose")])
async def test_after_tool_note_reaches_streamed_turns(row: _Row) -> None:
    """With ``stream=True`` the note reaches the next streamed request the same way (FR-003 AC-1)."""

    async def run(interventions: Interventions | None) -> Scenario:
        tool_text = _tool_text(row, "ok", {"q": "open"})
        final_text = _final_text(row, "streamed answer")
        llm = FakeLLMClient(
            [
                make_stream_chunks([tool_text[:12], tool_text[12:]]),
                make_stream_chunks([final_text[:10], final_text[10:]]),
            ]
        )
        loop = _loop(llm, row, _tools(), interventions=interventions, stream=True)
        _events, scenario = await _drive(loop, llm)
        return scenario

    base = await run(None)
    hooked = await run(Interventions(after_tool=_constant(_NOTE)))

    message = _assert_only_the_observation_gained_the_note(base, hooked, body=1, note=_NOTE)
    assert message["role"] == "assistant"
    assert hooked["bodies"][1]["stream"] is True
    assert hooked["events"] == base["events"]


# --- after_tool: AC-2 and G1 ------------------------------------------------------------


async def test_after_tool_notes_attach_to_their_own_call_in_a_concurrent_batch() -> None:
    """In a batch whose completion order is not call order, each note lands on its own call (FR-003 AC-2).

    ``a`` waits on an event that is set only once ``b`` and ``c`` have
    completed, so it completes last without any wall-clock sleep; ``b`` is
    ``is_error``; ``missing`` is unregistered and gets no note. Each note
    embeds the ``call_id``, name and ``args["q"]`` the hook received, so a note
    on a sibling's message fails.
    """
    completed: list[str] = []
    release_a = asyncio.Event()

    def release_a_once_b_and_c_are_done() -> None:
        if {"b", "c"} <= set(completed):
            release_a.set()

    tools: dict[str, FakeTool] = {
        "a": _CompletionRecordingTool(
            "a", completed, wait_for=release_a, result=ToolResult(output="out-a")
        ),
        "b": _CompletionRecordingTool(
            "b",
            completed,
            on_complete=release_a_once_b_and_c_are_done,
            result=ToolResult(is_error=True, error="err-b"),
        ),
        "c": _CompletionRecordingTool(
            "c",
            completed,
            on_complete=release_a_once_b_and_c_are_done,
            result=ToolResult(output="out-c"),
        ),
    }
    calls = [("a", {"q": "qa"}), ("b", {"q": "qb"}), ("c", {"q": "qc"}), ("missing", {"q": "qm"})]
    llm = FakeLLMClient([make_multi_tool_response(calls), make_response("done")])
    hook_calls: list[tuple[str, str]] = []

    def after_tool(_sid: object, call_id: str, name: str, args: dict[str, Any], _r: object) -> str:
        hook_calls.append((call_id, name))
        return f"{call_id}|{name}|{args['q']}"

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(after_tool=after_tool),
        max_concurrent_tool_calls=3,
        tool_timeout_seconds=5.0,
    )
    events = [event async for event in loop.run(list(_USER))]

    # Completion order really differs from call order: `a` finished last.
    assert completed == ["b", "c", "a"]
    started = {event.tool_name: event.call_id for event in _of(events, ToolStartedEvent)}
    follow_up = llm.calls[1].messages
    tool_messages = [m for m in follow_up if m.role == "tool"]
    assert [m.name for m in tool_messages] == ["a", "b", "c", "missing"]
    for message in tool_messages[:3]:
        assert message.tool_call_id == started[message.name]
        expected = f"{message.tool_call_id}|{message.name}|q{message.name}"
        assert message.content.endswith("\n\n" + expected)
        assert message.content.count("\n\n") == 1
    assert tool_messages[3].content == "ToolNotFound: tool 'missing' is not registered."
    assistant = follow_up[-5]
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == [m.tool_call_id for m in tool_messages]
    assert hook_calls == [(started[name], name) for name in ("a", "b", "c")]


@pytest.mark.parametrize("batch", [False, True], ids=["single_path", "native_batch"])
async def test_after_tool_runs_after_each_terminal_event_in_call_order(batch: bool) -> None:
    """``after_tool`` for a call runs after the consumer received THAT call's terminal event, one call at a time (FR-003 G1).

    The hook snapshots the terminal-event ids the consumer has seen so far;
    each snapshot must be exactly the ids up to and including its own call.
    """
    tools = {name: FakeTool(name, result=ToolResult(output=f"out-{name}")) for name in "xyz"}
    if batch:
        row = _NATIVE
        replies = [
            make_multi_tool_response([("x", {}), ("y", {}), ("z", {})]),
            make_response("done"),
        ]
    else:
        row = _JSON
        replies = [
            make_response(_json_tool("x", {})),
            make_response(_json_tool("y", {})),
            make_response(_json_final("done")),
        ]
    seen_terminal: list[str] = []
    snapshots: list[tuple[str, list[str]]] = []

    def after_tool(_sid: object, call_id: str, *_rest: object) -> None:
        snapshots.append((call_id, list(seen_terminal)))

    llm = FakeLLMClient(replies)
    loop = _loop(
        llm,
        row,
        tools,
        interventions=Interventions(after_tool=after_tool),
        max_concurrent_tool_calls=3,
    )
    async for event in loop.run(list(_USER)):
        if isinstance(event, ObservationEvent | ToolFailedEvent):
            seen_terminal.append(event.call_id)

    assert len(snapshots) == len(seen_terminal) == (3 if batch else 2)
    for index, (call_id, seen) in enumerate(snapshots):
        assert call_id == seen_terminal[index]
        assert seen == seen_terminal[: index + 1]


async def test_after_tool_is_not_called_for_not_found_timeout_or_denied_calls() -> None:
    """``after_tool`` runs only where a ToolResult exists: not for ToolNotFound, ToolTimeout or a denial (FR-003 D2)."""
    tools = _tools()
    tools["blocked"] = FakeTool("blocked")
    llm = FakeLLMClient(
        [
            make_response(_json_tool("missing", {})),
            make_response(_json_tool("slow", {})),
            make_response(_json_tool("blocked", {})),
            make_response(_json_tool("ok", {})),
            make_response(_json_final("done")),
        ]
    )
    after_calls: list[str] = []

    def before_tool(_sid: object, _cid: object, name: str, _args: object) -> DenyToolCall | None:
        return DenyToolCall(reason="blocked by policy") if name == "blocked" else None

    def after_tool(_sid: object, _cid: object, name: str, *_rest: object) -> None:
        after_calls.append(name)

    loop = _loop(
        llm,
        _JSON,
        tools,
        interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
    )
    events = [event async for event in loop.run(list(_USER))]

    assert after_calls == ["ok"]
    assert tools["blocked"].call_count == 0
    errors = [event.error for event in _of(events, ToolFailedEvent)]
    assert errors[0].startswith("ToolNotFound:")
    assert errors[1].startswith("ToolTimeout:")
    assert errors[2] == "Tool call denied: blocked by policy"
    assert isinstance(events[-1], FinalEvent)


@pytest.mark.parametrize("session_id", [None, "session-7"], ids=["no_runner", "session_id"])
async def test_after_tool_receives_session_id_call_id_and_dispatched_args(
    session_id: str | None,
) -> None:
    """The hook gets ``session_id`` (``None`` without a Runner), the event's ``call_id`` and a private copy of the dispatched args (FR-003 D2)."""
    tools = _tools()
    llm = FakeLLMClient(
        [make_response(_json_tool("ok", {"q": "open"})), make_response(_json_final("done"))]
    )
    received: dict[str, Any] = {}

    def after_tool(sid: object, call_id: str, name: str, args: dict[str, Any], _r: object) -> None:
        received.update(sid=sid, call_id=call_id, name=name, args=args, args_seen=dict(args))
        args["injected"] = True
        args["q"] = "tampered"

    loop = _loop(llm, _JSON, tools, interventions=Interventions(after_tool=after_tool))
    events = [event async for event in loop.run(list(_USER), session_id=session_id)]

    action = _of(events, ActionEvent)[0]
    observation = _of(events, ObservationEvent)[0]
    assert received["sid"] == session_id
    assert received["call_id"] == observation.call_id
    assert received["name"] == "ok"
    assert received["args_seen"] == {"q": "open"}
    assert received["args"] is not action.args
    assert received["args"] is not tools["ok"].last_args
    assert action.args == {"q": "open"}
    assert tools["ok"].last_args == {"q": "open"}


@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_after_tool_nested_edits_never_reach_events_the_model_turn_or_the_tool(
    batch: bool,
) -> None:
    """``after_tool`` edits its own deep copy, nested values included: events, the model's replayed turn and the tool's args equal a run without the hook (FR-003 review round 1).

    The hook returns ``None``, so the whole bodies must match. With a shallow
    copy the nested edits would reach ``ActionEvent.args``, the replayed turn
    and the args the tool received, which all share the model's nested values.
    """

    def after_tool(
        _sid: object, _cid: object, _name: object, args: dict[str, Any], _r: object
    ) -> None:
        _edit_at_every_depth(args)

    loop, llm, tools, names = _nested_args_loop(batch, Interventions(after_tool=after_tool))
    events, hooked = await _drive(loop, llm)
    base_loop, base_llm, _tools, _names = _nested_args_loop(batch, None)
    _events, base = await _drive(base_loop, base_llm)

    assert hooked == base
    assert [tools[name].last_args for name in names] == [_NESTED_ARGS] * len(names)
    assert [event.args for event in _of(events, ActionEvent)] == [_NESTED_ARGS] * len(names)


@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_after_tool_sees_the_args_as_dispatched_not_the_tools_later_edits(
    batch: bool,
) -> None:
    """A tool that edits its args dict at the top level cannot change what ``after_tool`` sees: dispatch hands the tool its own copy (FR-003 D2).

    Negative space: dispatch copies ONE level deep, as in 1.9.0, so a tool
    editing a nested value in place would still be visible to ``after_tool``;
    that is not pinned. Without any hook, dispatching the args object itself
    instead of a copy is an equivalent mutant (sentinel r4 "d1"): nothing
    reads the args after the tool runs, so no body or event can differ.
    """
    names = ["a", "b"] if batch else ["a"]
    tools = {
        name: _TopLevelArgEditingTool(name, result=ToolResult(output="done")) for name in names
    }
    calls = [(name, {"q": f"q{name}"}) for name in names]
    llm = FakeLLMClient([make_multi_tool_response(calls), make_response("done")])
    seen: list[dict[str, Any]] = []

    def after_tool(
        _sid: object, _cid: object, _name: object, args: dict[str, Any], _r: object
    ) -> None:
        seen.append(dict(args))

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(after_tool=after_tool),
        max_concurrent_tool_calls=2,
    )
    events = [event async for event in loop.run(list(_USER))]

    assert seen == [{"q": f"q{name}"} for name in names]
    assert [tools[name].last_args for name in names] == [
        {"q": "edited-by-tool", "tool_added": "TOOL-TOP"} for _name in names
    ]
    assert [event.args for event in _of(events, ActionEvent)] == [
        {"q": f"q{name}"} for name in names
    ]


# --- after_tool: AC-3 --------------------------------------------------------------------


@pytest.mark.parametrize("batch", [False, True], ids=["single_path", "native_batch"])
async def test_raising_after_tool_leaves_observation_unaugmented_and_run_continues(
    batch: bool,
) -> None:
    """A raising ``after_tool`` sends the no-hook bodies, the run reaches its final, one type-only WARNING per call (FR-003 AC-3)."""

    def build(interventions: Interventions | None) -> tuple[AgentLoop, FakeLLMClient]:
        if batch:
            row = _NATIVE
            replies = [
                make_multi_tool_response([("ok", {"q": "a"}), ("bad", {"q": "b"})]),
                make_response("done"),
            ]
        else:
            row = _JSON
            replies = [
                make_response(_json_tool("ok", {"q": "a"})),
                make_response(_json_tool("bad", {"q": "b"})),
                make_response(_json_final("done")),
            ]
        llm = FakeLLMClient(replies)
        loop = _loop(llm, row, _tools(), interventions=interventions, max_concurrent_tool_calls=2)
        return loop, llm

    def after_tool(*_args: object) -> str:
        raise RuntimeError(f"annotator failed on {_SECRET}")

    loop, llm = build(Interventions(after_tool=after_tool))
    with structlog.testing.capture_logs() as logs:
        events, hooked = await _drive(loop, llm)
    base_loop, base_llm = build(None)
    _events, base = await _drive(base_loop, base_llm)

    assert hooked == base
    assert isinstance(events[-1], FinalEvent)
    warnings = _warnings(logs)
    terminal_ids = [event.call_id for event in _of(events, ObservationEvent | ToolFailedEvent)]
    assert [w["call_id"] for w in warnings] == terminal_ids
    for warning in warnings:
        assert warning["event"] == "intervention.hook_failed"
        assert warning["hook_name"] == "after_tool"
        assert warning["error_type"] == "RuntimeError"
        assert warning["fallback"] == "observation_unaugmented"
        assert _SECRET not in str(warning)


@pytest.mark.parametrize("batch", [False, True], ids=["single_path", "native_batch"])
async def test_after_tool_cancelled_error_propagates_out_of_run(batch: bool) -> None:
    """``CancelledError`` raised by ``after_tool`` ends the run; no final, no further request (FR-003 AC-3)."""

    async def after_tool(*_args: object) -> str:
        raise asyncio.CancelledError

    if batch:
        row = _NATIVE
        replies = [
            make_multi_tool_response([("ok", {}), ("bad", {})]),
            make_response("done"),
        ]
    else:
        row = _JSON
        replies = [make_response(_json_tool("ok", {})), make_response(_json_final("done"))]
    llm = FakeLLMClient(replies)
    loop = _loop(
        llm,
        row,
        _tools(),
        interventions=Interventions(after_tool=after_tool),
        max_concurrent_tool_calls=2,
    )

    events: list[AgentEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in loop.run(list(_USER)):
            events.append(event)

    assert len(llm.calls) == 1
    assert not _of(events, FinalEvent)


# --- before_tool: AC-7 ------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_JSON, id="json_assistant"),
        pytest.param(_PROSE_USER, id="prose_user"),
        pytest.param(_NATIVE, id="native"),
        pytest.param(_LEGACY, id="legacy_tool"),
    ],
)
async def test_before_tool_deny_skips_dispatch_and_reports_the_reason(row: _Row) -> None:
    """A deny runs nothing and keeps the ToolNotFound event shape; the model reads the reason in its role's shape (FR-003 AC-7)."""
    tools = _tools()
    llm = FakeLLMClient([_tool_turn(row, "ok", {"q": "other-tenant"}), _final_turn(row)])
    seen: list[tuple[str, dict[str, Any]]] = []

    def before_tool(_sid: object, _cid: object, name: str, args: dict[str, Any]) -> DenyToolCall:
        seen.append((name, dict(args)))
        return DenyToolCall(reason="not in this tenant")

    loop = _loop(llm, row, tools, interventions=Interventions(before_tool=before_tool))
    events = [event async for event in loop.run(list(_USER))]

    denial = "Tool call denied: not in this tenant"
    assert tools["ok"].call_count == 0
    assert seen == [("ok", {"q": "other-tenant"})]
    assert [type(e) for e in events] == [
        ThoughtEvent,
        ActionEvent,
        ToolStartedEvent,
        ToolFailedEvent,
        ThoughtEvent,
        FinalEvent,
    ]
    action, started, failed = events[1], events[2], events[3]
    assert action.args == {"q": "other-tenant"}
    assert failed.error == denial
    assert failed.call_id == started.call_id
    assert failed.tool_name == started.tool_name == "ok"
    observation = llm.calls[1].messages[-1]
    assert observation.role == row.expected_role
    if row.expected_role == "tool":
        assert observation.content == denial
        assert observation.tool_call_id == started.call_id
        assert observation.name == "ok"
    else:
        assert observation.content == f"Tool ok failed: {denial}"
    if _is_native(row):
        assistant = llm.calls[1].messages[-2]
        assert assistant.tool_call_id == started.call_id


@pytest.mark.parametrize(
    "row", [pytest.param(_JSON, id="json"), pytest.param(_NATIVE, id="native")]
)
async def test_before_tool_replacement_is_dispatched_and_the_model_turn_is_not_rewritten(
    row: _Row,
) -> None:
    """The replacement is dispatched and shown on ``ActionEvent`` and to ``after_tool``; the model's own turn keeps its args (FR-003 D5)."""
    tools = _tools()
    model_args = {"q": "everything", "limit": 500}
    replacement = {"q": "everything", "limit": 50, "tenant": "t-1"}
    llm = FakeLLMClient([_tool_turn(row, "ok", model_args), _final_turn(row)])
    after_args: list[dict[str, Any]] = []

    def after_tool(
        _sid: object, _cid: object, _name: object, args: dict[str, Any], _r: object
    ) -> None:
        after_args.append(dict(args))

    loop = _loop(
        llm,
        row,
        tools,
        interventions=Interventions(
            before_tool=_constant(ReplaceToolArgs(args=replacement)), after_tool=after_tool
        ),
    )
    events = [event async for event in loop.run(list(_USER))]

    assert tools["ok"].last_args == replacement
    assert _of(events, ActionEvent)[0].args == replacement
    assert after_args == [replacement]
    assistant = llm.calls[1].messages[-2]
    assert assistant.role == "assistant"
    if _is_native(row):
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0].args == model_args
    else:
        assert assistant.content == _json_tool("ok", model_args)
        assert assistant.tool_calls is None


@pytest.mark.parametrize("kind", ["proceed", "raise_allow"])
@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_before_tool_nested_edits_never_reach_dispatch_events_or_the_model_turn(
    batch: bool, kind: str
) -> None:
    """``before_tool`` edits its own deep copy, nested values included: the tool, ``ActionEvent.args`` and the model's replayed turn keep the model's arguments (FR-003 review round 1).

    ``proceed`` returns ``None`` under the default ``DENY``; ``raise_allow``
    raises after editing, under ``ALLOW``, so the call falls open. Both paths
    dispatch the model's own args object, so a shallow copy would leak the
    nested edits into all three; the run must equal one without the hook.
    """

    def before_tool(_sid: object, _cid: object, _name: object, args: dict[str, Any]) -> None:
        _edit_at_every_depth(args)
        if kind == "raise_allow":
            raise RuntimeError("guard crashed after editing its copy")

    fallback = BeforeToolFallback.ALLOW if kind == "raise_allow" else BeforeToolFallback.DENY
    interventions = Interventions(before_tool=before_tool, before_tool_fallback=fallback)
    loop, llm, tools, names = _nested_args_loop(batch, interventions)
    events, hooked = await _drive(loop, llm)
    base_loop, base_llm, _tools, _names = _nested_args_loop(batch, None)
    _events, base = await _drive(base_loop, base_llm)

    assert hooked == base
    assert [tools[name].last_args for name in names] == [_NESTED_ARGS] * len(names)
    assert [event.args for event in _of(events, ActionEvent)] == [_NESTED_ARGS] * len(names)
    assistant = llm.calls[1].messages[2]
    assert assistant.tool_calls is not None
    assert [tc.args for tc in assistant.tool_calls] == [_NESTED_ARGS] * len(names)


async def test_before_tool_decisions_apply_per_call_in_a_batch() -> None:
    """In a batch, proceed, replace and deny each apply to their own call only (FR-003 AC-2, AC-7)."""
    tools = {
        "a": FakeTool("a", result=ToolResult(output="out-a")),
        "b": FakeTool("b", result=ToolResult(output="out-b")),
        "c": FakeTool("c", result=ToolResult(output="out-c")),
    }
    calls = [("a", {"q": "qa"}), ("b", {"q": "qb"}), ("c", {"q": "qc"})]
    llm = FakeLLMClient([make_multi_tool_response(calls), make_response("done")])
    hook_calls: list[tuple[str, str]] = []

    def before_tool(
        _sid: object, call_id: str, name: str, _args: object
    ) -> DenyToolCall | ReplaceToolArgs | None:
        hook_calls.append((call_id, name))
        if name == "b":
            return ReplaceToolArgs(args={"q": "qb-scoped"})
        if name == "c":
            return DenyToolCall(reason="c is off limits")
        return None

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(before_tool=before_tool),
        max_concurrent_tool_calls=3,
    )
    events = [event async for event in loop.run(list(_USER))]

    started = _of(events, ToolStartedEvent)
    assert [e.tool_name for e in started] == ["a", "b", "c"]
    assert hook_calls == [(e.call_id, e.tool_name) for e in started]
    assert [e.args for e in _of(events, ActionEvent)] == [
        {"q": "qa"},
        {"q": "qb-scoped"},
        {"q": "qc"},
    ]
    assert tools["a"].last_args == {"q": "qa"}
    assert tools["b"].last_args == {"q": "qb-scoped"}
    assert tools["c"].call_count == 0
    terminal = _of(events, ObservationEvent | ToolFailedEvent)
    assert [(type(e), e.call_id) for e in terminal] == [
        (ObservationEvent, started[0].call_id),
        (ObservationEvent, started[1].call_id),
        (ToolFailedEvent, started[2].call_id),
    ]
    assert terminal[2].error == "Tool call denied: c is off limits"
    tool_messages = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [(m.name, m.tool_call_id, m.content) for m in tool_messages] == [
        ("a", started[0].call_id, "out-a"),
        ("b", started[1].call_id, "out-b"),
        ("c", started[2].call_id, "Tool call denied: c is off limits"),
    ]
    assistant = llm.calls[1].messages[-4]
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == [e.call_id for e in started]
    assert [tc.args for tc in assistant.tool_calls] == [args for _name, args in calls]


@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_after_tool_receives_the_dispatched_args_for_every_decision(batch: bool) -> None:
    """``after_tool`` sees each call's DISPATCHED args: the replacement when ``before_tool`` replaced them, the model's when it proceeded (FR-003 D2, D5).

    ``a`` and ``d`` proceed, ``b`` and ``c`` are replaced, ``e`` is denied (so
    ``after_tool`` never runs for it). ``a``/``b`` return a success and
    ``c``/``d`` an ``is_error`` result, so all four ``after_tool`` call sites
    (single/batch x success/is_error) are covered, and each note, built from
    the args the hook saw, must reach its own call's message.
    """
    tools = {
        "a": FakeTool("a", result=ToolResult(output="out-a")),
        "b": FakeTool("b", result=ToolResult(output="out-b")),
        "c": FakeTool("c", result=ToolResult(is_error=True, error="err-c")),
        "d": FakeTool("d", result=ToolResult(is_error=True, error="err-d")),
        "e": FakeTool("e", result=ToolResult(output="out-e")),
    }
    names = ["a", "b", "c", "d", "e"]
    calls = [(name, {"q": f"q{name}"}) for name in names]
    if batch:
        replies = [make_multi_tool_response(calls), make_response("done")]
    else:
        replies = [make_multi_tool_response([call]) for call in calls] + [make_response("done")]
    llm = FakeLLMClient(replies)

    def before_tool(
        _sid: object, _cid: object, name: str, _args: object
    ) -> DenyToolCall | ReplaceToolArgs | None:
        if name in ("b", "c"):
            return ReplaceToolArgs(args={"q": f"q{name}-scoped"})
        if name == "e":
            return DenyToolCall(reason="e is off limits")
        return None

    seen: list[tuple[str, str, dict[str, Any]]] = []

    def after_tool(_sid: object, call_id: str, name: str, args: dict[str, Any], _r: object) -> str:
        seen.append((call_id, name, dict(args)))
        return f"note:{name}:{args['q']}"

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
        max_concurrent_tool_calls=5,
    )
    events = [event async for event in loop.run(list(_USER))]

    ids = {event.tool_name: event.call_id for event in _of(events, ToolStartedEvent)}
    dispatched = {"a": "qa", "b": "qb-scoped", "c": "qc-scoped", "d": "qd"}
    assert seen == [(ids[name], name, {"q": q}) for name, q in dispatched.items()]
    assert {name: tools[name].last_args for name in dispatched} == {
        name: {"q": q} for name, q in dispatched.items()
    }
    assert tools["e"].call_count == 0
    messages = {m.name: m.content for m in llm.calls[-1].messages if m.role == "tool"}
    for name, q in dispatched.items():
        assert messages[name].endswith(f"\n\nnote:{name}:{q}")
    assert messages["e"] == "Tool call denied: e is off limits"


def _failing_before_tool(kind: str) -> Callable[..., object]:
    """A ``before_tool`` that fails in one of the D5 failure classes."""

    def raising(_sid: object, _cid: object, _name: object, args: dict[str, Any]) -> None:
        args["q"] = "tampered"
        raise RuntimeError(f"guard failed on {_SECRET}")

    return {
        "raise": raising,
        "unrecognised": _constant({"q": "not-a-decision"}),
        "invalid_replace": _constant(ReplaceToolArgs.model_construct(args={1: "x"})),
    }[kind]


@pytest.mark.parametrize("kind", ["raise", "unrecognised", "invalid_replace"])
@pytest.mark.parametrize(
    "row", [pytest.param(_JSON, id="json"), pytest.param(_NATIVE, id="native")]
)
@pytest.mark.parametrize(
    "fallback",
    [
        pytest.param(BeforeToolFallback.DENY, id="deny"),
        pytest.param(BeforeToolFallback.ALLOW, id="allow"),
    ],
)
async def test_before_tool_failure_follows_the_configured_fallback(
    fallback: BeforeToolFallback, row: _Row, kind: str
) -> None:
    """A failed ``before_tool`` denies under ``DENY`` and runs the model's original args under ``ALLOW`` (FR-003 AC-3 analogue, single path)."""
    tools = _tools()
    model_args = {"q": "open"}
    llm = FakeLLMClient([_tool_turn(row, "ok", model_args), _final_turn(row)])
    after_calls: list[str] = []

    def after_tool(_sid: object, call_id: str, *_rest: object) -> None:
        after_calls.append(call_id)

    loop = _loop(
        llm,
        row,
        tools,
        interventions=Interventions(
            before_tool=_failing_before_tool(kind),
            after_tool=after_tool,
            before_tool_fallback=fallback,
        ),
    )
    with structlog.testing.capture_logs() as logs:
        events = [event async for event in loop.run(list(_USER))]

    assert isinstance(events[-1], FinalEvent)
    follow_up = [m.content for m in llm.calls[1].messages]
    warnings = _warnings(logs)
    assert len(warnings) == 1
    assert warnings[0]["hook_name"] == "before_tool"
    assert warnings[0]["call_id"] == _of(events, ToolStartedEvent)[0].call_id
    assert _SECRET not in str(warnings[0])
    assert _of(events, ActionEvent)[0].args == model_args
    if fallback is BeforeToolFallback.DENY:
        assert tools["ok"].call_count == 0
        assert [e.error for e in _of(events, ToolFailedEvent)] == [_SDK_DENIAL]
        assert any(_SDK_DENIAL in content for content in follow_up)
        assert after_calls == []
        assert warnings[0]["fallback"] == "call_denied"
    else:
        assert tools["ok"].call_count == 1
        assert tools["ok"].last_args == model_args
        assert len(_of(events, ObservationEvent)) == 1
        assert not any("Tool call denied" in content for content in follow_up)
        assert any('{"rows": [{"id": 1}, {"id": 2}]}' in content for content in follow_up)
        assert after_calls == [_of(events, ObservationEvent)[0].call_id]
        assert warnings[0]["fallback"] == "call_allowed"


@pytest.mark.parametrize("batch", [False, True], ids=["single_path", "native_batch"])
@pytest.mark.parametrize(
    "fallback",
    [
        pytest.param(BeforeToolFallback.DENY, id="deny"),
        pytest.param(BeforeToolFallback.ALLOW, id="allow"),
    ],
)
async def test_before_tool_cancelled_error_propagates(
    fallback: BeforeToolFallback, batch: bool
) -> None:
    """``CancelledError`` from ``before_tool`` propagates under both fallbacks; no tool runs (FR-003 D4)."""

    async def before_tool(*_args: object) -> None:
        raise asyncio.CancelledError

    tools = _tools()
    if batch:
        row = _NATIVE
        replies = [make_multi_tool_response([("ok", {}), ("bad", {})]), make_response("done")]
    else:
        row = _JSON
        replies = [make_response(_json_tool("ok", {})), make_response(_json_final("done"))]
    llm = FakeLLMClient(replies)
    loop = _loop(
        llm,
        row,
        tools,
        interventions=Interventions(before_tool=before_tool, before_tool_fallback=fallback),
        max_concurrent_tool_calls=2,
    )

    events: list[AgentEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in loop.run(list(_USER)):
            events.append(event)

    assert all(tool.call_count == 0 for tool in tools.values())
    assert not _of(events, ActionEvent)
    assert not _of(events, FinalEvent)


async def test_before_tool_fires_for_unregistered_tool_names() -> None:
    """``before_tool`` sees a hallucinated name; proceeding still yields the ToolNotFound observation (FR-003 D5).

    Only the registry's ToolNotFound path produces this text, so a deny branch
    taken for a proceeding hook, or a hook skipped for unknown names, fails
    here.
    """
    llm = FakeLLMClient(
        [make_response(_json_tool("ghost", {"q": "x"})), make_response(_json_final("done"))]
    )
    seen: list[str] = []

    def before_tool(_sid: object, _cid: object, name: str, _args: object) -> None:
        seen.append(name)

    loop = _loop(llm, _JSON, _tools(), interventions=Interventions(before_tool=before_tool))
    events = [event async for event in loop.run(list(_USER))]

    assert seen == ["ghost"]
    assert [e.error for e in _of(events, ToolFailedEvent)] == [
        "ToolNotFound: Tool 'ghost' is not registered"
    ]
    assert llm.calls[1].messages[-1].content == (
        "Tool ghost failed: ToolNotFound: tool 'ghost' is not registered."
    )


async def test_denied_call_counts_as_tool_invoked_for_require_tool_before_final() -> None:
    """A denied call satisfies BR-036's guard, as a ToolNotFound call does: no re-ask (FR-003 D5, R12)."""
    llm = FakeLLMClient(
        [make_response(_json_tool("ok", {})), make_response(_json_final("answered"))]
    )
    loop = _loop(
        llm,
        _JSON,
        _tools(),
        interventions=Interventions(before_tool=_constant(DenyToolCall(reason="no"))),
        require_tool_before_final=True,
    )
    events = [event async for event in loop.run(list(_USER))]

    assert len(llm.calls) == 2
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.text == "answered"


# --- AC-4 companion and construction ------------------------------------------------------


_PASSIVE = [
    pytest.param(Interventions, id="empty"),
    pytest.param(lambda: Interventions(after_tool=_constant(None)), id="after_none"),
    pytest.param(lambda: Interventions(after_tool=_constant("   ")), id="after_blank"),
    pytest.param(lambda: Interventions(before_tool=_constant(None)), id="before_none"),
    pytest.param(
        lambda: Interventions(before_tool=_constant(None), after_tool=_constant(None)),
        id="both_none",
    ),
    pytest.param(
        lambda: Interventions(before_tool_fallback=BeforeToolFallback.ALLOW),
        id="allow_without_hook",
    ),
    pytest.param(
        lambda: Interventions(before_tool_fallback="allow"),  # type: ignore[arg-type]
        id="allow_string_without_hook",
    ),
]


@pytest.mark.parametrize("make", _PASSIVE)
@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_LEGACY, id="legacy"),
        pytest.param(_JSON, id="json"),
        pytest.param(_PROSE, id="prose"),
        pytest.param(_NATIVE, id="native"),
    ],
)
async def test_passive_interventions_leave_requests_and_events_unchanged(
    row: _Row, make: Callable[[], Interventions]
) -> None:
    """Configured-but-passive interventions send the same bodies, events and tool args as ``interventions=None`` (FR-003 AC-4).

    ``before_tool_fallback`` without ``before_tool`` is inert, not an error.
    Covers ok, ``is_error`` and ToolNotFound outcomes, and a native batch.

    Negative space: both runs go through the same build, so this cannot see a
    change that moves them both (mutation probe M16, a separator appended even
    without a note, stays green here). The absolute no-intervention content is
    pinned by ``test_golden_1_9_0`` and the 1.7.0/1.8.0 goldens.
    """

    async def run(interventions: Interventions | None) -> tuple[Scenario, dict[str, Any]]:
        tools = _tools()
        if _is_native(row):
            replies = [
                make_multi_tool_response(
                    [("ok", {"q": "a"}), ("bad", {"q": "b"}), ("missing", {})]
                ),
                make_multi_tool_response([("ok", {"q": "c"})]),
                _final_turn(row),
            ]
        else:
            replies = [
                _tool_turn(row, "ok", {"q": "a"}),
                _tool_turn(row, "bad", {"q": "b"}),
                _tool_turn(row, "missing", {}),
                _final_turn(row),
            ]
        llm = FakeLLMClient(replies)
        loop = _loop(llm, row, tools, interventions=interventions, max_concurrent_tool_calls=3)
        _events, scenario = await _drive(loop, llm)
        return scenario, {name: tool.last_args for name, tool in tools.items()}

    base, base_args = await run(None)
    passive, passive_args = await run(make())

    assert passive == base
    assert passive_args == base_args


@pytest.mark.parametrize("value", [Hooks(), {"after_tool": print}], ids=["hooks", "dict"])
def test_agent_loop_rejects_a_non_interventions_value(value: object) -> None:
    """A ``Hooks`` or a ``dict`` passed as ``interventions`` raises ``TypeError`` at construction (FR-003 D1)."""
    with pytest.raises(TypeError, match="interventions"):
        AgentLoop(
            llm=FakeLLMClient([]),
            registry=Registry(),
            prompts=PromptSections(persona="p"),
            safety=SafetyConfig(),
            model=_MODEL,
            tool_mode=ToolMode.JSON,
            interventions=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "interventions",
    [pytest.param(None, id="no_interventions"), pytest.param(Interventions(), id="passive")],
)
@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_no_hook_runs_make_no_deep_copy(
    batch: bool, interventions: Interventions | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no hook set a run deep-copies nothing; the hook argument copies exist only when a hook does (FR-003 review round 1, m3).

    The spy replaces the ``copy.deepcopy`` attribute, so it records (and still
    performs) every call that looks the attribute up at call time. A deep copy
    made through a name bound at import time (``from copy import deepcopy``)
    escapes it, so this test does not pin that (review round 2, N2). This pins
    the deep-copy half of "no extra argument copy". An extra SHALLOW copy of
    the args is not observable from outside the loop, because every consumer
    already receives its own copy, so it is not pinned.
    """
    loop, _llm, tools, names = _nested_args_loop(batch, interventions)
    real_deepcopy = copy.deepcopy
    deep_copied: list[object] = []

    def recording_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        deep_copied.append(value)
        return real_deepcopy(value, memo)

    monkeypatch.setattr(copy, "deepcopy", recording_deepcopy)
    events = [event async for event in loop.run(list(_USER))]
    monkeypatch.undo()

    assert deep_copied == []
    assert isinstance(events[-1], FinalEvent)
    assert [tools[name].last_args for name in names] == [_NESTED_ARGS] * len(names)


def _two_call_loop(batch: bool, interventions: Interventions | None) -> AgentLoop:
    """A NATIVE loop whose model calls ``ok`` then ``ok2``, as one batch or as two single calls."""
    tools = {
        name: FakeTool(name, result=ToolResult(output=f"out-{name}")) for name in ("ok", "ok2")
    }
    calls = [("ok", {"q": "a"}), ("ok2", {"q": "b"})]
    if batch:
        replies = [make_multi_tool_response(calls), make_response("done")]
    else:
        replies = [make_multi_tool_response([call]) for call in calls] + [make_response("done")]
    return _loop(
        FakeLLMClient(replies),
        _NATIVE,
        tools,
        interventions=interventions,
        max_concurrent_tool_calls=2,
    )


@pytest.mark.parametrize("deny_second", [False, True], ids=["no_hook", "second_denied"])
@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_dispatch_logs_tool_invoked_per_call_and_tool_denied_for_a_denied_call(
    batch: bool, deny_second: bool
) -> None:
    """Each call logs one DEBUG dispatch line: ``tool_invoked`` (name, call_id, run_id), as 1.9.0 did, or ``tool_denied`` (call_id, run_id, no tool name) when ``before_tool`` denied it (FR-003 D4, D5)."""

    def before_tool(_sid: object, _cid: object, name: str, _args: object) -> DenyToolCall | None:
        return DenyToolCall(reason="not this one") if name == "ok2" else None

    interventions = Interventions(before_tool=before_tool) if deny_second else None
    loop = _two_call_loop(batch, interventions)
    with structlog.testing.capture_logs() as logs:
        events = [event async for event in loop.run(list(_USER))]

    ids = {event.tool_name: event.call_id for event in _of(events, ToolStartedEvent)}
    run_id = next(entry["run_id"] for entry in logs if entry["event"] == "agent_loop_started")
    dispatch_lines = [entry for entry in logs if entry["event"] in ("tool_invoked", "tool_denied")]
    first = {
        "event": "tool_invoked",
        "log_level": "debug",
        "name": "ok",
        "call_id": ids["ok"],
        "run_id": run_id,
    }
    if deny_second:
        second = {
            "event": "tool_denied",
            "log_level": "debug",
            "call_id": ids["ok2"],
            "run_id": run_id,
        }
    else:
        second = {**first, "name": "ok2", "call_id": ids["ok2"]}
    assert dispatch_lines == [first, second]


# --- before_tool: fallbacks per call in a batch ---------------------------------------------


@pytest.mark.parametrize(
    "fallback",
    [
        pytest.param(BeforeToolFallback.DENY, id="deny"),
        pytest.param(BeforeToolFallback.ALLOW, id="allow"),
    ],
)
async def test_before_tool_batch_failures_follow_the_fallback_per_call(
    fallback: BeforeToolFallback,
) -> None:
    """Each batch member's failure follows the fallback on its own; a returned deny is honoured under both (FR-003 AC-2, AC-3, AC-7).

    ``a`` raises after mutating its copy, ``b`` returns an unrecognised value,
    ``c`` an invalid replacement, ``d`` a deny with an unusable reason, and
    ``e`` proceeds.
    """
    names = ["a", "b", "c", "d", "e"]
    tools = {name: FakeTool(name, result=ToolResult(output=f"out-{name}")) for name in names}
    calls = [(name, {"q": f"q{name}"}) for name in names]
    llm = FakeLLMClient([make_multi_tool_response(calls), make_response("done")])
    hook_calls: list[tuple[str, str]] = []

    def before_tool(_sid: object, call_id: str, name: str, args: dict[str, Any]) -> object:
        hook_calls.append((call_id, name))
        if name == "a":
            args["q"] = "tampered"
            raise RuntimeError(f"guard failed on {_SECRET}")
        if name == "b":
            return "deny"
        if name == "c":
            return ReplaceToolArgs.model_construct(args={1: "x"})
        if name == "d":
            return DenyToolCall.model_construct(reason="  ")
        return None

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(before_tool=before_tool, before_tool_fallback=fallback),
        max_concurrent_tool_calls=5,
    )
    with structlog.testing.capture_logs() as logs:
        events = [event async for event in loop.run(list(_USER))]

    started = _of(events, ToolStartedEvent)
    ids = {e.tool_name: e.call_id for e in started}
    assert hook_calls == [(ids[name], name) for name in names]
    denied = {"a", "b", "c", "d"} if fallback is BeforeToolFallback.DENY else {"d"}
    for name in names:
        if name in denied:
            assert tools[name].call_count == 0
        else:
            assert tools[name].call_count == 1
            assert tools[name].last_args == {"q": f"q{name}"}
    terminal = {e.call_id: e for e in _of(events, ObservationEvent | ToolFailedEvent)}
    for name in names:
        event = terminal[ids[name]]
        if name in denied:
            assert isinstance(event, ToolFailedEvent)
            assert event.error == _SDK_DENIAL
        else:
            assert isinstance(event, ObservationEvent)
    tool_messages = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [m.name for m in tool_messages] == names
    assert [m.tool_call_id for m in tool_messages] == [ids[name] for name in names]
    assistant = llm.calls[1].messages[-6]
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == [ids[name] for name in names]
    warnings = _warnings(logs)
    assert [w["call_id"] for w in warnings] == [ids[name] for name in "abcd"]
    expected_fallback = {
        name: "call_denied" if name in denied else "call_allowed" for name in "abcd"
    }
    assert [w["fallback"] for w in warnings] == [expected_fallback[name] for name in "abcd"]
    assert [w.get("reason") for w in warnings] == [
        None,
        "unrecognised_decision",
        "invalid_args",
        "invalid_reason",
    ]
    assert _SECRET not in str(warnings)
    assert isinstance(events[-1], FinalEvent)


@pytest.mark.parametrize("batch", [False, True], ids=["native_single", "native_batch"])
async def test_uncopyable_args_take_the_one_level_fallback_in_the_loop(batch: bool) -> None:
    """Args ``copy.deepcopy`` rejects never deny a call under the default ``DENY``: every call runs, each hook runs on a one-level copy, one ``args_not_copyable`` WARNING per hook call, no ``hook_failed``, and the notes arrive (FR-003 review round 1).

    Complements the unit test of ``_hook_args``: this pins that the loop never
    turns a copy problem into a hook failure (which ``DENY`` would make a
    denial).
    """
    handle = _Uncopyable()
    names = ["a", "b"] if batch else ["a"]
    tools = {name: FakeTool(name, result=ToolResult(output=f"out-{name}")) for name in names}
    calls = [(name, {"q": f"q{name}", "handle": handle}) for name in names]
    llm = FakeLLMClient([make_multi_tool_response(calls), make_response("done")])
    before_seen: list[dict[str, Any]] = []

    def before_tool(_sid: object, _cid: object, _name: object, args: dict[str, Any]) -> None:
        before_seen.append(args)
        args["q"] = "edited-by-hook"

    def after_tool(_sid: object, _cid: object, name: str, *_rest: object) -> str:
        return f"note:{name}"

    loop = _loop(
        llm,
        _NATIVE,
        tools,
        interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
        max_concurrent_tool_calls=2,
    )
    with structlog.testing.capture_logs() as logs:
        events = [event async for event in loop.run(list(_USER))]

    ids = {event.tool_name: event.call_id for event in _of(events, ToolStartedEvent)}
    assert not _of(events, ToolFailedEvent)
    assert len(_of(events, ObservationEvent)) == len(names)
    assert [tools[name].call_count for name in names] == [1] * len(names)
    assert [tools[name].last_args["q"] for name in names] == [f"q{name}" for name in names]
    assert all(tools[name].last_args["handle"] is handle for name in names)
    assert len(before_seen) == len(names)
    assert all(args["handle"] is handle for args in before_seen)
    assert [
        (w["event"], w["hook_name"], w["call_id"], w["error_type"]) for w in _warnings(logs)
    ] == [
        ("intervention.args_not_copyable", hook_name, ids[name], "TypeError")
        for hook_name in ("before_tool", "after_tool")
        for name in names
    ]
    tool_messages = [m.content for m in llm.calls[-1].messages if m.role == "tool"]
    assert tool_messages == [f"out-{name}\n\nnote:{name}" for name in names]
