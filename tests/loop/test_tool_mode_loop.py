"""Wire-level tests for ``AgentLoop(tool_mode=...)`` (FR-001).

Each test drives a scripted run and asserts on what reaches the provider: the
request body (serialised through the real OpenAI wire translator), the system
prompt, parse routing, and the replayed message list. Construction-time rules
(the conflict matrix, defaults) live in ``tests/test_tool_mode.py``; the
legacy request-identity promise lives in ``tests/loop/test_legacy_golden.py``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from structlog.testing import capture_logs

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    ActionEvent,
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    JsonModeParser,
    PromptSections,
    Registry,
    SafetyConfig,
    ThoughtEvent,
    TokenEvent,
    ToolMode,
    render_system_prompt,
)
from fifty_agent_sdk.tool_mode import (
    _NATIVE_PARSER_RETRY_REMINDER,
    _PROSE_PARSER_RETRY_REMINDER,
)
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)
from tests.loop.golden_capture import _normalise_ids, wire_bodies

_PERSONA = "You are a test agent."
_USER = [ChatMessage(role="user", content="What is the answer?")]
_TOOL_BLOCK = "- search: Test fake tool: search\n  args: {}"

_JSON_TOOL_ENVELOPE = json.dumps(
    {
        "thought": "I should search",
        "action": "tool",
        "tool_name": "search",
        "tool_args": {"query": "x"},
        "answer": None,
    }
)
_PROSE_ACTION_BLOCK = 'Thought: I should search\nAction: search\nAction Input: {"query": "x"}'


# --- Helpers ----------------------------------------------------------------


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


def _registry(*tools: FakeTool) -> Registry:
    registry = Registry()
    for fake in tools or (FakeTool("search"),):
        registry.register(fake)
    return registry


def _loop(
    llm: FakeLLMClient,
    tool_mode: ToolMode | None,
    *,
    registry: Registry | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    **extra: Any,
) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _registry(),
        prompts=PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
        stream=stream,
        tool_mode=tool_mode,
        **extra,
    )


async def _collect(iterator: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in iterator]


def _unpaired_tool_replies(body: dict[str, Any]) -> list[int]:
    """Indices of ``role="tool"`` entries NOT paired to the nearest preceding assistant turn.

    A reply is paired when the nearest preceding assistant entry carries
    ``tool_calls`` and one of their ids equals the reply's ``tool_call_id``.
    A ``role="tool"`` entry after an assistant entry with no ``tool_calls`` is
    unpaired — the shape strict endpoints reject with HTTP 400.
    """
    unpaired: list[int] = []
    last_assistant: dict[str, Any] | None = None
    for index, message in enumerate(body["messages"]):
        if message["role"] == "assistant":
            last_assistant = message
        elif message["role"] == "tool":
            ids = [tc["id"] for tc in (last_assistant or {}).get("tool_calls") or []]
            if message.get("tool_call_id") not in ids:
                unpaired.append(index)
    return unpaired


def _assert_tool_replies_paired(body: dict[str, Any]) -> None:
    assert _unpaired_tool_replies(body) == [], body["messages"]


# --- AC-1: per-mode request body and system prompt ----------------------------


async def test_json_mode_request_and_prompt() -> None:
    """JSON: prompt tool block + JSON format, no ``tools``, results as role="assistant" (FR-001 AC-1)."""
    tool = FakeTool("search")
    llm = FakeLLMClient(
        [make_response(_JSON_TOOL_ENVELOPE), make_response(_json_final("forty-two"))]
    )
    loop = _loop(llm, ToolMode.JSON, registry=_registry(tool))

    events = await _collect(loop.run(_USER))

    assert loop._system_prompt == render_system_prompt(
        PromptSections(
            persona=_PERSONA, tool_descriptions=_TOOL_BLOCK, output_format=JSON_MODE_OUTPUT_FORMAT
        )
    )
    assert all(req.tools is None and req.tool_choice is None for req in llm.calls)
    bodies = await wire_bodies(llm, stream=False)
    assert all("tools" not in body and "tool_choice" not in body for body in bodies)
    assert bodies[1]["messages"][-1] == {"role": "assistant", "content": "Tool search returned: ok"}
    assert tool.call_count == 1
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "forty-two"


async def test_json_mode_equals_legacy_json_assistant_shape() -> None:
    """tool_mode=JSON sends request bodies equal to the legacy JSON + assistant-role shape (FR-001 AC-1)."""

    def script() -> FakeLLMClient:
        return FakeLLMClient(
            [
                make_response(_JSON_TOOL_ENVELOPE),
                make_response("- drift outside the envelope"),
                make_response(_json_final("same")),
            ]
        )

    explicit_llm = script()
    await _collect(_loop(explicit_llm, ToolMode.JSON).run(_USER))
    legacy_llm = script()
    legacy = AgentLoop(
        llm=legacy_llm,
        registry=_registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=_PERSONA),
        safety=SafetyConfig(),
        model="test-model",
        output_format=JSON_MODE_OUTPUT_FORMAT,
        tool_message_role="assistant",
    )
    await _collect(legacy.run(_USER))

    explicit = _normalise_ids(await wire_bodies(explicit_llm, stream=False))
    assert explicit == _normalise_ids(await wire_bodies(legacy_llm, stream=False))
    assert len(explicit) == 3


async def test_prose_mode_request_and_prompt() -> None:
    """PROSE: prompt tool block + prose format, prose routing, role="assistant" results (FR-001 AC-1)."""
    tool = FakeTool("search")
    llm = FakeLLMClient(
        [make_response(_PROSE_ACTION_BLOCK), make_response("Thought: done\nFinal Answer: prose")]
    )
    loop = _loop(llm, ToolMode.PROSE, registry=_registry(tool))

    events = await _collect(loop.run(_USER))

    assert loop._system_prompt == render_system_prompt(
        PromptSections(
            persona=_PERSONA, tool_descriptions=_TOOL_BLOCK, output_format=PROSE_MODE_OUTPUT_FORMAT
        )
    )
    bodies = await wire_bodies(llm, stream=False)
    assert all("tools" not in body for body in bodies)
    assert bodies[1]["messages"][-1] == {"role": "assistant", "content": "Tool search returned: ok"}
    assert tool.last_args == {"query": "x"}
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "prose"


async def test_native_mode_request_and_prompt() -> None:
    """NATIVE: no prompt tool/format sections, ``tools`` + auto, id-paired role="tool" reply (FR-001 AC-1)."""
    tool = FakeTool("search")
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "x"})]),
            make_response("The answer is 42."),
        ]
    )
    loop = _loop(llm, ToolMode.NATIVE, registry=_registry(tool))

    events = await _collect(loop.run(_USER))

    assert loop._system_prompt == f"# Persona\n{_PERSONA}"
    assert "# Tools" not in loop._system_prompt
    assert "# Output Format" not in loop._system_prompt
    bodies = await wire_bodies(llm, stream=False)
    for body in bodies:
        assert body["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in body["tools"]] == ["search"]
        _assert_tool_replies_paired(body)
    assistant, reply = bodies[1]["messages"][-2:]
    assert assistant["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": '{"query": "x"}',
    }
    assert reply["role"] == "tool"
    assert reply["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert tool.call_count == 1
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "The answer is 42."


async def test_native_mode_multi_call_replies_are_paired() -> None:
    """NATIVE multi-call: N distinct ids on the assistant turn, each reply paired (FR-001 AC-1, L-934)."""
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "a"}), ("lookup", {"id": 1})]),
            make_response("done"),
        ]
    )
    loop = _loop(
        llm,
        ToolMode.NATIVE,
        registry=_registry(FakeTool("search"), FakeTool("lookup")),
        safety=SafetyConfig(max_concurrent_tool_calls=2),
    )

    await _collect(loop.run(_USER))

    body = (await wire_bodies(llm, stream=False))[1]
    _assert_tool_replies_paired(body)
    ids = [tc["id"] for tc in body["messages"][-3]["tool_calls"]]
    assert len(set(ids)) == 2
    assert [m["tool_call_id"] for m in body["messages"][-2:]] == ids


@pytest.mark.parametrize("mode", [ToolMode.JSON, ToolMode.PROSE])
async def test_text_mode_user_role_override_is_honoured(mode: ToolMode) -> None:
    """JSON and PROSE send the tool result as role="user" when asked to (FR-001 AC-1)."""
    call, final = (
        (_JSON_TOOL_ENVELOPE, _json_final("ok"))
        if mode is ToolMode.JSON
        else (_PROSE_ACTION_BLOCK, "Thought: done\nFinal Answer: ok")
    )
    llm = FakeLLMClient([make_response(call), make_response(final)])

    await _collect(_loop(llm, mode, tool_message_role="user").run(_USER))

    last = (await wire_bodies(llm, stream=False))[1]["messages"][-1]
    assert last == {"role": "user", "content": "Tool search returned: ok"}


# --- AC-2: NATIVE never dispatches a text tool call ---------------------------


@pytest.mark.parametrize(
    "text",
    [_JSON_TOOL_ENVELOPE, _PROSE_ACTION_BLOCK],
    ids=["json_envelope", "prose_action_block"],
)
async def test_native_mode_text_tool_call_is_final(text: str) -> None:
    """NATIVE: text shaped like a tool call is the final answer, verbatim, and nothing runs (FR-001 AC-2)."""
    tool = FakeTool("search")
    llm = FakeLLMClient([make_response(text)])
    loop = _loop(llm, ToolMode.NATIVE, registry=_registry(tool))

    events = await _collect(loop.run(_USER))

    assert [type(e) for e in events] == [ThoughtEvent, FinalEvent]
    assert not any(isinstance(e, ActionEvent) for e in events)
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.text == text
    assert final.raw_completion == text
    assert tool.call_count == 0
    assert len(llm.calls) == 1


async def test_native_mode_never_sends_unpaired_tool_role() -> None:
    """NATIVE: after a native call, a text tool envelope ends the run; every body stays paired (FR-001 AC-2)."""
    tool = FakeTool("search")
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "first"})]),
            make_response(_JSON_TOOL_ENVELOPE),
        ]
    )
    loop = _loop(llm, ToolMode.NATIVE, registry=_registry(tool))

    events = await _collect(loop.run(_USER))

    bodies = await wire_bodies(llm, stream=False)
    assert len(bodies) == 2
    for body in bodies:
        _assert_tool_replies_paired(body)
    assert tool.call_count == 1
    assert isinstance(events[-1], FinalEvent) and events[-1].text == _JSON_TOOL_ENVELOPE


async def test_legacy_native_flag_still_dispatches_text_call() -> None:
    """Negative control: the legacy flag DOES dispatch the text call and sends an unpaired reply (FR-001 AC-2).

    Same script as ``test_native_mode_text_tool_call_is_final[json_envelope]``,
    so that test's zero-dispatch result is caused by ``ToolMode.NATIVE``, not by
    the script. It also proves ``_unpaired_tool_replies`` detects the 1.7.0
    half-native shape. This path is intentionally preserved (FR-001 D2).
    """
    tool = FakeTool("search")
    llm = FakeLLMClient([make_response(_JSON_TOOL_ENVELOPE), make_response(_json_final("done"))])
    loop = _loop(
        llm,
        None,
        registry=_registry(tool),
        parser=JsonModeParser(),
        output_format=JSON_MODE_OUTPUT_FORMAT,
        safety=SafetyConfig(native_tools_enabled=True),
    )

    await _collect(loop.run(_USER))

    assert tool.call_count == 1
    second = (await wire_bodies(llm, stream=False))[1]
    assert _unpaired_tool_replies(second) == [len(second["messages"]) - 1]


# --- AC-4: stream -------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "final"),
    [
        (ToolMode.JSON, _json_final("streamed")),
        (ToolMode.PROSE, "Thought: done\nFinal Answer: streamed"),
    ],
)
async def test_text_modes_allow_stream(mode: ToolMode, final: str) -> None:
    """JSON and PROSE construct with stream=True and emit TokenEvents for the final (FR-001 AC-4)."""
    llm = FakeLLMClient([make_stream_chunks([final[:8], final[8:]])])
    loop = _loop(llm, mode, stream=True)

    events = await _collect(loop.run(_USER))

    assert [e.text for e in events if isinstance(e, TokenEvent)] == [final[:8], final[8:]]
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "streamed"


# --- AC-5: native thought -------------------------------------------------------


async def test_native_turn_thought_event_carries_content() -> None:
    """A native turn's content becomes its ThoughtEvent; the replayed content stays verbatim (FR-001 AC-5)."""
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "x"})], content="  let me check\n"),
            make_response("done"),
        ]
    )
    loop = _loop(llm, ToolMode.NATIVE)

    events = await _collect(loop.run(_USER))

    assert isinstance(events[0], ThoughtEvent)
    assert events[0].text == "let me check"
    assistant = (await wire_bodies(llm, stream=False))[1]["messages"][-2]
    assert assistant["content"] == "  let me check\n"


# --- AC-6 + D6: retry reminders and the empty native final ---------------------


@pytest.mark.parametrize(
    ("mode", "bad", "good", "reminder"),
    [
        (
            ToolMode.PROSE,
            "garbage",
            "Thought: ok\nFinal Answer: fixed",
            _PROSE_PARSER_RETRY_REMINDER,
        ),
        (ToolMode.NATIVE, "", "fixed", _NATIVE_PARSER_RETRY_REMINDER),
    ],
)
async def test_retry_reminder_is_mode_appropriate(
    mode: ToolMode, bad: str, good: str, reminder: str
) -> None:
    """A PROSE or NATIVE parse failure re-prompts with a reminder that never mentions JSON (FR-001 AC-6)."""
    llm = FakeLLMClient([make_response(bad), make_response(good)])

    events = await _collect(_loop(llm, mode).run(_USER))

    retry_last = llm.calls[1].messages[-1]
    assert retry_last == ChatMessage(role="user", content=reminder)
    assert "JSON" not in retry_last.content
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "fixed"


async def test_json_mode_keeps_default_reminder() -> None:
    """JSON keeps the SafetyConfig JSON-envelope reminder, echoing the drift first (FR-001 AC-6)."""
    llm = FakeLLMClient([make_response("- drift"), make_response(_json_final("fixed"))])

    await _collect(_loop(llm, ToolMode.JSON).run(_USER))

    assert llm.calls[1].messages[-2:] == [
        ChatMessage(role="assistant", content="- drift"),
        ChatMessage(role="user", content=SafetyConfig().parser_retry_reminder),
    ]


@pytest.mark.parametrize(
    ("mode", "bad", "good"),
    [
        (ToolMode.JSON, "- drift", _json_final("fixed")),
        (ToolMode.PROSE, "garbage", "Thought: ok\nFinal Answer: fixed"),
        (ToolMode.NATIVE, "  ", "fixed"),
    ],
)
async def test_explicit_reminder_wins_in_every_mode(mode: ToolMode, bad: str, good: str) -> None:
    """A consumer-set reminder reaches the wire verbatim in every mode (FR-001 D11)."""
    llm = FakeLLMClient([make_response(bad), make_response(good)])
    safety = SafetyConfig(parser_retry_reminder="Custom reminder.")

    await _collect(_loop(llm, mode, safety=safety).run(_USER))

    assert llm.calls[1].messages[-1] == ChatMessage(role="user", content="Custom reminder.")


async def test_native_empty_final_retries_then_recovers() -> None:
    """An empty NATIVE completion retries once, without an empty assistant echo (FR-001 D6)."""
    llm = FakeLLMClient([make_response(""), make_response("answer")])

    events = await _collect(_loop(llm, ToolMode.NATIVE).run(_USER))

    assert len(llm.calls) == 2
    assert [m.role for m in llm.calls[1].messages] == ["system", "user", "user"]
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "answer"


async def test_explicit_json_blank_completion_is_not_echoed() -> None:
    """Under any explicit mode a blank completion is not echoed on retry; legacy still echoes (FR-001 D6).

    Only explicit JSON runs here; the legacy echo is pinned by golden ``json_parser_retry_blank``.
    """
    llm = FakeLLMClient([make_response("   "), make_response(_json_final("fixed"))])

    await _collect(_loop(llm, ToolMode.JSON).run(_USER))

    assert [m.role for m in llm.calls[1].messages] == ["system", "user", "user"]


@pytest.mark.parametrize(
    ("retry_enabled", "replies"),
    [(False, [""]), (True, ["", " "])],
    ids=["retry_disabled", "retry_exhausted"],
)
async def test_native_empty_final_terminates_with_parser_error(
    retry_enabled: bool, replies: list[str]
) -> None:
    """With no retry left, an empty NATIVE final ends with ParserError + fallback final (FR-001 D6)."""
    llm = FakeLLMClient([make_response(r) for r in replies])
    safety = SafetyConfig(parser_retry_enabled=retry_enabled)

    events = await _collect(_loop(llm, ToolMode.NATIVE, safety=safety).run(_USER))

    assert [type(e) for e in events] == [ErrorEvent, FinalEvent]
    error = events[0]
    assert isinstance(error, ErrorEvent)
    assert error.error_type == "ParserError"
    assert error.context["parser"] == "FinalOnlyParser"
    assert error.context["error_phase"] == "empty_completion"
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].text == safety.fallback_message
    assert len(llm.calls) == len(replies)


# --- D9: explicit text modes ignore unrequested native tool_calls ---------------


@pytest.mark.parametrize(
    ("mode", "content", "answer"),
    [
        (ToolMode.JSON, _json_final("text wins"), "text wins"),
        (ToolMode.PROSE, "Thought: done\nFinal Answer: text wins", "text wins"),
    ],
)
async def test_explicit_text_mode_ignores_native_tool_calls(
    mode: ToolMode, content: str, answer: str
) -> None:
    """Explicit JSON/PROSE parse the text and warn with a count only, never content (FR-001 D9)."""
    tool = FakeTool("search")
    llm = FakeLLMClient([make_multi_tool_response([("search", {"query": "x"})], content=content)])

    with capture_logs() as logs:
        events = await _collect(_loop(llm, mode, registry=_registry(tool)).run(_USER))

    assert tool.call_count == 0
    assert isinstance(events[-1], FinalEvent) and events[-1].text == answer
    warnings = [e for e in logs if e["event"] == "native_tool_calls_ignored"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["count"] == 1
    assert set(warnings[0]) == {"event", "log_level", "count", "run_id"}


async def test_legacy_native_precedence_is_unconditional() -> None:
    """Legacy (no tool_mode, flag off) still dispatches unrequested tool_calls natively, as 1.7.0 did (FR-001 D2)."""
    tool = FakeTool("search")
    llm = FakeLLMClient(
        [make_multi_tool_response([("search", {"query": "x"})]), make_response(_json_final("ok"))]
    )
    loop = _loop(llm, None, registry=_registry(tool), parser=JsonModeParser())

    with capture_logs() as logs:
        await _collect(loop.run(_USER))

    assert tool.call_count == 1
    assert not [e for e in logs if e["event"] == "native_tool_calls_ignored"]


# --- D10: NATIVE with an empty registry -----------------------------------------


async def test_native_mode_empty_registry_omits_tools() -> None:
    """NATIVE with no tools sends neither ``tools`` nor ``tool_choice`` (FR-001 D10)."""
    llm = FakeLLMClient([make_response("no tools needed")])

    await _collect(_loop(llm, ToolMode.NATIVE, registry=Registry()).run(_USER))

    assert llm.calls[0].tools is None and llm.calls[0].tool_choice is None
    body = (await wire_bodies(llm, stream=False))[0]
    assert "tools" not in body and "tool_choice" not in body


class _RegistersOnFirstCallLLM(FakeLLMClient):
    """Registers ``tool`` into ``registry`` while serving the first request."""

    def __init__(
        self,
        replies: list[ChatResponse | list[ChatResponse] | Exception],
        *,
        registry: Registry,
        tool: FakeTool,
    ) -> None:
        super().__init__(replies)
        self._registry = registry
        self._tool = tool

    async def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.calls:
            self._registry.register(self._tool)
        return await super().complete(request)


async def test_native_mode_late_registration_declared() -> None:
    """A tool registered mid-run appears in the next NATIVE request (FR-001 D10)."""
    registry = Registry()
    late = FakeTool("late")
    llm = _RegistersOnFirstCallLLM(
        [make_multi_tool_response([("late", {})]), make_response("done")],
        registry=registry,
        tool=late,
    )

    await _collect(_loop(llm, ToolMode.NATIVE, registry=registry).run(_USER))

    assert llm.calls[0].tools is None
    assert [t["function"]["name"] for t in llm.calls[1].tools or []] == ["late"]
    assert late.call_count == 1
