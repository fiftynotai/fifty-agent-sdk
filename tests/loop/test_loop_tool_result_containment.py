"""Tool results that used to crash a run with a raw exception (BR-022).

Two shapes left ``AgentLoop.run()`` through an untyped exception, with no
``ErrorEvent`` or ``FinalEvent`` (BR-022 evidence, P1-P3, measured on CPython
3.11.15, 3.13.2 and 3.14.3):

* A surrogate code point (U+D800-U+DFFF) in tool-result message text. With
  the shipped ``OpenAICompatibleClient`` (``openai`` 2.43.0 and 2.54.0, which
  encode the request body as strict UTF-8) the next request raised
  ``UnicodeEncodeError`` before it was sent, after one request. Since BR-022
  ``AgentLoop._build_tool_message`` writes each one as its six-character
  ``\\udXXX`` escape, whatever produced the text.
* A non-string result that cannot be rendered. A value nested past
  ``json.dumps``' reach raised ``RecursionError``; a value whose ``repr``
  fallback raised (an ``int`` over the digit limit, a raising ``__repr__``)
  raised that; and any other ``Exception`` from a value's ``__str__`` under
  ``default=str`` (not ``TypeError`` or ``ValueError``, which already went
  to ``repr``) propagated. Since BR-022 ``_serialize_tool_output`` returns a
  fixed sentence for the first two, and for any ``RecursionError`` from
  ``json.dumps``, and logs one WARNING that carries the call and run ids and
  type names, never the value or the exception text. It falls back to
  ``repr`` for the third (an ``Exception`` other than ``TypeError``,
  ``ValueError`` or ``RecursionError``). The run continues.

What these pin: ``_serialize_tool_output``'s arms directly (the
``RecursionError`` is induced with a patched, delegating ``dumps_for_model``,
so the unit tests do not depend on the interpreter), including that a
``BaseException`` which is not an ``Exception`` still propagates; the
exact tool-result message in every (tool mode, tool-result role) pair,
from every source of tool-message text, and in a native batch, through
``FakeLLMClient``; that text without surrogate code points is unchanged;
the WARNING's key set, and that it names the failing call's ``call_id``
and the run's ``run_id`` on the single path and in a native batch; that a
result is rendered once per call and after ``after_tool``; a real value
nested past the limit the running interpreter measures (I1; the Runner's
twin is in ``tests/runner/test_runner_audit.py``); and the request body
the real client builds, non-streamed (W3) and streamed (W4).

What these do NOT pin: HTTP bytes; how another ``openai``/``httpx`` release
encodes the body (W3's docstring); the time or memory ``json.dumps`` and
``repr`` spend before they fail (BR-022 evidence, P4, measured outside the
suite); surrogate code points in text the loop does not build (the model's
own completion and tool names, messages passed to ``run()``), which still
make the shipped client raise (BR-022 evidence, P1c); and CPython 3.12 or
Linux, where I1's depth was not measured before CI.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Literal, NamedTuple

import pytest
import structlog
from pytest_httpx import HTTPXMock

import fifty_agent_sdk.loop as loop_module
from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ChatResponse,
    DenyToolCall,
    ErrorEvent,
    FinalEvent,
    Interventions,
    JsonModeParser,
    ObservationEvent,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ThoughtEvent,
    ToolFailedEvent,
    ToolMode,
    ToolResult,
    ToolStartedEvent,
)
from fifty_agent_sdk.loop import _UNRENDERED_TOOL_OUTPUT, _serialize_tool_output
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    dict_chain,
    make_multi_tool_response,
    make_response,
    measured_unrenderable_depth,
)

_SECRET = "SECRET-BR022-tool-output-DO-NOT-LOG"
_USER = [ChatMessage(role="user", content="Look the customer up.")]
_SURROGATE_RESULT = "before " + chr(0xD800) + " after فاطمة"
_ESCAPED_RESULT = "before \\ud800 after فاطمة"
_WARNING_KEYS = {"event", "log_level", "call_id", "run_id", "output_type", "error_type"}
_DIGIT_LIMIT = sys.get_int_max_str_digits()
_NEEDS_DIGIT_LIMIT = pytest.mark.skipif(
    _DIGIT_LIMIT == 0, reason="the int-to-str digit limit is disabled (0) in this interpreter"
)
_ENDPOINT = "https://example.com/v1/chat/completions"

Role = Literal["tool", "user", "assistant"]


# --- Test doubles ---------------------------------------------------------------------


class _CountingRepr:
    """A value whose ``repr`` counts its calls and shows a secret; JSON cannot render it."""

    def __init__(self) -> None:
        self.repr_calls = 0
        self.payload = _SECRET

    def __repr__(self) -> str:
        self.repr_calls += 1
        return f"<counting {self.payload}>"


class _StrRaises:
    """``default=str`` raises the given exception; ``repr`` works."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __str__(self) -> str:
        raise self._exc

    def __repr__(self) -> str:
        return "<_StrRaises>"


class _StrAndReprRaise:
    """``__str__`` and ``__repr__`` both raise, with the secret in their messages."""

    def __str__(self) -> str:
        raise TypeError(f"str failed {_SECRET}")

    def __repr__(self) -> str:
        raise RuntimeError(f"repr failed {_SECRET}")


class _ReprIsSurrogate:
    """JSON cannot render it (``__str__`` raises ``TypeError``), and its ``repr`` holds U+D800."""

    def __str__(self) -> str:
        raise TypeError("no str")

    def __repr__(self) -> str:
        return "r" + chr(0xD800)


class _CountsStr:
    """Rendered through ``default=str``; counts the calls and records a flag when called."""

    def __init__(self, flag: dict[str, bool] | None = None) -> None:
        self.str_calls = 0
        self.flag = flag
        self.flag_seen: list[bool] = []

    def __str__(self) -> str:
        self.str_calls += 1
        if self.flag is not None:
            self.flag_seen.append(self.flag["after_tool_ran"])
        return "counted"


def _raise_recursion_for(target: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the loop's ``dumps_for_model`` raise ``RecursionError`` for ``target`` only.

    Every other value, including the tool list the loop renders at
    construction (``_render_tool_descriptions``), goes to the real function.
    """
    real = loop_module.dumps_for_model

    def patched(obj: Any, **kwargs: Any) -> str:
        if obj is target:
            raise RecursionError(f"induced for BR-022 {_SECRET}")
        return real(obj, **kwargs)

    monkeypatch.setattr(loop_module, "dumps_for_model", patched)


# --- Rows: every (tool mode, tool-result role) pair -----------------------------------


class _Row(NamedTuple):
    """One loop configuration and the role its tool observation goes out in."""

    tool_mode: ToolMode | None  # None: the legacy path (JsonModeParser, no tool_mode)
    role: Role | None  # the tool_message_role kwarg; None: not passed
    native_flag: bool  # legacy SafetyConfig(native_tools_enabled=True)
    expected_role: Role


_NATIVE = _Row(ToolMode.NATIVE, None, False, "tool")
_JSON = _Row(ToolMode.JSON, None, False, "assistant")
_JSON_USER = _Row(ToolMode.JSON, "user", False, "user")

_EVERY_ROW = [
    pytest.param(_JSON, id="json"),
    pytest.param(_JSON_USER, id="json_user"),
    pytest.param(_Row(ToolMode.PROSE, None, False, "assistant"), id="prose"),
    pytest.param(_Row(ToolMode.PROSE, "user", False, "user"), id="prose_user"),
    pytest.param(_NATIVE, id="native"),
    pytest.param(_Row(None, None, False, "tool"), id="legacy_tool"),
    pytest.param(_Row(None, "assistant", False, "assistant"), id="legacy_assistant"),
    pytest.param(_Row(None, "user", False, "user"), id="legacy_user"),
    pytest.param(_Row(None, None, True, "tool"), id="legacy_native_flag"),
]
"""Every (tool mode, tool-result role) pair, as in the FR-003 and BR-020 loop tests."""

_TOOL_AND_USER_ROWS = [
    pytest.param(_NATIVE, id="tool_role"),
    pytest.param(_JSON_USER, id="user_role"),
]


def _is_native(row: _Row) -> bool:
    return row.tool_mode is ToolMode.NATIVE or row.native_flag


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


def _tool_turn(row: _Row, name: str, args: dict[str, Any]) -> ChatResponse:
    """One model turn calling ``name`` in the row's protocol."""
    if _is_native(row):
        return make_multi_tool_response([(name, args)])
    if row.tool_mode is ToolMode.PROSE:
        return make_response(
            f"Thought: calling {name}\nAction: {name}\nAction Input: {json.dumps(args)}"
        )
    return make_response(_json_tool(name, args))


def _final_turn(row: _Row) -> ChatResponse:
    if row.tool_mode is ToolMode.NATIVE:
        return make_response("done")
    if row.tool_mode is ToolMode.PROSE:
        return make_response("Thought: done\nFinal Answer: done")
    return make_response(_json_final("done"))


def _make_loop(
    llm: FakeLLMClient,
    row: _Row,
    tools: list[FakeTool],
    *,
    interventions: Interventions | None = None,
    **safety_kwargs: Any,
) -> AgentLoop:
    """A loop for ``row`` over ``tools``; the legacy rows use the 1.7.0 construction."""
    registry = Registry()
    for tool in tools:
        registry.register(tool)
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
        registry=registry,
        prompts=PromptSections(persona="You are a careful records assistant."),
        safety=SafetyConfig(**safety_kwargs),
        model="containment-model",
        interventions=interventions,
        **kwargs,
    )


async def _run(loop: AgentLoop) -> list[AgentEvent]:
    return [event async for event in loop.run(list(_USER))]


def _layout(row: _Row, tool_name: str, text: str) -> str:
    """The success-layout tool-result message content for ``text`` in the row's role."""
    if row.expected_role == "tool":
        return text
    return f"Tool {tool_name} returned: {text}"


def _not_rendered_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in logs if entry.get("event") == "tool_output_not_rendered"]


def _run_id_of(logs: list[dict[str, Any]]) -> str:
    """The ``run_id`` of the one run in ``logs``, from its ``agent_loop_started`` line."""
    started = [entry for entry in logs if entry.get("event") == "agent_loop_started"]
    assert len(started) == 1
    run_id = started[0]["run_id"]
    assert isinstance(run_id, str) and run_id
    return run_id


# --- _serialize_tool_output -----------------------------------------------------------


def test_recursion_error_gives_the_fixed_text_without_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``RecursionError`` from rendering gives the fixed sentence, and ``repr`` is never tried (BR-022).

    On the BR-019 tree (``1a2832e``) and 1.10.1 the ``RecursionError``
    propagated. The error is induced, so this holds on every interpreter.
    """
    value = _CountingRepr()
    _raise_recursion_for(value, monkeypatch)

    result = _serialize_tool_output(value)

    assert result == _UNRENDERED_TOOL_OUTPUT
    assert value.repr_calls == 0


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError(f"str failed {_SECRET}"), id="runtime_error"),
        pytest.param(KeyError(_SECRET), id="key_error"),
    ],
)
def test_other_render_exceptions_fall_back_to_repr(exc: Exception) -> None:
    """An ``Exception`` other than ``TypeError``, ``ValueError`` or ``RecursionError`` from a value's ``__str__`` under ``default=str`` falls back to ``repr``, and logs nothing (BR-022).

    On ``1a2832e`` and 1.10.1 it propagated out of ``_serialize_tool_output``
    and, in a loop, out of ``AgentLoop.run()`` after one LLM call (BR-022
    evidence, P3 and the base-tree loop probe). ``TypeError`` and
    ``ValueError`` are pinned by
    ``test_serialize_tool_output_falls_back_to_repr_when_dumps_raises``; a
    ``RecursionError`` gets the fixed sentence instead
    (``test_recursion_error_gives_the_fixed_text_without_repr``).
    """
    value = _StrRaises(exc)
    # Precondition: json.dumps itself raises this exception for the value.
    with pytest.raises(type(exc)):
        json.dumps(value, default=str)

    with structlog.testing.capture_logs() as logs:
        result = _serialize_tool_output(value)

    assert result == "<_StrRaises>"
    assert _not_rendered_logs(logs) == []


def _over_the_digit_limit() -> int:
    """An ``int`` with one digit more than ``sys.get_int_max_str_digits()`` allows."""
    return 10**_DIGIT_LIMIT


@pytest.mark.parametrize(
    ("make", "repr_error"),
    [
        pytest.param(
            _over_the_digit_limit, ValueError, id="int_over_digit_limit", marks=_NEEDS_DIGIT_LIMIT
        ),
        pytest.param(
            lambda: {"n": _over_the_digit_limit()},
            ValueError,
            id="int_over_digit_limit_in_dict",
            marks=_NEEDS_DIGIT_LIMIT,
        ),
        pytest.param(_StrAndReprRaise, RuntimeError, id="str_and_repr_raise"),
    ],
)
def test_repr_failure_gives_the_fixed_text(make: Any, repr_error: type[Exception]) -> None:
    """A value whose ``repr`` fallback raises gives the fixed sentence (BR-022).

    On ``1a2832e`` and 1.10.1 the ``repr`` error propagated (``ValueError``
    for the integer, on CPython 3.11.15, 3.13.2 and 3.14.3; BR-022 evidence,
    P3). The integer has ``sys.get_int_max_str_digits() + 1`` digits.
    """
    value = make()
    # Precondition: repr itself raises for the value.
    with pytest.raises(repr_error):
        repr(value)

    assert _serialize_tool_output(value) == _UNRENDERED_TOOL_OUTPUT


def test_the_fixed_text_quotes_nothing_from_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed sentence is the documented ASCII text and quotes nothing from the value or the exception (BR-022)."""
    assert _UNRENDERED_TOOL_OUTPUT == (
        "The tool's output could not be converted to text, so it is not shown."
    )
    assert _UNRENDERED_TOOL_OUTPUT.isascii()
    deep_stand_in = _CountingRepr()  # its repr and an attribute hold the secret
    _raise_recursion_for(deep_stand_in, monkeypatch)  # the induced error's text holds it too

    for value in (deep_stand_in, _StrAndReprRaise()):
        result = _serialize_tool_output(value)
        assert result == _UNRENDERED_TOOL_OUTPUT
        assert _SECRET not in result


@pytest.mark.parametrize("route", ["recursion", "repr_raises"])
def test_unrendered_output_logs_one_warning_with_type_codes_only(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Each fixed sentence logs exactly one WARNING carrying ids and type names only, never the value or the exception text (BR-022)."""
    value: Any
    if route == "recursion":
        value = _CountingRepr()
        _raise_recursion_for(value, monkeypatch)
        expected = {"output_type": "_CountingRepr", "error_type": "RecursionError"}
    else:
        value = _StrAndReprRaise()
        expected = {"output_type": "_StrAndReprRaise", "error_type": "RuntimeError"}

    with structlog.testing.capture_logs() as logs:
        result = _serialize_tool_output(value, call_id="call-7", run_id="run-7")

    assert result == _UNRENDERED_TOOL_OUTPUT
    assert logs  # a leak check over an empty capture would pass vacuously
    assert logs == [
        {
            "event": "tool_output_not_rendered",
            "log_level": "warning",
            "call_id": "call-7",
            "run_id": "run-7",
            **expected,
        }
    ]
    assert set(logs[0]) == _WARNING_KEYS
    assert _SECRET not in str(logs)


class _RaisesBase:
    """``__str__`` (and, for the ``repr`` route, ``__repr__``) raise a ``BaseException``."""

    def __init__(self, exc_type: type[BaseException], *, in_repr: bool) -> None:
        self._exc_type = exc_type
        self._in_repr = in_repr

    def __str__(self) -> str:
        if self._in_repr:
            raise TypeError("send it to repr")
        raise self._exc_type()

    def __repr__(self) -> str:
        if self._in_repr:
            raise self._exc_type()
        return "<_RaisesBase>"


@pytest.mark.parametrize("in_repr", [False, True], ids=["from_str", "from_repr"])
@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_base_exceptions_from_rendering_propagate(
    exc_type: type[BaseException], in_repr: bool
) -> None:
    """A ``BaseException`` that is not an ``Exception``, raised by a value's ``__str__`` or by its ``repr`` fallback, propagates from ``_serialize_tool_output`` (BR-022).

    The broad arms catch ``Exception`` only, as the docstring states.
    """
    with pytest.raises(exc_type):
        _serialize_tool_output(_RaisesBase(exc_type, in_repr=in_repr))


def test_circular_and_unserialisable_results_still_get_repr() -> None:
    """A self-referential list and a tuple-keyed dict still get ``repr``, as in 1.10.1 (BR-022 control).

    ``json.dumps`` raises ``ValueError`` and ``TypeError`` for them, which
    1.10.1's arm already handled; BR-022 must not change their text.
    """
    circular: list[Any] = []
    circular.append(circular)
    tuple_keyed = {(1, 2): "v"}

    assert _serialize_tool_output(circular) == "[[...]]"
    assert _serialize_tool_output(tuple_keyed) == "{(1, 2): 'v'}"


# --- Loop level: surrogate code points in the tool-result message ---------------------


@pytest.mark.parametrize("row", _EVERY_ROW)
async def test_str_result_with_surrogate_reaches_the_model_escaped(row: _Row) -> None:
    """A ``str`` result holding U+D800 reaches the model with it written as ``\\ud800``, other text literal, in every tool mode and role; the event keeps the tool's own string (BR-022).

    Before BR-022 the message held U+D800 itself, and with the shipped client
    the next request raised ``UnicodeEncodeError`` (BR-022 evidence, P1).
    ``FakeLLMClient`` never encodes a request, so this pins the content; W3
    pins the request through the real client.
    """
    result = ToolResult(output=_SURROGATE_RESULT)
    llm = FakeLLMClient([_tool_turn(row, "lookup", {"q": "x"}), _final_turn(row)])

    events = await _run(_make_loop(llm, row, [FakeTool("lookup", result=result)]))

    assert len(llm.calls) == 2
    message = llm.calls[1].messages[-1]
    assert message.role == row.expected_role
    assert message.content == _layout(row, "lookup", _ESCAPED_RESULT)
    message.content.encode("utf-8")  # must not raise
    observation = next(e for e in events if isinstance(e, ObservationEvent))
    assert observation.result is result
    assert observation.result.output is _SURROGATE_RESULT
    assert isinstance(events[-1], FinalEvent)


class _Source(NamedTuple):
    """One source of tool-message text holding a surrogate code point."""

    tool: FakeTool
    interventions: Interventions | None
    tool_name: str  # the name the model calls
    for_tool_role: str  # the expected content in the "tool" role, escaped
    for_other_role: str  # the expected content in the "user"/"assistant" roles, escaped
    raw_event_error: str | None  # the ToolFailedEvent.error the run must carry, raw


def _source(name: str) -> _Source:
    ok = ToolResult(output="ok")
    if name == "is_error_text":
        error = "upstream " + chr(0xD800)
        tool = FakeTool("lookup", result=ToolResult(is_error=True, error=error))
        return _Source(
            tool,
            None,
            "lookup",
            "Tool error: upstream \\ud800",
            "Tool lookup failed: upstream \\ud800",
            error,
        )
    if name == "after_tool_note":
        hooks = Interventions(after_tool=lambda *_: "note " + chr(0xDC80))
        return _Source(
            FakeTool("lookup", result=ok),
            hooks,
            "lookup",
            "ok\n\nnote \\udc80",
            "Tool lookup returned: ok\n\nnote \\udc80",
            None,
        )
    if name == "denial_reason":
        hooks = Interventions(before_tool=lambda *_: DenyToolCall(reason="denied " + chr(0xD800)))
        return _Source(
            FakeTool("lookup", result=ok),
            hooks,
            "lookup",
            "Tool call denied: denied \\ud800",
            "Tool lookup failed: Tool call denied: denied \\ud800",
            "Tool call denied: denied " + chr(0xD800),
        )
    if name == "repr_fallback":
        return _Source(
            FakeTool("lookup", result=ToolResult(output=_ReprIsSurrogate())),
            None,
            "lookup",
            "r\\ud800",
            "Tool lookup returned: r\\ud800",
            None,
        )
    assert name == "tool_not_found_text"
    not_found = "ToolNotFound: tool 'find\\ud800' is not registered."
    return _Source(
        FakeTool("lookup", result=ok),
        None,
        "find" + chr(0xD800),
        not_found,
        f"Tool find\\ud800 failed: {not_found}",
        None,
    )


@pytest.mark.parametrize("row", _TOOL_AND_USER_ROWS)
@pytest.mark.parametrize(
    "source_name",
    ["is_error_text", "after_tool_note", "denial_reason", "repr_fallback", "tool_not_found_text"],
)
async def test_every_tool_message_source_is_escaped(row: _Row, source_name: str) -> None:
    """Each source of tool-message text reaches the model with its surrogate code point escaped, in the tool and user roles; ``ToolFailedEvent.error`` keeps the raw text (BR-022).

    Sources: an ``is_error`` text, an ``after_tool`` note (U+DC80) on a clean
    result, a ``before_tool`` denial reason, a ``repr`` fallback that returns
    U+D800, and the ``ToolNotFound`` text for a tool name the model wrote.
    Before BR-022 each made the shipped client raise ``UnicodeEncodeError`` on
    the next request (BR-022 evidence, P1b). Not escaped, and not pinned as a
    fix: the ``name`` field of the ``"tool"``-role reply and the model's own
    assistant turn, which carry the model's tool name (a documented residual).
    """
    source = _source(source_name)
    llm = FakeLLMClient([_tool_turn(row, source.tool_name, {"q": "x"}), _final_turn(row)])
    loop = _make_loop(llm, row, [source.tool], interventions=source.interventions)

    events = await _run(loop)

    message = llm.calls[1].messages[-1]
    assert message.role == row.expected_role
    expected = source.for_tool_role if row.expected_role == "tool" else source.for_other_role
    assert message.content == expected
    message.content.encode("utf-8")  # must not raise
    if row.expected_role == "tool":
        assert message.name == source.tool_name  # the model's name, not escaped (residual)
    if source.raw_event_error is not None:
        failure = next(e for e in events if isinstance(e, ToolFailedEvent))
        assert failure.error == source.raw_event_error
    assert isinstance(events[-1], FinalEvent)


async def test_native_batch_escapes_only_the_message_that_needs_it() -> None:
    """In a native batch only the message whose text held a surrogate code point changes; the Arabic member is unchanged and the ids still pair (BR-022)."""
    lookup = FakeTool("lookup", result=ToolResult(output=_SURROGATE_RESULT))
    profile = FakeTool("profile", result=ToolResult(output={"city": "الرياض"}))
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("lookup", {"q": "a"}), ("profile", {"q": "b"})]),
            make_response("done"),
        ]
    )
    loop = _make_loop(llm, _NATIVE, [lookup, profile], max_concurrent_tool_calls=2)

    await _run(loop)

    assistant = llm.calls[1].messages[-3]
    replies = llm.calls[1].messages[-2:]
    assert assistant.tool_calls is not None
    assert [m.content for m in replies] == [_ESCAPED_RESULT, '{"city": "الرياض"}']
    assert [m.tool_call_id for m in replies] == [tc.id for tc in assistant.tool_calls]
    assert [m.name for m in replies] == ["lookup", "profile"]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("فاطمة الزهراء", id="arabic"),
        pytest.param("the six characters \\ud800 typed literally", id="literal_escape_text"),
        pytest.param("del \x7f here", id="u007f"),
        pytest.param("astral 😀 emoji", id="astral_emoji"),
    ],
)
@pytest.mark.parametrize("row", _EVERY_ROW)
async def test_str_results_without_surrogates_are_unchanged(row: _Row, text: str) -> None:
    """A ``str`` result with no surrogate code point reaches the model exactly as the tool returned it, in every tool mode and role (BR-022 control)."""
    llm = FakeLLMClient([_tool_turn(row, "lookup", {"q": "x"}), _final_turn(row)])

    await _run(_make_loop(llm, row, [FakeTool("lookup", result=ToolResult(output=text))]))

    assert llm.calls[1].messages[-1].content == _layout(row, "lookup", text)


# --- Loop level: a result that cannot be rendered -------------------------------------


@pytest.mark.parametrize("row", _EVERY_ROW)
async def test_unrendered_result_gives_the_fixed_text_in_every_layout(
    monkeypatch: pytest.MonkeyPatch, row: _Row
) -> None:
    """A result that cannot be rendered reaches the model as the fixed sentence in the success layout, the run ends on the model's answer, and one WARNING names the call (BR-022).

    The ``RecursionError`` is induced for one value (the patched
    ``dumps_for_model`` delegates for everything else, including the tool list
    the loop renders at construction). Before BR-022 it propagated out of
    ``run()``.
    """
    value = _CountingRepr()
    _raise_recursion_for(value, monkeypatch)
    llm = FakeLLMClient([_tool_turn(row, "deep", {"q": "x"}), _final_turn(row)])
    loop = _make_loop(llm, row, [FakeTool("deep", result=ToolResult(output=value))])

    with structlog.testing.capture_logs() as logs:
        events = await _run(loop)

    assert len(llm.calls) == 2
    assert llm.calls[1].messages[-1].content == _layout(row, "deep", _UNRENDERED_TOOL_OUTPUT)
    assert isinstance(events[-2], ThoughtEvent) and isinstance(events[-1], FinalEvent)
    assert events[-1].text == "done"
    assert not any(isinstance(e, ErrorEvent) for e in events)
    started = next(e for e in events if isinstance(e, ToolStartedEvent))
    warnings = _not_rendered_logs(logs)
    assert len(warnings) == 1
    assert warnings[0]["call_id"] == started.call_id
    assert warnings[0]["run_id"] == _run_id_of(logs)
    assert set(warnings[0]) == _WARNING_KEYS
    assert value.repr_calls == 0


async def test_unrendered_batch_member_warning_names_its_own_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a native batch, the WARNING for the member that cannot be rendered carries that member's ``call_id`` and the run's ``run_id``, and the other member's message is unchanged (BR-022).

    The unrenderable member is the second call, so a WARNING that names the
    first call, or no call (``call_id=None``), fails here. The single-path
    twin is ``test_unrendered_result_gives_the_fixed_text_in_every_layout``.
    """
    value = _CountingRepr()
    _raise_recursion_for(value, monkeypatch)
    tools = [
        FakeTool("clean", result=ToolResult(output={"city": "الرياض"})),
        FakeTool("deep", result=ToolResult(output=value)),
    ]
    llm = FakeLLMClient(
        [make_multi_tool_response([("clean", {}), ("deep", {})]), make_response("done")]
    )
    loop = _make_loop(llm, _NATIVE, tools, max_concurrent_tool_calls=2)

    with structlog.testing.capture_logs() as logs:
        events = await _run(loop)

    replies = llm.calls[1].messages[-2:]
    assert [m.content for m in replies] == ['{"city": "الرياض"}', _UNRENDERED_TOOL_OUTPUT]
    started = {e.tool_name: e.call_id for e in events if isinstance(e, ToolStartedEvent)}
    assert len(started) == 2 and started["clean"] != started["deep"]
    warnings = _not_rendered_logs(logs)
    assert len(warnings) == 1
    assert warnings[0]["call_id"] == started["deep"]
    assert warnings[0]["run_id"] == _run_id_of(logs)
    assert set(warnings[0]) == _WARNING_KEYS
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"


@pytest.mark.parametrize("batch", [False, True], ids=["single", "native_batch"])
async def test_non_string_result_is_rendered_once_per_call(batch: bool) -> None:
    """A non-string result is rendered once per call: a ``default=str`` conversion runs once, on the single path and per batch member (BR-022).

    Before BR-022 both layouts were computed for every call, so it ran twice.
    """
    first, second = _CountsStr(), _CountsStr()
    if batch:
        tools = [
            FakeTool("a", result=ToolResult(output={"v": first})),
            FakeTool("b", result=ToolResult(output={"v": second})),
        ]
        llm = FakeLLMClient(
            [make_multi_tool_response([("a", {}), ("b", {})]), make_response("done")]
        )
        loop = _make_loop(llm, _NATIVE, tools, max_concurrent_tool_calls=2)
    else:
        tools = [FakeTool("a", result=ToolResult(output={"v": first}))]
        llm = FakeLLMClient([_tool_turn(_JSON, "a", {}), _final_turn(_JSON)])
        loop = _make_loop(llm, _JSON, tools)

    await _run(loop)

    assert first.str_calls == 1
    assert second.str_calls == (1 if batch else 0)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "native_batch"])
async def test_result_is_rendered_after_after_tool_runs(batch: bool) -> None:
    """The result is rendered after its ``after_tool`` hook has run, as before BR-022 (BR-022).

    Rendering once must not move it earlier: a value whose ``__str__`` reads
    host state that ``after_tool`` changes would otherwise render
    differently.
    """
    flag = {"after_tool_ran": False}

    def after_tool(*_: Any) -> None:
        flag["after_tool_ran"] = True

    value = _CountsStr(flag)
    hooks = Interventions(after_tool=after_tool)
    tools = [FakeTool("a", result=ToolResult(output={"v": value}))]
    if batch:
        # Two calls: one native call takes the single path.
        tools.append(FakeTool("b", result=ToolResult(output="ok")))
        llm = FakeLLMClient(
            [make_multi_tool_response([("a", {}), ("b", {})]), make_response("done")]
        )
        loop = _make_loop(llm, _NATIVE, tools, interventions=hooks, max_concurrent_tool_calls=2)
    else:
        llm = FakeLLMClient([_tool_turn(_JSON, "a", {}), _final_turn(_JSON)])
        loop = _make_loop(llm, _JSON, tools, interventions=hooks)

    await _run(loop)

    assert value.flag_seen == [True]


@pytest.mark.parametrize("setup", ["json_single", "native_batch"])
async def test_result_nested_past_the_measured_dumps_limit_does_not_crash_the_loop(
    setup: str,
) -> None:
    """A real dict chain one level past what ``json.dumps`` renders in this test's frame reaches the model as the fixed sentence, and the run ends on the model's answer (BR-022).

    The depth comes from ``measured_unrenderable_depth`` (``tests/loop/
    conftest.py``), called from this test's frame, so it is derived from the
    running interpreter; its docstring says why the loop, which renders the
    value deeper on the stack, cannot render it either. The depth it derived
    was about 950 levels on CPython 3.11.15, 9,980 on 3.13.2 and 61,470 on
    3.14.3, moving by a few levels with the test harness (3.12 and Linux
    first run in CI). Before
    BR-022 this run raised a raw ``RecursionError`` out of ``run()`` from
    ``json.dumps`` in ``_model_json.dumps_for_model``, after one LLM call,
    on all three (BR-022 evidence, P2). The batch's clean member is
    unchanged. The deep value only meets the code under test: only scalars
    and short strings reach an ``assert``.
    """
    depth = measured_unrenderable_depth(include_repr=False, shape=dict_chain)
    deep = ToolResult(output=dict_chain(depth))
    tools = [FakeTool("deep", result=deep)]
    if setup == "native_batch":
        tools.append(FakeTool("clean", result=ToolResult(output={"city": "الرياض"})))
        llm = FakeLLMClient(
            [make_multi_tool_response([("deep", {}), ("clean", {})]), make_response("done")]
        )
        loop = _make_loop(llm, _NATIVE, tools, max_concurrent_tool_calls=2)
    else:
        llm = FakeLLMClient([_tool_turn(_JSON, "deep", {}), _final_turn(_JSON)])
        loop = _make_loop(llm, _JSON, tools)

    with structlog.testing.capture_logs() as logs:
        events = await _run(loop)

    kinds = [type(e).__name__ for e in events]
    assert kinds[-2:] == ["ThoughtEvent", "FinalEvent"]
    assert "ErrorEvent" not in kinds
    observations = [e for e in events if isinstance(e, ObservationEvent)]
    same_result = observations[0].result is deep
    assert same_result
    assert len(llm.calls) == 2
    if setup == "native_batch":
        contents = [m.content for m in llm.calls[1].messages[-2:]]
        assert contents == [_UNRENDERED_TOOL_OUTPUT, '{"city": "الرياض"}']
    else:
        assert llm.calls[1].messages[-1].content == f"Tool deep returned: {_UNRENDERED_TOOL_OUTPUT}"
    warnings = _not_rendered_logs(logs)
    assert len(warnings) == 1
    assert warnings[0]["output_type"] == "dict"
    assert warnings[0]["error_type"] == "RecursionError"
    deep_call_id = next(
        e.call_id for e in events if isinstance(e, ToolStartedEvent) and e.tool_name == "deep"
    )
    assert warnings[0]["call_id"] == deep_call_id
    assert warnings[0]["run_id"] == _run_id_of(logs)


# --- Through the real client -----------------------------------------------------------


def _native_tool_body() -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _text_body(content: str) -> dict[str, Any]:
    return {
        "id": "cmpl-2",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize(
    "row", [pytest.param(_NATIVE, id="native"), pytest.param(_JSON, id="json")]
)
async def test_str_result_with_surrogate_completes_through_the_real_client(
    httpx_mock: HTTPXMock, row: _Row
) -> None:
    """Through the real ``OpenAICompatibleClient``, a ``str`` result holding U+D800 no longer stops the run: the second request is sent with the escaped text and the run ends on the model's answer (BR-022).

    Release dependence: on ``openai`` 2.43.0 and 2.54.0 (BR-022 evidence, P7)
    the client encodes the request body as strict UTF-8, and before BR-022 this
    run raised ``UnicodeEncodeError`` out of ``run()`` after one request (P1).
    On a release whose encoder escaped the body itself, removing the loop's
    escape would not make this test raise, but the exact-content assertion
    would still see it (by reading; not run);
    ``test_str_result_with_surrogate_reaches_the_model_escaped``
    pins the loop on any release.
    """
    first = _native_tool_body() if _is_native(row) else _text_body(_json_tool("lookup", {"q": "x"}))
    final = _text_body("done" if _is_native(row) else _json_final("done"))
    httpx_mock.add_response(method="POST", url=_ENDPOINT, json=first)
    httpx_mock.add_response(method="POST", url=_ENDPOINT, json=final)
    loop = AgentLoop(
        llm=OpenAICompatibleClient(
            api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
        ),
        registry=_registry_with(FakeTool("lookup", result=ToolResult(output=_SURROGATE_RESULT))),
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=row.tool_mode,
    )

    events = await _run(loop)

    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    reply = json.loads(requests[1].content)["messages"][-1]
    assert reply["role"] == row.expected_role
    assert reply["content"] == _layout(row, "lookup", _ESCAPED_RESULT)
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"
    assert not any(isinstance(e, ErrorEvent) for e in events)


def _sse_body(content: str) -> bytes:
    """A ``text/event-stream`` body that streams ``content`` and then ``stop``."""
    chunks = [
        {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    events = [
        {
            "id": "cmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [choice],
        }
        for choice in chunks
    ]
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


async def test_str_result_with_surrogate_completes_through_the_real_client_streamed(
    httpx_mock: HTTPXMock,
) -> None:
    """With ``stream=True`` through the real ``OpenAICompatibleClient``, a ``str`` result holding U+D800 reaches the second streamed request escaped, and the run ends on the model's answer (BR-022).

    The request body is encoded by the ``create()`` call that opens each
    stream. On ``1a2832e`` this run raised ``UnicodeEncodeError`` raw from
    that call after one request (``openai`` 2.43.0 and 2.54.0; BR-022
    evidence, round 2). The release dependence is W3's: on a release whose
    encoder escaped the body itself, the run would not raise, but the
    exact-content assertion would still see a missing escape (by reading;
    not run).
    """
    for content in (_json_tool("lookup", {"q": "x"}), _json_final("done")):
        httpx_mock.add_response(
            method="POST",
            url=_ENDPOINT,
            content=_sse_body(content),
            headers={"content-type": "text/event-stream"},
        )
    loop = AgentLoop(
        llm=OpenAICompatibleClient(
            api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
        ),
        registry=_registry_with(FakeTool("lookup", result=ToolResult(output=_SURROGATE_RESULT))),
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=ToolMode.JSON,
        stream=True,
    )

    events = await _run(loop)

    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    body = json.loads(requests[1].content)
    assert body["stream"] is True
    reply = body["messages"][-1]
    assert reply["role"] == "assistant"
    assert reply["content"] == f"Tool lookup returned: {_ESCAPED_RESULT}"
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"
    assert not any(isinstance(e, ErrorEvent) for e in events)


def _registry_with(tool: FakeTool) -> Registry:
    registry = Registry()
    registry.register(tool)
    return registry
