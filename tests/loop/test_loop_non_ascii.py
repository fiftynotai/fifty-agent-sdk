"""Non-ASCII text in the JSON the loop writes for the model (BR-020).

Tool results that are not strings and the text-mode tool list are serialised
through ``fifty_agent_sdk._model_json.dumps_for_model``, so Arabic, CJK and
accented text reach the model literally instead of as ``\\uXXXX`` escapes.

What these pin:

* Unit level: ``_serialize_tool_output`` keeps Arabic literal, its length
  equals the ``ensure_ascii=False`` length (``default=str`` route included),
  results whose strings hold only U+0000-U+007E equal the 1.10.1 expression
  ``json.dumps(output, default=str)`` as text and as UTF-8 bytes, and a
  surrogate code point gets escaped JSON rather than the ``repr`` fallback
  (a ``default=str`` value beside it included). ``_render_tool_descriptions``
  keeps non-ASCII schema text literal, and a schema holding a surrogate code
  point gets the escaped JSON with sorted keys.
* Loop level: the exact tool-result message the model receives, in every
  (tool mode, tool-result role) pair on the single-call path and in a native
  batch; the provider-facing body of a native round trip (through
  ``tests.loop.golden_capture.wire_bodies``), including the replayed
  ``arguments`` string and the ``tool_calls[].id`` / ``tool_call_id``
  pairing; and the text-mode system prompt.

What these do NOT pin: HTTP bytes (the ``openai`` SDK's outer serialisation),
or how a real model tokenises the text. The three golden fixtures, whose
SDK-written JSON holds only U+0000-U+007E, still pin their 46 scenarios
unchanged; these tests do not replace them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal, NamedTuple

import pytest

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    ChatMessage,
    ChatResponse,
    JsonModeParser,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    ToolResult,
)
from fifty_agent_sdk.loop import _render_tool_descriptions, _serialize_tool_output
from fifty_agent_sdk.tools.protocol import ToolSchema
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
)
from tests.loop.golden_capture import wire_bodies

_ARABIC = {"name": "فاطمة"}
_ARABIC_JSON = '{"name": "فاطمة"}'
_USER = [ChatMessage(role="user", content="Who is the customer?")]

Role = Literal["tool", "user", "assistant"]


# --- Test doubles and builders -------------------------------------------------------


class _StrIsArabic:
    """Not JSON-serialisable; ``default=str`` turns it into Arabic text."""

    def __str__(self) -> str:
        return "تقرير شهري"


class _StrIsAscii:
    """Not JSON-serialisable; ``default=str`` turns it into ASCII text."""

    def __str__(self) -> str:
        return "plain <object> 1"


class _Row(NamedTuple):
    """One loop configuration and the role its tool observation goes out in."""

    tool_mode: ToolMode | None  # None: the legacy path (JsonModeParser, no tool_mode)
    role: Role | None  # the tool_message_role kwarg; None: not passed
    native_flag: bool  # legacy SafetyConfig(native_tools_enabled=True)
    expected_role: Role


_EVERY_ROW = [
    pytest.param(_Row(ToolMode.JSON, None, False, "assistant"), id="json"),
    pytest.param(_Row(ToolMode.JSON, "user", False, "user"), id="json_user"),
    pytest.param(_Row(ToolMode.PROSE, None, False, "assistant"), id="prose"),
    pytest.param(_Row(ToolMode.PROSE, "user", False, "user"), id="prose_user"),
    pytest.param(_Row(ToolMode.NATIVE, None, False, "tool"), id="native"),
    pytest.param(_Row(None, None, False, "tool"), id="legacy_tool"),
    pytest.param(_Row(None, "assistant", False, "assistant"), id="legacy_assistant"),
    pytest.param(_Row(None, "user", False, "user"), id="legacy_user"),
    pytest.param(_Row(None, None, True, "tool"), id="legacy_native_flag"),
]
"""Every (tool mode, tool-result role) pair, as in the FR-003 intervention tests."""

_NATIVE_ROWS = [
    pytest.param(_Row(ToolMode.NATIVE, None, False, "tool"), id="native"),
    pytest.param(_Row(None, None, True, "tool"), id="legacy_native_flag"),
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
    llm: FakeLLMClient, row: _Row, tools: list[FakeTool], **safety_kwargs: Any
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
        model="non-ascii-model",
        **kwargs,
    )


async def _run(loop: AgentLoop) -> None:
    async for _event in loop.run(list(_USER)):
        pass


def _expected_content(row: _Row, tool_name: str, serialised: str) -> str:
    if row.expected_role == "tool":
        return serialised
    return f"Tool {tool_name} returned: {serialised}"


# --- _serialize_tool_output ------------------------------------------------------------


def test_serialize_tool_output_keeps_arabic_literal() -> None:
    """A dict tool result holding Arabic serialises to the literal text, not ``\\u`` escapes (BR-020)."""
    assert _serialize_tool_output(_ARABIC) == _ARABIC_JSON


def test_serialize_tool_output_non_ascii_length_matches_unescaped() -> None:
    """A non-ASCII result's output and length equal ``json.dumps(..., default=str, ensure_ascii=False)``, ``default=str`` route included, and it decodes as 1.10.1's did (BR-020)."""
    payload: dict[str, Any] = {
        "name": "فاطمة الزهراء",
        "city": "東京",
        "accented": "José Núñez",
        "emoji": "ok 😀",
        "مفتاح": ["قيمة", {"nested": "naïve café"}],
        "report": _StrIsArabic(),
    }
    escaped = json.dumps(payload, default=str)
    unescaped = json.dumps(payload, default=str, ensure_ascii=False)
    # Precondition: the payload really exercises the bug, including via default=str.
    assert len(unescaped) < len(escaped)
    assert '"report": "تقرير شهري"' in unescaped

    result = _serialize_tool_output(payload)

    assert result == unescaped
    assert len(result) == len(unescaped)
    assert json.loads(result) == json.loads(escaped)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"b": [1, 2.5, None], "a": {"z": True, "y": "x"}}, id="nested_dict"),
        pytest.param([{"id": 1}, {"id": 2}], id="list_of_dicts"),
        pytest.param({"quote": 'say "hi"', "path": "C:\\dir", "nl": "a\nb\tc"}, id="escapes"),
        pytest.param(
            {"when": datetime(2026, 10, 2, 9, 1, tzinfo=UTC), "obj": _StrIsAscii()},
            id="default_str_objects",
        ),
        pytest.param(42, id="int"),
        pytest.param(None, id="none"),
        pytest.param(("t", 1), id="tuple"),
    ],
)
def test_serialize_tool_output_ascii_is_byte_identical_to_1_10_1(payload: Any) -> None:
    """A result whose strings hold only U+0000-U+007E equals the 1.10.1 expression ``json.dumps(output, default=str)``, as text and as UTF-8 bytes (BR-020).

    U+007F, the one ASCII exception, is pinned in ``tests/test_model_json.py``.
    """
    expected = json.dumps(payload, default=str)  # the 1.10.1 expression, frozen here

    result = _serialize_tool_output(payload)

    assert result == expected
    assert result.encode("utf-8") == expected.encode("utf-8")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"s": chr(0xD800)}, id="lone_high_surrogate"),
        pytest.param({"a": "فاطمة", "s": chr(0xDC00)}, id="surrogate_beside_arabic"),
        pytest.param(
            {"when": datetime(2026, 10, 2, 9, 1, tzinfo=UTC), "s": chr(0xD800)},
            id="datetime_beside_surrogate",
        ),
    ],
)
def test_serialize_tool_output_lone_surrogate_keeps_json_not_repr(payload: dict[str, Any]) -> None:
    """A result holding a surrogate code point gets the escaped 1.10.1 JSON, never the ``repr`` fallback, and encodes as UTF-8 (BR-020).

    ``UnicodeEncodeError`` is a ``ValueError`` subclass, so if it escaped the
    helper, ``_serialize_tool_output``'s ``except (TypeError, ValueError)``
    arm would return ``repr(output)``. The datetime case needs ``default=str``
    in the fallback too: without it the fallback raises ``TypeError``, which
    also lands in that arm.
    """
    result = _serialize_tool_output(payload)

    assert result == json.dumps(payload, default=str)
    assert result != repr(payload)
    result.encode("utf-8")  # must not raise


# --- Loop level: the tool-result message ----------------------------------------------


@pytest.mark.parametrize("row", _EVERY_ROW)
async def test_tool_returning_arabic_dict_reaches_model_as_literal_text(row: _Row) -> None:
    """A tool returning ``{"name": "فاطمة"}`` reaches the model as that literal text, in every tool mode and role (BR-020)."""
    tool = FakeTool("lookup", result=ToolResult(output=dict(_ARABIC)))
    llm = FakeLLMClient([_tool_turn(row, "lookup", {"q": "x"}), _final_turn(row)])
    loop = _make_loop(llm, row, [tool])

    await _run(loop)

    assert tool.call_count == 1
    assert len(llm.calls) == 2
    message = llm.calls[1].messages[-1]
    assert message.role == row.expected_role
    assert message.content == _expected_content(row, "lookup", _ARABIC_JSON)
    assert "\\u" not in message.content


@pytest.mark.parametrize("row", _NATIVE_ROWS)
async def test_native_batch_tool_results_keep_arabic_literal(row: _Row) -> None:
    """Both results of a native two-call batch reach the model as literal Arabic, in call order (BR-020)."""
    lookup = FakeTool("lookup", result=ToolResult(output=dict(_ARABIC)))
    profile = FakeTool("profile", result=ToolResult(output={"city": "الرياض"}))
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("lookup", {"q": "a"}), ("profile", {"q": "b"})]),
            _final_turn(row),
        ]
    )
    loop = _make_loop(llm, row, [lookup, profile], max_concurrent_tool_calls=2)

    await _run(loop)

    assert lookup.call_count == profile.call_count == 1
    replies = llm.calls[1].messages[-2:]
    assert [m.role for m in replies] == ["tool", "tool"]
    assert [m.name for m in replies] == ["lookup", "profile"]
    assert [m.content for m in replies] == [_ARABIC_JSON, '{"city": "الرياض"}']


@pytest.mark.parametrize("row", _NATIVE_ROWS)
async def test_native_round_trip_wire_body_keeps_arabic_and_pairing(row: _Row) -> None:
    """In the provider body of a native round trip, the replayed ``arguments`` and the tool content are literal Arabic, and the ids still pair (BR-020).

    The body is built by the real ``OpenAICompatibleClient._build_body``; it
    is not the HTTP bytes.
    """
    tool = FakeTool("lookup", result=ToolResult(output=dict(_ARABIC)))
    llm = FakeLLMClient([_tool_turn(row, "lookup", dict(_ARABIC)), _final_turn(row)])
    loop = _make_loop(llm, row, [tool])

    await _run(loop)
    bodies = await wire_bodies(llm, stream=False)

    assert tool.last_args == _ARABIC
    assistant, reply = bodies[1]["messages"][-2:]
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 1
    call = assistant["tool_calls"][0]
    assert call["function"]["arguments"] == _ARABIC_JSON
    assert json.loads(call["function"]["arguments"]) == _ARABIC
    assert reply["role"] == "tool"
    assert reply["content"] == _ARABIC_JSON
    assert isinstance(call["id"], str) and call["id"]
    assert call["id"] == reply["tool_call_id"]


# --- The text-mode tool list ----------------------------------------------------------


async def test_text_mode_tool_list_keeps_non_ascii_schema_text() -> None:
    """Non-ASCII schema text reaches the text-mode tool list, and the system prompt, literally (BR-020)."""
    properties = {
        "name": {"type": "string", "description": "الاسم الكامل"},
        "age": {"type": "integer", "description": "العمر بالسنوات"},
    }
    tool = FakeTool("lookup")
    tool.schema = ToolSchema(properties=properties)
    args_line = "  args: " + json.dumps(properties, sort_keys=True, ensure_ascii=False)
    # Precondition: the escaped form differs, so the comparison can see a revert.
    assert json.dumps(properties, sort_keys=True) not in args_line

    assert _render_tool_descriptions([tool]) == f"- lookup: {tool.description}\n{args_line}"

    row = _Row(ToolMode.JSON, None, False, "assistant")
    llm = FakeLLMClient([_final_turn(row)])
    await _run(_make_loop(llm, row, [tool]))

    system = llm.calls[0].messages[0]
    assert system.role == "system"
    assert args_line in system.content


def test_text_mode_tool_list_with_surrogate_keeps_sorted_escaped_json() -> None:
    """A schema holding a surrogate code point gets the escaped 1.10.1 ``args`` JSON, keys still sorted (BR-020)."""
    properties = {
        "zone": {"type": "string", "description": "bad text " + chr(0xD800)},
        "area": {"type": "string", "description": "المنطقة"},
    }
    tool = FakeTool("lookup")
    tool.schema = ToolSchema(properties=properties)
    expected_args = json.dumps(properties, sort_keys=True)  # the 1.10.1 expression
    # Precondition: sorting changes the escaped text, so a fallback that drops it is visible.
    assert expected_args != json.dumps(properties)

    rendered = _render_tool_descriptions([tool])

    assert rendered == f"- lookup: {tool.description}\n  args: {expected_args}"
    rendered.encode("utf-8")  # must not raise
