"""Tests for ``fifty_agent_sdk.parser.native_tools``.

Covers the concrete :class:`NativeToolsParser` (BR-007): it consumes a
populated ``response.message.tool_calls`` list and returns a single
:class:`ThoughtAction`, satisfying :class:`NativeToolsParserProtocol`. Mirrors
the helper-built ``ChatResponse`` + ``pytest.raises(ParserError)`` +
context-dict style of the text-parser suites.
"""

from __future__ import annotations

import pytest
import structlog

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ChatMessage, ChatResponse, ToolCall, Usage
from fifty_agent_sdk.parser import NativeToolsParser, NativeToolsParserProtocol
from fifty_agent_sdk.parser.base import ParseResult, ThoughtAction


def _make_response(
    *,
    tool_calls: list[ToolCall] | None,
    content: str = "",
) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_native_parser_satisfies_protocol() -> None:
    """The concrete class satisfies the runtime-checkable Protocol."""
    assert isinstance(NativeToolsParser(), NativeToolsParserProtocol)


def test_protocol_runtime_checkable_with_fake() -> None:
    """A trivial fake with the right method shape satisfies the Protocol."""

    class _Fake:
        def parse(self, response: ChatResponse) -> ParseResult:
            raise NotImplementedError

    assert isinstance(_Fake(), NativeToolsParserProtocol)


def test_protocol_rejects_class_without_parse() -> None:
    class _Empty:
        pass

    assert not isinstance(_Empty(), NativeToolsParserProtocol)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_native_parser_single_call_returns_thought_action() -> None:
    call = ToolCall(name="search", args={"q": "x"})
    parser = NativeToolsParser()

    result = parser.parse(_make_response(tool_calls=[call]))

    assert isinstance(result, ThoughtAction)
    # Native calls carry no preceding reasoning text — thought is empty.
    assert result.thought == ""
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_native_parser_single_call_keeps_empty_args() -> None:
    call = ToolCall(name="ping", args={})
    parser = NativeToolsParser()

    result = parser.parse(_make_response(tool_calls=[call]))

    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "ping"
    assert result.tool_call.args == {}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_native_parser_empty_tool_calls_raises() -> None:
    parser = NativeToolsParser()
    response = _make_response(tool_calls=None, content="some assistant text")

    with pytest.raises(ParserError) as excinfo:
        parser.parse(response)

    ctx = excinfo.value.context
    assert ctx["parser"] == "NativeToolsParser"
    assert ctx["error_phase"] == "empty_tool_calls"
    assert ctx["completion_excerpt"] == "some assistant text"


def test_native_parser_empty_list_tool_calls_raises() -> None:
    parser = NativeToolsParser()
    response = _make_response(tool_calls=[])

    with pytest.raises(ParserError) as excinfo:
        parser.parse(response)

    assert excinfo.value.context["error_phase"] == "empty_tool_calls"


def test_native_parser_malformed_entry_raises() -> None:
    """A hand-built entry with an empty name fails schema validation.

    Built via ``model_construct`` to bypass the SDK ``ToolCall`` schema
    validation, exercising the parser's own defensive check.
    """
    bad_call = ToolCall.model_construct(name="", args={})
    parser = NativeToolsParser()
    response = _make_response(tool_calls=[bad_call])  # type: ignore[list-item]

    with pytest.raises(ParserError) as excinfo:
        parser.parse(response)

    ctx = excinfo.value.context
    assert ctx["parser"] == "NativeToolsParser"
    assert ctx["error_phase"] == "schema_validation"


def test_native_parser_non_dict_args_raises() -> None:
    """A hand-built entry with non-dict args fails schema validation."""
    bad_call = ToolCall.model_construct(name="search", args="not-a-dict")  # type: ignore[arg-type]
    parser = NativeToolsParser()
    response = _make_response(tool_calls=[bad_call])  # type: ignore[list-item]

    with pytest.raises(ParserError) as excinfo:
        parser.parse(response)

    assert excinfo.value.context["error_phase"] == "schema_validation"


# ---------------------------------------------------------------------------
# Multi-call (BR-007 scope limitation)
# ---------------------------------------------------------------------------


def test_native_parser_multi_call_uses_first_and_logs() -> None:
    first = ToolCall(name="search", args={"q": "first"})
    second = ToolCall(name="lookup", args={"id": "second"})
    parser = NativeToolsParser()

    with structlog.testing.capture_logs() as logs:
        result = parser.parse(_make_response(tool_calls=[first, second]))

    # The FIRST call is used; the rest are dropped until BR-006.
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "first"}

    # The truncation is observable at DEBUG, not silent.
    truncated = [e for e in logs if e.get("event") == "native_tool_calls_truncated"]
    assert len(truncated) == 1
    assert truncated[0]["count"] == 2
    assert truncated[0]["log_level"] == "debug"
