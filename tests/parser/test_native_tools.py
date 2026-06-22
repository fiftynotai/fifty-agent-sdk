"""Tests for ``fifty_agent_sdk.parser.native_tools``."""

from __future__ import annotations

import pytest

from fifty_agent_sdk.llm.types import ChatMessage, ChatResponse, Usage
from fifty_agent_sdk.parser import (
    NativeToolsParser,
    NativeToolsParserStub,
    ParseResult,
)


def _make_response() -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content="ignored"),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def test_stub_raises_not_implemented_error() -> None:
    stub = NativeToolsParserStub()
    with pytest.raises(NotImplementedError) as excinfo:
        stub.parse(_make_response())
    # The message should point future implementers at the integration spot.
    assert "native function calling" in str(excinfo.value).lower()


def test_stub_is_a_native_tools_parser() -> None:
    """Structural conformance: the stub satisfies the Protocol."""
    assert isinstance(NativeToolsParserStub(), NativeToolsParser)


def test_protocol_runtime_checkable_with_fake() -> None:
    """A trivial fake with the right method shape satisfies the Protocol."""

    class _Fake:
        def parse(self, response: ChatResponse) -> ParseResult:
            raise NotImplementedError

    assert isinstance(_Fake(), NativeToolsParser)


def test_protocol_rejects_class_without_parse() -> None:
    class _Empty:
        pass

    assert not isinstance(_Empty(), NativeToolsParser)
