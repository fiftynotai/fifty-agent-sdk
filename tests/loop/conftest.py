"""Shared test fakes for the loop integration tests.

Three stand-ins:

* :class:`FakeLLMClient` — replays a scripted sequence of
  :class:`fifty_agent_sdk.llm.types.ChatResponse` values (or exceptions) on
  successive :meth:`complete` / :meth:`stream` calls. Records every
  inbound :class:`fifty_agent_sdk.llm.types.ChatRequest` for assertion.
* :class:`DriftsOnceFakeLLM` — returns scripted prose drift on the first
  call and a clean JSON envelope on every subsequent call. Used by the
  BR-018 parser-retry tests to model the real failure pattern.
* :class:`FakeTool` — a configurable :class:`fifty_agent_sdk.tools.protocol.Tool`
  whose :meth:`invoke` either returns a scripted
  :class:`fifty_agent_sdk.tools.protocol.ToolResult` or raises a configured
  exception, with optional latency.

Helpers :func:`make_response` and :func:`make_stream_chunks` reduce
boilerplate in the individual test files.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fifty_agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ToolCall,
    Usage,
)
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema


class FakeLLMClient:
    """Scripted LLM client for loop integration tests.

    Each scripted reply is consumed in order on successive ``complete()``
    or ``stream()`` calls. A reply may be:

    * A :class:`ChatResponse` — yielded as the entire result (one chunk
      in stream mode).
    * A ``list[ChatResponse]`` — yielded chunk-by-chunk in stream mode.
      In ``complete()`` mode this raises an assertion (chunked replies
      are stream-only).
    * An :class:`Exception` — raised immediately.

    Every inbound :class:`ChatRequest` is appended to :attr:`calls` so
    tests can assert on what was sent to the LLM.
    """

    def __init__(self, replies: list[ChatResponse | list[ChatResponse] | Exception]) -> None:
        self._replies: list[ChatResponse | list[ChatResponse] | Exception] = list(replies)
        self.calls: list[ChatRequest] = []

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if not self._replies:
            raise AssertionError("FakeLLMClient: no more scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            raise AssertionError(
                "FakeLLMClient.complete() got a chunked reply; "
                "use stream() or pass a single ChatResponse"
            )
        return reply

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        self.calls.append(request)
        if not self._replies:
            raise AssertionError("FakeLLMClient: no more scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            for chunk in reply:
                yield chunk
            return
        yield reply


def make_response(content: str, finish_reason: FinishReason = "stop") -> ChatResponse:
    """Build a non-streaming :class:`ChatResponse` with zeroed usage figures."""
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content),
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        finish_reason=finish_reason,
    )


def make_multi_tool_response(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    content: str = "",
) -> ChatResponse:
    """Build a non-streaming ChatResponse carrying N native tool_calls entries.

    Mirrors :func:`make_response` but populates ``message.tool_calls`` with one
    :class:`~fifty_agent_sdk.llm.types.ToolCall` per ``(name, args)`` tuple, in
    the given (CALL) order. Used by the BR-006 multi-call dispatch tests to
    model a provider-native function-calling reply that requests several tools
    in a single turn.
    """
    return ChatResponse(
        message=ChatMessage(
            role="assistant",
            content=content,
            tool_calls=[ToolCall(name=name, args=dict(args)) for name, args in calls],
        ),
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        finish_reason="tool_calls",
    )


def make_stream_chunks(parts: list[str]) -> list[ChatResponse]:
    """Build a list of chunked :class:`ChatResponse`.

    All chunks but the last carry ``finish_reason="in_progress"``; the
    last one terminates with ``"stop"``.
    """
    if not parts:
        raise ValueError("make_stream_chunks: need at least one chunk part")
    chunks: list[ChatResponse] = []
    last_index = len(parts) - 1
    for index, part in enumerate(parts):
        chunks.append(
            ChatResponse(
                message=ChatMessage(role="assistant", content=part),
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="stop" if index == last_index else "in_progress",
            )
        )
    return chunks


class DriftsOnceFakeLLM:
    """Returns scripted prose drift on first call, then ``json_reply`` on subsequent calls.

    Models the BR-018 failure pattern: model emits a Markdown list outside
    the envelope on the first call, then on the format-reminder retry
    returns a clean envelope. Locks the retry mitigation: without the
    retry, this fake causes a ``ParserError``-terminated run; with the
    retry, the loop self-heals.

    Args:
        prose_reply: The drift content returned on the first call (the one
            that triggers :class:`fifty_agent_sdk.errors.ParserError`).
        json_reply: The well-formed JSON envelope returned on every
            subsequent call.
    """

    def __init__(self, *, prose_reply: str, json_reply: str) -> None:
        self._prose_reply = prose_reply
        self._json_reply = json_reply
        self.call_count = 0
        self._calls: list[ChatRequest] = []

    def _select_reply(self) -> str:
        """Pick the prose drift on call 1, the JSON envelope on every call after."""
        self.call_count += 1
        if self.call_count == 1:
            return self._prose_reply
        return self._json_reply

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self._calls.append(request)
        return make_response(self._select_reply())

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        self._calls.append(request)
        yield make_response(self._select_reply())

    @property
    def calls(self) -> list[ChatRequest]:
        """Recorded inbound requests, in call order, for assertion."""
        return self._calls


class FakeTool:
    """Configurable :class:`fifty_agent_sdk.tools.protocol.Tool` for loop tests.

    Args:
        name: The tool name as the registry will key it.
        result: Optional :class:`ToolResult` to return when ``raises`` is
            ``None``. Defaults to ``ToolResult(output="ok")``.
        raises: Optional exception to raise instead of returning a result.
        sleep_seconds: Optional latency for the invoke coroutine (used to
            exercise the registry's timeout path).
    """

    def __init__(
        self,
        name: str,
        *,
        result: ToolResult | None = None,
        raises: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.name = name
        self.description = f"Test fake tool: {name}"
        self.schema = ToolSchema()
        self._result = result or ToolResult(output="ok")
        self._raises = raises
        self._sleep = sleep_seconds
        self.call_count = 0
        self.last_args: dict[str, Any] | None = None

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        self.call_count += 1
        self.last_args = args
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._raises:
            raise self._raises
        return self._result


__all__ = [
    "DriftsOnceFakeLLM",
    "FakeLLMClient",
    "FakeTool",
    "make_multi_tool_response",
    "make_response",
    "make_stream_chunks",
]
