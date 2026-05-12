"""Shared test fakes for the loop integration tests.

Two stand-ins:

* :class:`FakeLLMClient` — replays a scripted sequence of
  :class:`agent_sdk.llm.types.ChatResponse` values (or exceptions) on
  successive :meth:`complete` / :meth:`stream` calls. Records every
  inbound :class:`agent_sdk.llm.types.ChatRequest` for assertion.
* :class:`FakeTool` — a configurable :class:`agent_sdk.tools.protocol.Tool`
  whose :meth:`invoke` either returns a scripted
  :class:`agent_sdk.tools.protocol.ToolResult` or raises a configured
  exception, with optional latency.

Helpers :func:`make_response` and :func:`make_stream_chunks` reduce
boilerplate in the individual test files.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    Usage,
)
from agent_sdk.tools.protocol import ToolResult, ToolSchema


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

    def __init__(
        self, replies: list[ChatResponse | list[ChatResponse] | Exception]
    ) -> None:
        self._replies: list[ChatResponse | list[ChatResponse] | Exception] = list(
            replies
        )
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


def make_response(
    content: str, finish_reason: FinishReason = "stop"
) -> ChatResponse:
    """Build a non-streaming :class:`ChatResponse` with zeroed usage figures."""
    return ChatResponse(
        message=ChatMessage(role="assistant", content=content),
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        finish_reason=finish_reason,
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


class FakeTool:
    """Configurable :class:`agent_sdk.tools.protocol.Tool` for loop tests.

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


__all__ = ["FakeLLMClient", "FakeTool", "make_response", "make_stream_chunks"]
