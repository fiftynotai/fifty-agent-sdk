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

BR-022 adds :func:`dict_chain` and :func:`list_chain` (deep values built
without recursion) and :func:`measured_unrenderable_depth`, which derives
from the running interpreter a depth at which a tool result cannot be
rendered. The Runner tests import them from here too.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from fifty_agent_sdk._model_json import dumps_for_model
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


def dict_chain(depth: int) -> Any:
    """``{"a": {"a": ... 0}}``, ``depth`` dicts deep, built without recursion (BR-022)."""
    value: Any = 0
    for _ in range(depth):
        value = {"a": value}
    return value


def list_chain(depth: int) -> Any:
    """``[[... 0]]``, ``depth`` lists deep, built without recursion (BR-022)."""
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


_MAX_REPR_DOUBLINGS = 4
"""How many times :func:`measured_unrenderable_depth` may double its depth looking for ``repr``'s limit."""


def measured_unrenderable_depth(*, include_repr: bool, shape: Callable[[int], Any]) -> int:
    """A depth at which ``shape(depth)`` cannot be rendered, measured in this frame (BR-022).

    Every attempt is made directly from this function's own frame, with
    inline loops and no helper call, because one frame more or less moves
    the limit on CPython 3.11 (BR-019's lesson; BR-022 plan P4):

    1. ``E``, the deepest ``shape(d)`` that
       ``dumps_for_model(value, default=str)`` renders here, is found by
       doubling from 128 and then a binary search.
    2. The candidate is ``E + 1``. Precondition, asserted here: it raises
       ``RecursionError`` from ``dumps_for_model``.
    3. With ``include_repr``, the candidate must also make ``repr`` raise
       ``RecursionError`` here. While it does not, the candidate is doubled
       (at most :data:`_MAX_REPR_DOUBLINGS` times) and both preconditions
       are checked again. So the depth returned is past both limits, but
       not necessarily one level past ``repr``'s. ``repr``'s limit is not
       bisected: on CPython 3.14.3 a ``repr`` near its limit took about a
       second, so a bisection took about 17 s (BR-022 evidence, P4).

    Why a depth that fails here also fails where the SDK renders the value:
    the test calls this function, and later drives the code under test from
    the same test frame. ``loop._serialize_tool_output`` runs inside the
    loop's generator (test -> ``run`` -> ``_serialize_tool_output`` ->
    ``dumps_for_model``), and the Runner's ``_bounded_repr`` inside
    ``AgentRunner.run`` (test -> ``run`` -> ``_tool_invocation_payload`` ->
    ``_bounded_repr``), each at least as deep on the stack as this frame.
    The recursion budget only shrinks as the stack grows; with the
    containment removed (M3), I1 and I2 raised at the real sites on
    3.11.15, 3.13.2 and 3.14.3; 3.12 was not measured.

    What this returns depends on the interpreter, the platform and the stack
    the calling test runs on, so no assertion relies on a figure. For scale
    (macOS; BR-022's and its sentinel's runs, in different harnesses, where
    each figure moved by up to 10 levels): I1 (dict chain, without
    ``include_repr``) got about 950 levels on CPython 3.11.15, about 9,980 on
    3.13.2 and about 61,470 on 3.14.3; I2 (list chain, with ``include_repr``)
    got about 1,900 (one doubling), about 9,980 (none) and about 104,500
    (none). 3.12 and Linux were not measured; the CI run is their first.

    Deep values built here are freed before this returns and never reach an
    ``assert`` or ``repr`` outside the calls being measured.

    Args:
        include_repr: Also require ``repr`` to fail at the returned depth.
        shape: Builds the value for a depth, without recursion
            (:func:`dict_chain` or :func:`list_chain`).

    Returns:
        The depth to build the unrenderable tool result with.
    """
    lo, hi = 1, 128
    while True:
        try:
            dumps_for_model(shape(hi), default=str)
        except RecursionError:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            dumps_for_model(shape(mid), default=str)
        except RecursionError:
            hi = mid
        else:
            lo = mid
    depth = lo + 1
    for _ in range(_MAX_REPR_DOUBLINGS + 1):
        try:
            dumps_for_model(shape(depth), default=str)
        except RecursionError:
            pass
        else:
            pytest.fail(
                f"precondition failed: dumps_for_model rendered a value nested {depth} levels "
                "in this frame although the search found it could not; see BR-022 plan P4"
            )
        if not include_repr:
            return depth
        try:
            repr(shape(depth))
        except RecursionError:
            return depth
        depth *= 2
    pytest.fail(
        f"precondition failed: repr still rendered a value nested {depth // 2} levels in this "
        f"frame after {_MAX_REPR_DOUBLINGS} doublings; see BR-022 plan P4"
    )


__all__ = [
    "DriftsOnceFakeLLM",
    "FakeLLMClient",
    "FakeTool",
    "dict_chain",
    "list_chain",
    "make_multi_tool_response",
    "make_response",
    "make_stream_chunks",
    "measured_unrenderable_depth",
]
