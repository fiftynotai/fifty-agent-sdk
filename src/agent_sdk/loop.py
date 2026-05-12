"""ReACT loop entry point.

:class:`AgentLoop` is the orchestrator that drives a Thought-Action-Observation
cycle against a pluggable LLM, parser, tool registry, and prompt set. It is
the bridge that turns the four foundation layers (BR-003 prompts, BR-004
tools, BR-005 parser, BR-003 LLM) into a single async iterator of typed
:class:`agent_sdk.streaming.AgentEvent` values.

Statelessness
    Every :meth:`AgentLoop.run` call is its own scoped iteration. The
    loop holds no state across calls — conversation persistence and
    retry orchestration are a higher Runner's job (see BR-007).

System prompt snapshot
    The system prompt is constructed ONCE at :meth:`AgentLoop.run` start
    from a snapshot of :meth:`agent_sdk.tools.registry.Registry.list`.
    Tools registered AFTER ``run()`` begins are NOT visible to the
    model. Dynamic-tool consumers must rebuild the loop.

Cancellation
    :class:`asyncio.CancelledError` propagates untouched. The most
    recent tool call may still be running depending on its own
    cancellation discipline (see the
    :class:`agent_sdk.tools.protocol.Tool` contract).

Streaming semantics
    When ``stream=True``, :class:`agent_sdk.streaming.TokenEvent` deltas
    are emitted ONLY for the terminal final answer. Intermediate
    thought/action iterations are accumulated fully before any event
    is emitted — the parser requires the complete structured completion
    to disambiguate ``tool`` versus ``final``. This is a deliberate
    contract trade-off.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Final, TypeVar
from uuid import uuid4

import structlog

from agent_sdk.errors import LLMError, ParserError, ToolNotFound, ToolTimeout
from agent_sdk.llm.protocol import LLMClient
from agent_sdk.llm.types import ChatMessage, ChatRequest, ChatResponse
from agent_sdk.parser.base import FinalAnswer, Parser
from agent_sdk.prompts import PromptSections, render_system_prompt
from agent_sdk.safety import SafetyConfig
from agent_sdk.streaming import (
    ActionEvent,
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ThoughtEvent,
    TokenEvent,
    ToolFailedEvent,
    ToolStartedEvent,
)
from agent_sdk.tools.protocol import Tool
from agent_sdk.tools.registry import Registry

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger. INFO at iteration boundaries; DEBUG per sub-event."""

_AgentEventT = TypeVar(
    "_AgentEventT",
    ThoughtEvent,
    ActionEvent,
    ToolStartedEvent,
    ObservationEvent,
    ToolFailedEvent,
    TokenEvent,
    FinalEvent,
    ErrorEvent,
)
"""Constrained TypeVar over the concrete event classes emitted by the loop.

Excludes :class:`agent_sdk.streaming.ToolProgressEvent` because the v1 loop
does not emit it (see :mod:`agent_sdk.streaming` for the reservation rationale).
"""


def _render_tool_descriptions(tools: list[Tool]) -> str:
    """Format a snapshot of registered tools for the system prompt.

    Output is deterministic — same input always produces the same string.
    Empty list returns ``""`` so :func:`render_system_prompt` will omit
    the tool section entirely.

    Format::

        - <name>: <description>
          args: <JSON of schema.properties with sorted keys>

    Args:
        tools: Snapshot of registered tools from
            :meth:`agent_sdk.tools.registry.Registry.list`.

    Returns:
        A formatted block describing each tool, or ``""`` for an empty
        list.
    """
    if not tools:
        return ""
    lines: list[str] = []
    for tool in tools:
        args_json = json.dumps(tool.schema.properties, sort_keys=True)
        lines.append(f"- {tool.name}: {tool.description}\n  args: {args_json}")
    return "\n".join(lines)


def _serialize_tool_output(output: Any) -> str:
    """Convert a tool's return value into a string suitable for a ``role="tool"`` message.

    Strategy:

    1. If ``output`` is already a string, return it unchanged.
    2. Otherwise attempt :func:`json.dumps` with ``default=str`` to handle
       common non-serializable types (datetimes, paths, custom classes).
    3. As a last resort fall back to :func:`repr`.

    The shape of the tool message content is provider-tolerated; both
    JSON-mode and prose-mode LLMs accept a stringified payload.

    Args:
        output: Anything a tool may return as :attr:`agent_sdk.tools.
            protocol.ToolResult.output`.

    Returns:
        A string representation of the output, guaranteed not to raise.
    """
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except (TypeError, ValueError):
        return repr(output)


async def _accumulate_stream(
    llm: LLMClient, request: ChatRequest
) -> tuple[str, list[str]]:
    """Drain :meth:`LLMClient.stream` into the joined completion plus per-chunk deltas.

    Stops as soon as a chunk reports a non-``"in_progress"`` finish reason.
    Empty deltas are skipped so they cannot leak into
    :class:`agent_sdk.streaming.TokenEvent` replay.

    Args:
        llm: The :class:`agent_sdk.llm.protocol.LLMClient` to drive.
        request: The :class:`agent_sdk.llm.types.ChatRequest` to stream.

    Returns:
        A two-tuple ``(completion, deltas)`` where ``completion`` is the
        concatenation of every non-empty delta and ``deltas`` is the list
        of those deltas in order. Both are usable for downstream parsing
        and :class:`agent_sdk.streaming.TokenEvent` replay.

    Raises:
        agent_sdk.errors.LLMError: Forwarded from the underlying stream.
    """
    deltas: list[str] = []
    async for chunk in llm.stream(request):
        if chunk.message.content:
            deltas.append(chunk.message.content)
        if chunk.finish_reason != "in_progress":
            break
    return "".join(deltas), deltas


class AgentLoop:
    """Production ReACT loop.

    The loop drives a Thought-Action-Observation cycle, emits a fully-typed
    event stream, and terminates cleanly under every documented failure
    mode (LLM error, parser error, tool failure, iteration cap). See the
    module docstring for the cross-cutting contracts.

    Every :meth:`run` ends with exactly one :class:`agent_sdk.streaming.
    FinalEvent`. Consumers can rely on this as their "iteration done"
    signal.

    Args:
        llm: Pluggable :class:`agent_sdk.llm.protocol.LLMClient`
            implementation. Must satisfy the protocol's error contract
            (raise :class:`agent_sdk.errors.LLMError` on any provider
            failure).
        registry: :class:`agent_sdk.tools.registry.Registry` of available
            tools. A snapshot of :meth:`Registry.list` is taken at
            construction and embedded in the system prompt.
        parser: :class:`agent_sdk.parser.base.Parser` matching the LLM's
            output format.
        prompts: :class:`agent_sdk.prompts.PromptSections` slots. The
            loop OWNS the ``tool_descriptions`` slot (it is rebuilt from
            the registry snapshot); the caller supplies the rest. If
            ``prompts.output_format`` is empty and ``output_format`` is
            also empty, no output-format section appears in the system
            prompt.
        safety: :class:`agent_sdk.safety.SafetyConfig` with iteration cap,
            tool timeout, and fallback message.
        model: Model identifier embedded in every
            :class:`agent_sdk.llm.types.ChatRequest`.
        stream: If ``True``, use :meth:`LLMClient.stream` and emit
            :class:`agent_sdk.streaming.TokenEvent` for the FINAL answer
            only. Default ``False`` — uses :meth:`LLMClient.complete`,
            no token events.
        output_format: Optional override for the system prompt's
            ``output_format`` slot. When non-empty, replaces whatever is
            in ``prompts.output_format``. Useful for callers that want
            to pin the parser-aligned format without rebuilding
            ``prompts``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        # TODO(BR-008+): Widen `registry` to a RegistryProtocol once non-in-process
        # providers (MCP, RPC) land — current type pins us to the concrete class.
        registry: Registry,
        parser: Parser,
        prompts: PromptSections,
        safety: SafetyConfig,
        model: str,
        stream: bool = False,
        output_format: str = "",
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._parser = parser
        self._safety = safety
        self._model = model
        self._stream = stream
        # Snapshot tool descriptions ONCE; subsequent registry mutations are
        # invisible to this loop instance. Documented behavior.
        self._system_prompt = self._build_system_prompt(prompts, output_format)

    def _build_system_prompt(
        self, prompts: PromptSections, output_format: str
    ) -> str:
        """Render the system prompt from a registry snapshot and the provided slots."""
        tool_block = _render_tool_descriptions(self._registry.list())
        chosen_format = output_format or prompts.output_format
        composed = PromptSections(
            persona=prompts.persona,
            tool_descriptions=tool_block,
            output_format=chosen_format,
            additional_context=prompts.additional_context,
        )
        return render_system_prompt(composed)

    def _build_request(self, messages: list[ChatMessage]) -> ChatRequest:
        """Wrap the working message list in a :class:`ChatRequest` for this loop's model."""
        return ChatRequest(messages=list(messages), model=self._model)

    async def run(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[AgentEvent]:
        """Drive a single ReACT run.

        Yields events as they happen. Always terminates with a
        :class:`agent_sdk.streaming.FinalEvent`. Recoverable failures
        (``ToolNotFound``, ``ToolTimeout``, ``ToolResult(is_error=True)``)
        are emitted as :class:`agent_sdk.streaming.ToolFailedEvent` and
        the loop continues so the model can reason about the failure.
        Non-recoverable failures (``LLMError``, ``ParserError``, iteration
        cap exhaustion) emit an :class:`agent_sdk.streaming.ErrorEvent`
        followed by a fallback :class:`agent_sdk.streaming.FinalEvent`,
        then return.

        :class:`asyncio.CancelledError` and any
        :class:`agent_sdk.errors.AgentSdkError` subclass not explicitly
        handled propagate out of the generator.

        Args:
            messages: Caller's conversation messages. Not mutated. The
                loop builds a private working list prefixed with the
                system message it constructed at ``__init__`` time.

        Yields:
            :class:`agent_sdk.streaming.AgentEvent` values in monotonic
            ``sequence`` order. The terminal event is always a
            :class:`agent_sdk.streaming.FinalEvent`.
        """
        run_id = uuid4().hex
        sequence_box: list[int] = [0]
        _log.info(
            "agent_loop_started",
            run_id=run_id,
            max_iterations=self._safety.max_iterations,
            stream=self._stream,
        )

        working: list[ChatMessage] = [
            ChatMessage(role="system", content=self._system_prompt),
            *messages,
        ]

        iteration = 0
        while iteration < self._safety.max_iterations:
            iteration += 1
            _log.info("iteration_started", iteration=iteration, run_id=run_id)

            # ----- 1. Build request and call LLM ----------------------------
            request = self._build_request(working)
            try:
                completion, deltas = await self._call_llm(request)
            except LLMError as exc:
                yield self._make_event(
                    ErrorEvent,
                    sequence_box,
                    error_type="LLMError",
                    message=exc.message,
                    context=dict(exc.context),
                )
                yield self._make_event(
                    FinalEvent,
                    sequence_box,
                    text=self._safety.fallback_message,
                )
                _log.info(
                    "agent_loop_completed",
                    iterations=iteration,
                    run_id=run_id,
                    terminated_by="llm_error",
                )
                return

            # ----- 2. Parse the completion ---------------------------------
            try:
                parsed = self._parser.parse(completion)
            except ParserError as exc:
                yield self._make_event(
                    ErrorEvent,
                    sequence_box,
                    error_type="ParserError",
                    message=exc.message,
                    context=dict(exc.context),
                )
                yield self._make_event(
                    FinalEvent,
                    sequence_box,
                    text=self._safety.fallback_message,
                )
                _log.info(
                    "agent_loop_completed",
                    iterations=iteration,
                    run_id=run_id,
                    terminated_by="parser_error",
                )
                return

            _log.debug("parsed_result", kind=parsed.kind, run_id=run_id)

            # ----- 3. Branch on parse result -------------------------------
            if isinstance(parsed, FinalAnswer):
                yield self._make_event(
                    ThoughtEvent, sequence_box, text=parsed.thought
                )
                if self._stream:
                    for delta in deltas:
                        yield self._make_event(
                            TokenEvent, sequence_box, text=delta
                        )
                yield self._make_event(
                    FinalEvent, sequence_box, text=parsed.content
                )
                _log.info(
                    "agent_loop_completed",
                    iterations=iteration,
                    run_id=run_id,
                    terminated_by="final_answer",
                )
                return

            # parsed is a ThoughtAction (narrowed by isinstance check above).
            yield self._make_event(
                ThoughtEvent, sequence_box, text=parsed.thought
            )
            yield self._make_event(
                ActionEvent,
                sequence_box,
                tool_name=parsed.tool_call.name,
                args=dict(parsed.tool_call.args),
            )

            # Append the assistant turn to history before emitting ToolStarted, so the
            # subsequent tool reply sits on top of the model's reasoning turn.
            working.append(ChatMessage(role="assistant", content=completion))

            # ----- 4. Invoke tool ------------------------------------------
            call_id = uuid4().hex
            tool_name = parsed.tool_call.name
            yield self._make_event(
                ToolStartedEvent,
                sequence_box,
                tool_name=tool_name,
                call_id=call_id,
            )
            _log.debug(
                "tool_invoked",
                name=tool_name,
                call_id=call_id,
                run_id=run_id,
            )

            try:
                tool_result = await self._registry.invoke(
                    tool_name,
                    dict(parsed.tool_call.args),
                    timeout=self._safety.tool_timeout_seconds,
                )
            except ToolNotFound as exc:
                yield self._make_event(
                    ToolFailedEvent,
                    sequence_box,
                    tool_name=tool_name,
                    call_id=call_id,
                    error=f"ToolNotFound: {exc.message}",
                )
                working.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call_id,
                        name=tool_name,
                        content=(
                            f"ToolNotFound: tool '{tool_name}' is not registered."
                        ),
                    )
                )
                continue
            except ToolTimeout as exc:
                yield self._make_event(
                    ToolFailedEvent,
                    sequence_box,
                    tool_name=tool_name,
                    call_id=call_id,
                    error=f"ToolTimeout: {exc.message}",
                )
                working.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call_id,
                        name=tool_name,
                        content=f"ToolTimeout: {exc.message}",
                    )
                )
                continue

            # ----- 5. Classify ToolResult ----------------------------------
            if tool_result.is_error:
                error_text = (
                    tool_result.error
                    if tool_result.error is not None
                    else "tool reported error with no message"
                )
                yield self._make_event(
                    ToolFailedEvent,
                    sequence_box,
                    tool_name=tool_name,
                    call_id=call_id,
                    error=error_text,
                )
                working.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call_id,
                        name=tool_name,
                        content=f"Tool error: {error_text}",
                    )
                )
            else:
                yield self._make_event(
                    ObservationEvent,
                    sequence_box,
                    tool_name=tool_name,
                    call_id=call_id,
                    result=tool_result,
                )
                working.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call_id,
                        name=tool_name,
                        content=_serialize_tool_output(tool_result.output),
                    )
                )
            # Loop continues.

        # ----- 6. Iteration cap reached ------------------------------------
        _log.warning(
            "max_iterations_exceeded",
            iteration=iteration,
            max_iterations=self._safety.max_iterations,
            run_id=run_id,
        )
        yield self._make_event(
            ErrorEvent,
            sequence_box,
            error_type="MaxIterationsExceeded",
            message=(
                f"loop did not terminate within "
                f"{self._safety.max_iterations} iterations"
            ),
            context={
                "max_iterations": self._safety.max_iterations,
                "iteration_count": iteration,
            },
        )
        yield self._make_event(
            FinalEvent, sequence_box, text=self._safety.fallback_message
        )
        _log.info(
            "agent_loop_completed",
            iterations=iteration,
            run_id=run_id,
            terminated_by="safety_cap",
        )

    async def _call_llm(self, request: ChatRequest) -> tuple[str, list[str]]:
        """Call the LLM in either streaming or non-streaming mode.

        Returns a tuple ``(completion, deltas)``. In non-stream mode
        ``deltas`` is empty; in stream mode it carries the per-chunk
        deltas accumulated by :func:`_accumulate_stream`.

        Args:
            request: The :class:`ChatRequest` to issue.

        Returns:
            A two-tuple of the full completion string and the deltas list.

        Raises:
            agent_sdk.errors.LLMError: Forwarded from the underlying client.
        """
        if self._stream:
            return await _accumulate_stream(self._llm, request)
        response: ChatResponse = await self._llm.complete(request)
        return response.message.content, []

    def _make_event(
        self,
        event_cls: type[_AgentEventT],
        sequence_box: list[int],
        **payload: Any,
    ) -> _AgentEventT:
        """Construct an event with the next sequence number and a fresh UTC timestamp.

        Mutates ``sequence_box[0]`` so the next call sees the incremented
        counter. Returns the constructed event.
        """
        event = event_cls(
            sequence=sequence_box[0],
            timestamp=datetime.now(UTC),
            **payload,
        )
        sequence_box[0] += 1
        _log.debug(
            "event_emitted",
            event_type=event.event_type,
            sequence=event.sequence,
        )
        return event


__all__ = ["AgentLoop"]
