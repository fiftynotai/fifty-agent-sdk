"""ReACT loop entry point.

:class:`AgentLoop` is the orchestrator that drives a Thought-Action-Observation
cycle against a pluggable LLM, parser, tool registry, and prompt set. It is
the bridge that turns the four foundation layers (BR-003 prompts, BR-004
tools, BR-005 parser, BR-003 LLM) into a single async iterator of typed
:class:`fifty_agent_sdk.streaming.AgentEvent` values.

Statelessness
    Every :meth:`AgentLoop.run` call is its own scoped iteration. The
    loop holds no state across calls — conversation persistence and
    retry orchestration are a higher Runner's job (see BR-007).

System prompt snapshot
    The system prompt is constructed ONCE at :meth:`AgentLoop.run` start
    from a snapshot of :meth:`fifty_agent_sdk.tools.registry.Registry.list`.
    Tools registered AFTER ``run()`` begins are NOT visible to the
    model. Dynamic-tool consumers must rebuild the loop.

Cancellation
    :class:`asyncio.CancelledError` propagates untouched. The most
    recent tool call may still be running depending on its own
    cancellation discipline (see the
    :class:`fifty_agent_sdk.tools.protocol.Tool` contract).

Streaming semantics
    When ``stream=True``, :class:`fifty_agent_sdk.streaming.TokenEvent` deltas
    are emitted ONLY for the terminal final answer. Intermediate
    thought/action iterations are accumulated fully before any event
    is emitted — the parser requires the complete structured completion
    to disambiguate ``tool`` versus ``final``. This is a deliberate
    contract trade-off.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Final, Literal, TypeVar
from uuid import uuid4

import structlog

from fifty_agent_sdk.errors import LLMError, ParserError, ToolNotFound, ToolTimeout
from fifty_agent_sdk.llm.protocol import LLMClient
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest, ChatResponse, Usage
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.observability.hooks import invoke_hook
from fifty_agent_sdk.parser.base import FinalAnswer, Parser, ParseResult
from fifty_agent_sdk.prompts import PromptSections, render_system_prompt
from fifty_agent_sdk.safety import SafetyConfig
from fifty_agent_sdk.streaming import (
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
from fifty_agent_sdk.tools.protocol import Tool
from fifty_agent_sdk.tools.registry import Registry

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

Excludes :class:`fifty_agent_sdk.streaming.ToolProgressEvent` because the v1 loop
does not emit it (see :mod:`fifty_agent_sdk.streaming` for the reservation rationale).
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
            :meth:`fifty_agent_sdk.tools.registry.Registry.list`.

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
        output: Anything a tool may return as :attr:`fifty_agent_sdk.tools.
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


async def _accumulate_stream(llm: LLMClient, request: ChatRequest) -> tuple[str, list[str]]:
    """Drain :meth:`LLMClient.stream` into the joined completion plus per-chunk deltas.

    Stops as soon as a chunk reports a non-``"in_progress"`` finish reason.
    Empty deltas are skipped so they cannot leak into
    :class:`fifty_agent_sdk.streaming.TokenEvent` replay.

    Args:
        llm: The :class:`fifty_agent_sdk.llm.protocol.LLMClient` to drive.
        request: The :class:`fifty_agent_sdk.llm.types.ChatRequest` to stream.

    Returns:
        A two-tuple ``(completion, deltas)`` where ``completion`` is the
        concatenation of every non-empty delta and ``deltas`` is the list
        of those deltas in order. Both are usable for downstream parsing
        and :class:`fifty_agent_sdk.streaming.TokenEvent` replay.

    Raises:
        fifty_agent_sdk.errors.LLMError: Forwarded from the underlying stream.
    """
    deltas: list[str] = []
    async for chunk in llm.stream(request):
        if chunk.message.content:
            deltas.append(chunk.message.content)
        if chunk.finish_reason != "in_progress":
            break
    return "".join(deltas), deltas


def _synthesize_stream_response(completion: str) -> ChatResponse:
    """Build a :class:`ChatResponse` standing in for a streamed completion.

    A stream has no single :class:`ChatResponse` — :func:`_accumulate_stream`
    drains the per-chunk responses and discards them. The ``on_llm_call``
    observability hook is contracted to always receive a real
    :class:`ChatResponse`, so in stream mode the loop synthesizes one from
    the accumulated ``completion``: an ``assistant`` :class:`ChatMessage`
    carrying the full text, zero-filled :class:`Usage` (a stream provides no
    aggregate token counts here), and ``finish_reason="stop"``.

    Args:
        completion: The full accumulated completion string.

    Returns:
        A synthesized :class:`ChatResponse` for the ``on_llm_call`` hook.
    """
    return ChatResponse(
        message=ChatMessage(role="assistant", content=completion),
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        finish_reason="stop",
    )


class AgentLoop:
    """Production ReACT loop.

    The loop drives a Thought-Action-Observation cycle, emits a fully-typed
    event stream, and terminates cleanly under every documented failure
    mode (LLM error, parser error, tool failure, iteration cap). See the
    module docstring for the cross-cutting contracts.

    Every :meth:`run` ends with exactly one :class:`fifty_agent_sdk.streaming.
    FinalEvent`. Consumers can rely on this as their "iteration done"
    signal.

    Args:
        llm: Pluggable :class:`fifty_agent_sdk.llm.protocol.LLMClient`
            implementation. Must satisfy the protocol's error contract
            (raise :class:`fifty_agent_sdk.errors.LLMError` on any provider
            failure).
        registry: :class:`fifty_agent_sdk.tools.registry.Registry` of available
            tools. A snapshot of :meth:`Registry.list` is taken at
            construction and embedded in the system prompt.
        parser: :class:`fifty_agent_sdk.parser.base.Parser` matching the LLM's
            output format.
        prompts: :class:`fifty_agent_sdk.prompts.PromptSections` slots. The
            loop OWNS the ``tool_descriptions`` slot (it is rebuilt from
            the registry snapshot); the caller supplies the rest. If
            ``prompts.output_format`` is empty and ``output_format`` is
            also empty, no output-format section appears in the system
            prompt.
        safety: :class:`fifty_agent_sdk.safety.SafetyConfig` with iteration cap,
            tool timeout, and fallback message.
        model: Model identifier embedded in every
            :class:`fifty_agent_sdk.llm.types.ChatRequest`.
        stream: If ``True``, use :meth:`LLMClient.stream` and emit
            :class:`fifty_agent_sdk.streaming.TokenEvent` for the FINAL answer
            only. Default ``False`` — uses :meth:`LLMClient.complete`,
            no token events.
        output_format: Optional override for the system prompt's
            ``output_format`` slot. When non-empty, replaces whatever is
            in ``prompts.output_format``. Useful for callers that want
            to pin the parser-aligned format without rebuilding
            ``prompts``.
        tool_message_role: Wire-format role used for the synthetic message
            the loop appends after a tool invocation. The default
            ``"tool"`` preserves the OpenAI tool-role wire format and is
            correct for any provider whose chat-message schema accepts
            ``role="tool"`` with ``tool_call_id`` and ``name`` fields. Set
            to ``"user"`` or ``"assistant"`` for providers whose
            chat-message schema only accepts ``system | user | assistant``
            (e.g. some OpenAI-compatible gateways/proxies, which
            return HTTP 500 on ``role="tool"``). In non-``"tool"`` mode
            the synthetic message carries the tool name inline in its
            content (so the model retains identity context) and omits
            ``tool_call_id`` / ``name``.
        hooks: Optional :class:`fifty_agent_sdk.observability.Hooks`. When set,
            the loop fires the two Loop-tier hooks — ``on_iteration`` once
            per ReACT iteration and ``on_llm_call`` once per successful LLM
            call. The other five hooks are Runner-tier; wire the SAME
            :class:`Hooks` instance into :class:`fifty_agent_sdk.runner.
            AgentRunner` as well so all seven fire (the loop does NOT
            receive ``hooks`` from a Runner — the two are wired
            independently by the consumer). A raising hook never aborts the
            loop. When ``None`` (default), hook dispatch is zero-overhead.
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
        tool_message_role: Literal["tool", "user", "assistant"] = "tool",
        hooks: Hooks | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._parser = parser
        self._safety = safety
        self._model = model
        self._stream = stream
        self._tool_message_role: Literal["tool", "user", "assistant"] = tool_message_role
        self._hooks = hooks
        # Snapshot tool descriptions ONCE; subsequent registry mutations are
        # invisible to this loop instance. Documented behavior.
        self._system_prompt = self._build_system_prompt(prompts, output_format)

    def _build_system_prompt(self, prompts: PromptSections, output_format: str) -> str:
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
        self, messages: list[ChatMessage], *, session_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Drive a single ReACT run.

        Yields events as they happen. Always terminates with a
        :class:`fifty_agent_sdk.streaming.FinalEvent`. Recoverable failures
        (``ToolNotFound``, ``ToolTimeout``, ``ToolResult(is_error=True)``)
        are emitted as :class:`fifty_agent_sdk.streaming.ToolFailedEvent` and
        the loop continues so the model can reason about the failure.
        Non-recoverable failures (``LLMError``, ``ParserError``, iteration
        cap exhaustion) emit an :class:`fifty_agent_sdk.streaming.ErrorEvent`
        followed by a fallback :class:`fifty_agent_sdk.streaming.FinalEvent`,
        then return.

        :class:`asyncio.CancelledError` and any
        :class:`fifty_agent_sdk.errors.AgentSdkError` subclass not explicitly
        handled propagate out of the generator.

        Args:
            messages: Caller's conversation messages. Not mutated. The
                loop builds a private working list prefixed with the
                system message it constructed at ``__init__`` time.
            session_id: Opaque session identifier forwarded verbatim to the
                Loop-tier observability hooks (``on_iteration`` and
                ``on_llm_call``). ``None`` — the default — when the loop is
                driven directly without an :class:`fifty_agent_sdk.runner.
                AgentRunner`; a Runner passes the conversation's
                ``session_id`` down. The hook contract types this parameter
                as ``str | None`` for exactly this reason.

        Yields:
            :class:`fifty_agent_sdk.streaming.AgentEvent` values in monotonic
            ``sequence`` order. The terminal event is always a
            :class:`fifty_agent_sdk.streaming.FinalEvent`.
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

        # BR-036 require-tool-before-final state, scoped to THIS run() call.
        # `tool_invoked_this_run` starts False so a brand-new conversational
        # turn cannot inherit "a tool already ran" from a prior turn — the
        # current turn must invoke a tool of its own to satisfy the guard.
        # `tool_final_forced` makes the force-reconsider strictly one-shot per
        # run (mirrors the per-iteration parser-retry budget): after one forced
        # reconsideration the next `final` is accepted at the loop layer.
        tool_invoked_this_run = False
        tool_final_forced = False

        iteration = 0
        while iteration < self._safety.max_iterations:
            iteration += 1
            _log.info("iteration_started", iteration=iteration, run_id=run_id)
            await self._invoke_hook(
                "on_iteration",
                self._hooks.on_iteration if self._hooks is not None else None,
                session_id,
                iteration,
            )

            # ----- 1+2. LLM call and parse, with BR-018 one-shot retry -----
            # Retry budget is PER-iteration (not cumulative): every outer
            # iteration starts fresh so multi-tool runs cannot silently
            # exhaust a global budget on the first drift. The inner loop
            # runs once on the success path, twice when a single parse
            # failure triggers the format-reminder retry.
            parser_retries_this_iteration: int = 0
            completion: str = ""
            deltas: list[str] = []
            parsed: ParseResult
            while True:
                # ----- 1. Build request and call LLM ------------------------
                request = self._build_request(working)
                try:
                    llm_t0 = time.perf_counter()
                    completion, deltas, response = await self._call_llm(request)
                    llm_duration_ms = (time.perf_counter() - llm_t0) * 1000
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

                # `on_llm_call` fires once per SUCCESSFUL LLM call only — an
                # LLMError surfaces via the Runner-tier `on_error` hook
                # (driven by the ErrorEvent above), so firing here with no
                # response would be ill-defined. In stream mode `_call_llm`
                # returns `response is None` (a stream has no single
                # response object); synthesize a minimal ChatResponse from
                # the accumulated completion so the hook always receives a
                # real ChatResponse. On a BR-018 retry this hook fires
                # TWICE per outer iteration (once per inner LLM call) by
                # design.
                if self._hooks is not None and self._hooks.on_llm_call is not None:
                    llm_response = (
                        response
                        if response is not None
                        else _synthesize_stream_response(completion)
                    )
                    await self._invoke_hook(
                        "on_llm_call",
                        self._hooks.on_llm_call,
                        session_id,
                        request,
                        llm_response,
                        llm_duration_ms,
                    )

                # ----- 2. Parse the completion -----------------------------
                try:
                    parsed = self._parser.parse(completion)
                except ParserError as exc:
                    if self._safety.parser_retry_enabled and parser_retries_this_iteration < 1:
                        # BR-018 one-shot retry: echo the malformed
                        # completion back as an assistant turn (so the
                        # model sees what it actually said) and inject
                        # the reminder as a user turn, then re-enter the
                        # inner loop. No events are emitted for the
                        # failed attempt — the iteration is still "in
                        # progress" from the consumer's view.
                        _log.info(
                            "parser_retry_triggered",
                            iteration=iteration,
                            error_phase=exc.context.get("error_phase"),
                            run_id=run_id,
                        )
                        working.append(ChatMessage(role="assistant", content=completion))
                        working.append(
                            ChatMessage(
                                role="user",
                                content=self._safety.parser_retry_reminder,
                            )
                        )
                        parser_retries_this_iteration += 1
                        continue
                    # Retry disabled OR per-iteration budget exhausted:
                    # preserve the pre-BR-018 terminal ParserError shape
                    # verbatim — ErrorEvent + fallback FinalEvent + return.
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
                # Parse succeeded — leave the inner retry loop and
                # continue with the existing branching logic below.
                break

            _log.debug("parsed_result", kind=parsed.kind, run_id=run_id)

            # ----- 3. Branch on parse result -------------------------------
            if isinstance(parsed, FinalAnswer):
                # BR-036 force-reconsider: if the consumer opted in, the model
                # tried to finalize without invoking ANY tool this run, and we
                # have not already forced once, do NOT emit the final. Instead
                # echo the completion back as an assistant turn, inject the
                # permissive tool-required reminder as a user turn, mark the
                # one-shot budget spent, and `continue` — re-entering the outer
                # iteration (still bounded by `max_iterations`). This mirrors
                # the BR-018 parser-retry mechanism. When the flag is OFF (the
                # default), or a tool already ran, or the budget is spent, this
                # block is skipped entirely and the original emit-and-return
                # logic below runs byte-for-byte unchanged.
                if (
                    self._safety.require_tool_before_final
                    and not tool_invoked_this_run
                    and not tool_final_forced
                ):
                    _log.info(
                        "tool_required_force_triggered",
                        iteration=iteration,
                        run_id=run_id,
                    )
                    working.append(ChatMessage(role="assistant", content=completion))
                    working.append(
                        ChatMessage(
                            role="user",
                            content=self._safety.tool_required_reminder,
                        )
                    )
                    # Re-anchor the ORIGINAL question as the model's latest
                    # message. Without this the reminder is the most recent
                    # user turn, and the model tends to "reply to the reminder"
                    # with a meta-acknowledgment ("Thank you, I will use the
                    # tool…") as its next final instead of acting. Re-appending
                    # the current turn's last user message makes the question —
                    # not the reminder — the thing the model must answer next.
                    # The reminder still precedes it, so it frames HOW to act
                    # (search-then-answer, or direct-answer a greeting). When
                    # the caller sent no user message (degenerate), nothing is
                    # re-anchored and the reminder alone drives the retry.
                    last_user_message = next(
                        (m for m in reversed(messages) if m.role == "user"), None
                    )
                    if last_user_message is not None:
                        working.append(ChatMessage(role="user", content=last_user_message.content))
                    tool_final_forced = True
                    continue

                yield self._make_event(ThoughtEvent, sequence_box, text=parsed.thought)
                if self._stream:
                    for delta in deltas:
                        yield self._make_event(TokenEvent, sequence_box, text=delta)
                yield self._make_event(
                    FinalEvent,
                    sequence_box,
                    text=parsed.content,
                    raw_completion=completion,
                )
                _log.info(
                    "agent_loop_completed",
                    iterations=iteration,
                    run_id=run_id,
                    terminated_by="final_answer",
                )
                return

            # parsed is a ThoughtAction (narrowed by isinstance check above).
            yield self._make_event(ThoughtEvent, sequence_box, text=parsed.thought)
            yield self._make_event(
                ActionEvent,
                sequence_box,
                tool_name=parsed.tool_call.name,
                args=dict(parsed.tool_call.args),
            )

            # Append the assistant turn to history before emitting ToolStarted, so the
            # subsequent tool reply sits on top of the model's reasoning turn.
            working.append(ChatMessage(role="assistant", content=completion))

            # BR-036: a tool is being dispatched this run, so a later `final`
            # is now grounded (from the loop's structural view) and must NOT
            # trigger the force-reconsider guard.
            tool_invoked_this_run = True

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
                    self._build_tool_message(
                        tool_name=tool_name,
                        call_id=call_id,
                        content_for_tool_role=(
                            f"ToolNotFound: tool '{tool_name}' is not registered."
                        ),
                        content_for_other_role=(
                            f"Tool {tool_name} failed: "
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
                    self._build_tool_message(
                        tool_name=tool_name,
                        call_id=call_id,
                        content_for_tool_role=f"ToolTimeout: {exc.message}",
                        content_for_other_role=(
                            f"Tool {tool_name} failed: ToolTimeout: {exc.message}"
                        ),
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
                    self._build_tool_message(
                        tool_name=tool_name,
                        call_id=call_id,
                        content_for_tool_role=f"Tool error: {error_text}",
                        content_for_other_role=(f"Tool {tool_name} failed: {error_text}"),
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
                    self._build_tool_message(
                        tool_name=tool_name,
                        call_id=call_id,
                        content_for_tool_role=_serialize_tool_output(tool_result.output),
                        content_for_other_role=(
                            f"Tool {tool_name} returned: "
                            f"{_serialize_tool_output(tool_result.output)}"
                        ),
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
            message=(f"loop did not terminate within {self._safety.max_iterations} iterations"),
            context={
                "max_iterations": self._safety.max_iterations,
                "iteration_count": iteration,
            },
        )
        yield self._make_event(FinalEvent, sequence_box, text=self._safety.fallback_message)
        _log.info(
            "agent_loop_completed",
            iterations=iteration,
            run_id=run_id,
            terminated_by="safety_cap",
        )

    def _build_tool_message(
        self,
        *,
        tool_name: str,
        call_id: str,
        content_for_tool_role: str,
        content_for_other_role: str,
    ) -> ChatMessage:
        """Build the synthetic post-tool message in the configured wire-format role.

        When :attr:`_tool_message_role` is the default ``"tool"``, the
        message uses the OpenAI tool-role envelope: ``role="tool"`` with
        ``tool_call_id`` and ``name`` populated and ``content_for_tool_role``
        as the body. When it is ``"user"`` or ``"assistant"`` (used for
        providers that reject ``role="tool"``), the envelope is collapsed
        to a plain ``user``/``assistant`` message whose content is
        ``content_for_other_role`` — the caller prefixes the tool name into
        that string so the model retains identity context without
        ``tool_call_id`` / ``name`` fields.

        Args:
            tool_name: The name of the tool that was invoked (only used in
                the ``"tool"`` role branch; the caller embeds it into
                ``content_for_other_role`` for the other branches).
            call_id: The synthesized tool-call identifier (only used in the
                ``"tool"`` role branch).
            content_for_tool_role: Body string for the ``role="tool"``
                envelope. MUST remain byte-identical to the previous
                implementation so default-path callers see no wire change.
            content_for_other_role: Body string for the
                ``role="user"`` / ``role="assistant"`` envelope. Carries the
                tool name inline since the envelope drops ``name``.

        Returns:
            A :class:`ChatMessage` ready to append to the loop's working
            message list.
        """
        if self._tool_message_role == "tool":
            return ChatMessage(
                role="tool",
                tool_call_id=call_id,
                name=tool_name,
                content=content_for_tool_role,
            )
        return ChatMessage(
            role=self._tool_message_role,
            content=content_for_other_role,
        )

    async def _call_llm(self, request: ChatRequest) -> tuple[str, list[str], ChatResponse | None]:
        """Call the LLM in either streaming or non-streaming mode.

        Returns a tuple ``(completion, deltas, response)``:

        * ``completion`` — the full completion string.
        * ``deltas`` — the per-chunk deltas. Empty in non-stream mode; the
          accumulated deltas in stream mode.
        * ``response`` — the underlying :class:`ChatResponse` in non-stream
          mode, or ``None`` in stream mode. A stream has no single response
          object — the per-chunk responses are drained by
          :func:`_accumulate_stream` and discarded. The ``None`` is surfaced
          (rather than synthesized here) so the caller can decide whether a
          synthesized response is needed; the ``on_llm_call`` fire site
          synthesizes one only when the hook is actually wired.

        Args:
            request: The :class:`ChatRequest` to issue.

        Returns:
            A three-tuple of the completion string, the deltas list, and the
            :class:`ChatResponse` (``None`` in stream mode).

        Raises:
            fifty_agent_sdk.errors.LLMError: Forwarded from the underlying client.
        """
        if self._stream:
            completion, deltas = await _accumulate_stream(self._llm, request)
            return completion, deltas, None
        response: ChatResponse = await self._llm.complete(request)
        return response.message.content, [], response

    async def _invoke_hook(self, hook_name: str, hook: Any, *args: Any) -> None:
        """Dispatch a single Loop-tier observability hook.

        A thin wrapper over :func:`fifty_agent_sdk.observability.hooks.invoke_hook`
        that keeps the loop's two fire points terse. ``hook`` is ``None``
        when no :class:`Hooks` is configured or the specific field is unset,
        in which case the delegate short-circuits with zero overhead. A
        raising hook is logged and swallowed by the delegate;
        :class:`asyncio.CancelledError` is re-raised untouched.

        Args:
            hook_name: Stable name of the hook for the failure log line.
            hook: The hook callable, or ``None`` for a no-op.
            *args: Positional arguments forwarded verbatim to the hook.
        """
        await invoke_hook(hook, hook_name, *args)

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
