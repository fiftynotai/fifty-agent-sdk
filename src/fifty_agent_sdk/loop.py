"""ReACT loop entry point.

:class:`AgentLoop` is the orchestrator that drives a Thought-Action-Observation
cycle against a pluggable LLM, parser, tool registry, and prompt set. It is
the bridge that turns the four foundation layers (BR-003 prompts, BR-004
tools, BR-005 parser, BR-003 LLM) into a single async iterator of typed
:class:`fifty_agent_sdk.streaming.AgentEvent` values.

Tool modes (FR-001)
    ``AgentLoop(tool_mode=...)`` takes a :class:`fifty_agent_sdk.tool_mode.
    ToolMode` (``JSON``, ``PROSE`` or ``NATIVE``) that sets the text parser,
    output format, tool-result role, tool declaration and parser-retry
    reminder together; see :mod:`fifty_agent_sdk.tool_mode` for the table and
    the conflict rule. Under ``NATIVE`` a tool call is only ever a structured
    ``tool_calls`` entry: text is always the final answer, so a
    ``role="tool"`` reply always follows an assistant turn whose
    ``tool_calls`` carry its id. When ``tool_mode`` is omitted the loop
    sends the same request bodies (same keys, values and JSON types) and
    system prompt as 1.7.0, including the four independent knobs; the one
    event-level change is ``ThoughtEvent.text`` on native tool turns (see
    CHANGELOG).

Request options (FR-002)
    ``AgentLoop(reasoning_effort=..., temperature=...)`` set request parameters
    on EVERY request the loop builds. All of them go through the single
    ``_build_request`` call site: tool steps, parser-retry and
    require-tool-before-final re-asks, native and streamed turns. With both
    kwargs omitted the loop sends the same request bodies as 1.8.0 (same keys,
    values and JSON types). The two are independent: setting
    ``reasoning_effort`` never drops or changes ``temperature``.

Statelessness
    Every :meth:`AgentLoop.run` call is its own scoped iteration. The
    loop holds no state across calls — conversation persistence and
    retry orchestration are a higher Runner's job (see BR-007).

System prompt snapshot
    The system prompt is constructed ONCE at ``AgentLoop.__init__``
    from a snapshot of :meth:`fifty_agent_sdk.tools.registry.Registry.list`.
    In the text modes (``ToolMode.JSON`` / ``ToolMode.PROSE``, or the legacy
    default), tools registered AFTER construction are NOT visible to the
    model. Dynamic-tool consumers must rebuild the loop.
    Under native-tools mode (``ToolMode.NATIVE``, or
    :attr:`fifty_agent_sdk.safety.SafetyConfig.native_tools_enabled` on the
    legacy path) the prompt-side tool block is suppressed and the OpenAI ``tools`` request
    param is rebuilt from the registry on EVERY iteration, so late
    registrations ARE declared to the provider on subsequent turns.

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

    Streaming is INCOMPATIBLE with native-tools mode: a streamed response
    carries no structured ``tool_calls`` (the adapter deliberately drops
    streamed tool-call deltas), so a ``stream=True`` +
    ``native_tools_enabled=True`` loop would parse empty completions and
    terminate on the fallback. ``AgentLoop(...)`` raises ``ValueError``
    at construction when both are enabled, and likewise for
    ``stream=True`` + ``tool_mode=ToolMode.NATIVE``.
"""

from __future__ import annotations

import asyncio
import enum
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Final, Literal, TypeVar
from uuid import uuid4

import structlog
from pydantic import ValidationError

from fifty_agent_sdk.errors import (
    AgentSdkError,
    LLMError,
    ParserError,
    ToolNotFound,
    ToolTimeout,
)
from fifty_agent_sdk.llm.protocol import LLMClient
from fifty_agent_sdk.llm.types import (
    _REASONING_EFFORT_PATTERN,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
    Usage,
)
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.observability.hooks import invoke_hook
from fifty_agent_sdk.parser.base import FinalAnswer, MultiAction, Parser, ParseResult
from fifty_agent_sdk.parser.native_tools import NativeToolsParser
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
from fifty_agent_sdk.tool_mode import ToolMode, _resolve_tool_mode
from fifty_agent_sdk.tools.protocol import Tool, ToolResult
from fifty_agent_sdk.tools.registry import Registry

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger. INFO at iteration boundaries; DEBUG per sub-event."""


class _Unset(enum.Enum):
    """Private "not passed" marker for ``AgentLoop(temperature=...)`` (FR-002 D9).

    ``temperature=None`` is a real instruction (omit the key from every
    request), so ``None`` cannot also mean "not passed". A single-member enum
    lets mypy narrow ``is _UNSET`` / ``is not _UNSET`` exactly, which a bare
    ``object()`` sentinel does not. Consumers never pass it: they omit the
    kwarg. Not exported and not part of the public API.
    """

    UNSET = "UNSET"


_UNSET: Final = _Unset.UNSET
"""The single :class:`_Unset` member, the default of ``AgentLoop(temperature=)``."""

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


def _render_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools`` request-param envelope from registered tools.

    Each tool becomes::

        {"type": "function",
         "function": {"name": t.name, "description": t.description,
                      "parameters": {"type": t.schema.type,
                                     "properties": t.schema.properties,
                                     "required": t.schema.required,
                                     "additionalProperties": t.schema.additionalProperties}}}

    The ``parameters`` sub-object mirrors the four fields on
    :class:`fifty_agent_sdk.tools.protocol.ToolSchema` exactly. This is only
    called when native tools are declared — ``tool_mode=ToolMode.NATIVE``, or
    :attr:`SafetyConfig.native_tools_enabled` on the legacy path. Every other
    path never builds this envelope, so no tool declaration reaches the wire
    and the request is byte-for-byte pre-BR-008.

    Args:
        tools: Snapshot of registered tools from
            :meth:`fifty_agent_sdk.tools.registry.Registry.list`.

    Returns:
        A list of OpenAI ``tools`` entries, one per tool. Empty list for an
        empty tool set (the caller decides whether to omit the param —
        ``ToolMode.NATIVE`` omits it, the legacy flag sends ``[]``).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": tool.schema.type,
                    "properties": tool.schema.properties,
                    "required": tool.schema.required,
                    "additionalProperties": tool.schema.additionalProperties,
                },
            },
        }
        for tool in tools
    ]


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


def _validate_reasoning_effort(reasoning_effort: object) -> str | None:
    """Check ``AgentLoop(reasoning_effort=)`` at construction (FR-002 D4).

    Uses the same lexical guard as :attr:`ChatRequest.reasoning_effort`.
    Validating here, rather than letting the first ``_build_request`` raise a
    pydantic ``ValidationError`` inside :meth:`AgentLoop.run`, keeps the
    "every run ends with a FinalEvent" guarantee: a bad value never reaches
    the generator.

    Raises:
        ValueError: When the value is neither ``None`` nor a lowercase token.
    """
    if reasoning_effort is None:
        return None
    if not isinstance(reasoning_effort, str) or not re.fullmatch(
        _REASONING_EFFORT_PATTERN, reasoning_effort
    ):
        raise ValueError(
            "reasoning_effort must be a lowercase provider level such as 'low', "
            f"'medium' or 'high', or None to omit it; got {reasoning_effort!r}"
        )
    return reasoning_effort


def _validate_temperature(temperature: float | None | _Unset) -> float | None | _Unset:
    """Check ``AgentLoop(temperature=)`` at construction (FR-002 D10).

    The omitted marker and ``None`` pass through unchanged. Any other value
    must be an ``int`` or ``float`` (``bool`` excluded; a numeric ``str`` is
    rejected rather than coerced) and is then validated through the real
    :attr:`ChatRequest.temperature` field, so the ``[0.0, 2.0]`` range has a
    single source of truth. The VALIDATED value is returned, so an ``int``
    such as ``1`` is stamped as the float ``1.0``, the value a direct
    ``ChatRequest(temperature=1)`` carries.

    Raises:
        ValueError: When the value is not a number in ``[0.0, 2.0]`` (NaN and
            infinities included) and is not ``None``.
    """
    if temperature is _UNSET or temperature is None:
        return temperature
    message = f"temperature must be a number in [0.0, 2.0], or None to omit it; got {temperature!r}"
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError(message)
    try:
        validated = ChatRequest.model_validate(
            {"messages": [], "model": "_", "temperature": temperature}
        ).temperature
    except ValidationError as exc:
        raise ValueError(message) from exc
    return validated


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
            output format. REQUIRED when ``tool_mode`` is omitted (a
            :class:`TypeError` otherwise, as in 1.7.0). Optional under
            ``ToolMode.JSON`` / ``ToolMode.PROSE``, where the mode supplies
            its parser and a compatible custom parser (for example a wrapper
            around :class:`JsonModeParser`) is honoured. Must be omitted under
            ``ToolMode.NATIVE``.
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
            no token events. MUST NOT be combined with
            ``SafetyConfig(native_tools_enabled=True)`` or
            ``tool_mode=ToolMode.NATIVE`` — streamed responses carry no
            structured ``tool_calls``, so those combinations are rejected at
            construction (see Raises).
        output_format: Optional override for the system prompt's
            ``output_format`` slot. When non-empty, replaces whatever is
            in ``prompts.output_format``. Useful for callers that want
            to pin the parser-aligned format without rebuilding
            ``prompts``. Under an explicit ``tool_mode`` an empty value (in
            both slots) takes the mode's own format, and a shipped format
            constant belonging to a different mode raises
            :class:`ValueError`.
        tool_message_role: Wire-format role used for the synthetic message
            the loop appends after a tool invocation. ``None`` (the default)
            means "not set": it resolves to ``"tool"`` when ``tool_mode`` is
            omitted (the 1.7.0 default), to ``"assistant"`` under
            ``ToolMode.JSON`` / ``ToolMode.PROSE`` (``"user"`` allowed,
            ``"tool"`` rejected — a text tool call has no id to pair with),
            and to ``"tool"`` under ``ToolMode.NATIVE`` (the only allowed
            value). On the legacy path, ``"tool"`` preserves the OpenAI
            tool-role wire format and is
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
        tool_mode: FR-001. A :class:`fifty_agent_sdk.tool_mode.ToolMode` (or
            its string value) that configures the parser, output format,
            tool-result role, tool declaration and parser-retry reminder
            together, so switching protocol is a one-value change. Passing
            it together with an owned knob (``parser``, ``output_format``,
            ``tool_message_role``, ``SafetyConfig.native_tools_enabled``)
            whose value belongs to another mode raises :class:`ValueError`.
            ``None`` (default) sends the same request bodies and system
            prompt as 1.7.0 (same keys, values and JSON types); the one
            event-level change is ``ThoughtEvent.text`` on native tool
            turns (see CHANGELOG).
        reasoning_effort: FR-002. A provider reasoning level such as
            ``"low"``, ``"medium"`` or ``"high"`` (a lowercase token; which
            levels a model accepts is up to the provider). Set as
            :attr:`ChatRequest.reasoning_effort` on EVERY request this loop
            builds: tool steps, parser-retry and require-tool-before-final
            re-asks, native and streamed turns. It is sent for every model
            name, with no model-name filter, so a provider that rejects it
            ends the run on the first call with an ``ErrorEvent`` and the
            fallback ``FinalEvent``. The string ``"none"`` is a level and is
            sent; ``None`` (default) omits it and sends the same request
            bodies as 1.8.0 (same keys, values and JSON types). There is no
            per-run override: use a direct ``ChatRequest`` or an
            :class:`~fifty_agent_sdk.llm.protocol.LLMClient` wrapper for
            per-request values. Independent of ``tool_mode``.
        temperature: FR-002. Omitted (the default), the loop keeps sending
            ``temperature`` ``0.0`` as in 1.8.0 (the same request bodies:
            same keys, values and JSON types). A
            number in ``[0.0, 2.0]`` is sent on every request this loop
            builds. ``None`` OMITS the key from every request, for providers
            or models that reject sampling parameters; it means "leave it
            out", not "use the default". Independent of ``reasoning_effort``
            and ``tool_mode``: the loop never drops ``temperature`` because
            ``reasoning_effort`` is set. If your provider rejects
            ``temperature`` alongside ``reasoning_effort`` (OpenAI documents
            this for gpt-5.1 with a level other than ``"none"``; not verified
            by this SDK), pass ``temperature=None`` yourself. Omit the kwarg
            rather than passing the private default marker.

    Raises:
        TypeError: When both ``parser`` and ``tool_mode`` are omitted.
        ValueError: When ``reasoning_effort`` is neither ``None`` nor a
            lowercase token; when ``temperature`` is neither ``None`` nor a
            number in ``[0.0, 2.0]`` (``bool`` and numeric strings are
            rejected); when ``tool_mode`` is not a valid :class:`ToolMode`; when
            ``tool_mode`` conflicts with an owned knob (see ``tool_mode``);
            when ``tool_mode=ToolMode.NATIVE`` is combined with
            ``stream=True``; and when ``stream=True`` is combined with
            ``SafetyConfig(native_tools_enabled=True)``. A streamed
            completion carries no structured ``tool_calls`` (the LLM
            adapter deliberately drops streamed tool-call deltas), so the
            combination can never dispatch a native tool call — every
            streamed turn would parse empty content, burn the parser
            retry, and terminate on the fallback message. The
            misconfiguration is rejected at construction instead.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        # TODO(BR-008+): Widen `registry` to a RegistryProtocol once non-in-process
        # providers (MCP, RPC) land — current type pins us to the concrete class.
        registry: Registry,
        parser: Parser | None = None,
        prompts: PromptSections,
        safety: SafetyConfig,
        model: str,
        stream: bool = False,
        output_format: str = "",
        tool_message_role: Literal["tool", "user", "assistant"] | None = None,
        hooks: Hooks | None = None,
        tool_mode: ToolMode | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None | _Unset = _UNSET,
    ) -> None:
        # FR-002: validate the request options first, before the tool-mode
        # resolver, so a bad value fails cheaply at construction and never as
        # a pydantic ValidationError inside run().
        reasoning_effort = _validate_reasoning_effort(reasoning_effort)
        temperature = _validate_temperature(temperature)
        # FR-001: resolve every tool-calling knob together. `tool_mode=None`
        # is the frozen legacy path — each resolved value equals what 1.7.0
        # read directly (parser required, role "tool", native behaviour keyed
        # on `safety.native_tools_enabled`, the stream/native ValueError
        # verbatim) — so omitting it sends the same request bodies as 1.7.0
        # (AC-3: content pinned by tests/loop/test_legacy_golden.py; key order
        # by the unchanged _build_body).
        resolved = _resolve_tool_mode(
            tool_mode=tool_mode,
            parser=parser,
            prompts=prompts,
            output_format=output_format,
            tool_message_role=tool_message_role,
            safety=safety,
            stream=stream,
        )
        self._llm = llm
        self._registry = registry
        self._parser = resolved.parser
        # BR-007: native (provider-structured) tool-call parser. Constructed
        # here (no constructor arg — concrete class) so the precedence branch
        # in `run()` can dispatch a populated `response.message.tool_calls`
        # through it. Independent of the text `Parser`; does not participate
        # in prompt building (the system-prompt snapshot below is unchanged).
        self._native_parser = NativeToolsParser()
        # The consumer's own frozen SafetyConfig, never copied or mutated.
        # Values derived from the tool mode live in the attributes below.
        self._safety = safety
        self._model = model
        self._stream = stream
        self._tool_message_role: Literal["tool", "user", "assistant"] = resolved.tool_message_role
        self._hooks = hooks
        self._tool_mode: ToolMode | None = resolved.mode
        self._declare_native_tools = resolved.declare_native_tools
        self._native_precedence = resolved.native_precedence
        self._omit_empty_tools = resolved.omit_empty_tools
        self._parser_retry_reminder = resolved.parser_retry_reminder
        self._echo_blank_retry = resolved.echo_blank_retry
        # FR-002 D7: request options, composed ONCE and spread into every
        # ChatRequest in `_build_request`. A key is present only when the
        # consumer passed the kwarg (`temperature=None` included, which is a
        # real "omit the key" instruction), so a loop with both omitted builds
        # requests whose values for 1.8.0's fields, and `model_fields_set`,
        # match 1.8.0 (the new `reasoning_effort` field stays None). The two
        # options are never coupled (D9).
        self._reasoning_effort = reasoning_effort
        self._temperature = temperature
        self._request_options: dict[str, Any] = {}
        if reasoning_effort is not None:
            self._request_options["reasoning_effort"] = reasoning_effort
        if temperature is not _UNSET:
            self._request_options["temperature"] = temperature
        # Snapshot tool descriptions ONCE; subsequent registry mutations are
        # invisible to this loop instance. Documented behavior.
        self._system_prompt = self._build_system_prompt(prompts, resolved.output_format)

    def _build_system_prompt(self, prompts: PromptSections, output_format: str) -> str:
        """Render the system prompt from a registry snapshot and the provided slots.

        When native tools are declared (``ToolMode.NATIVE``, or
        :attr:`SafetyConfig.native_tools_enabled` on the legacy path), the
        prompt-side tool-description block is SUPPRESSED (left empty) so tools
        are declared to the model ONLY via the OpenAI ``tools`` request param.
        Declaring them in both places risks the model emitting text-mode JSON
        tool calls instead of native ``tool_calls`` — the structured ``tools``
        param is the sole source of truth under native mode. Otherwise the
        renderer runs exactly as before — byte-for-byte.

        ``output_format`` is the resolved value from the tool-mode resolver:
        the raw kwarg on the legacy path, the mode's effective format under an
        explicit ``tool_mode`` (FR-001).
        """
        if self._declare_native_tools:
            tool_block = ""
        else:
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
        """Wrap the working message list in a :class:`ChatRequest` for this loop's model.

        When native tools are declared (``ToolMode.NATIVE``, or
        :attr:`SafetyConfig.native_tools_enabled` on the legacy path), the
        request carries the registry's tools as the OpenAI ``tools`` param
        (built via :func:`_render_openai_tools`) plus ``tool_choice="auto"``.
        Otherwise the request is ``ChatRequest(messages=..., model=...)``
        plus the request options described below.

        Under ``ToolMode.NATIVE`` an EMPTY registry sends neither ``tools`` nor
        ``tool_choice`` (OpenAI rejects an empty ``tools`` array); the registry
        is re-read every iteration, so a later registration is declared on the
        next turn. The legacy flag keeps sending ``tools=[]`` (FR-001 D10).

        Every request, on all three branches, also carries the loop's request
        options (FR-002): ``reasoning_effort`` when set, and ``temperature``
        when passed (``None`` included, which omits the key on the wire). This
        method is the one place every loop request is built, so tool steps,
        re-asks, native and streamed turns all get them. With both options
        omitted nothing extra is passed to ``ChatRequest``, so each request's
        values for 1.8.0's fields, and its ``model_fields_set``, match 1.8.0's
        (the new ``reasoning_effort`` field stays ``None``).
        """
        if self._declare_native_tools:
            tools = _render_openai_tools(self._registry.list())
            if not tools and self._omit_empty_tools:
                return ChatRequest(
                    messages=list(messages), model=self._model, **self._request_options
                )
            return ChatRequest(
                messages=list(messages),
                model=self._model,
                tools=tools,
                tool_choice="auto",
                **self._request_options,
            )
        return ChatRequest(messages=list(messages), model=self._model, **self._request_options)

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
            native_turn: bool = False
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
                # BR-007 precedence branch: if the adapter populated a
                # structured `response.message.tool_calls`, dispatch through
                # the native parser (bypassing text parsing entirely). Native
                # parse does NOT participate in the BR-018 one-shot retry
                # (D4): a structured `tool_calls` array comes from the
                # provider SDK already formed, so a malformed one is an
                # unrecoverable adapter/provider error — emit the same
                # terminal ErrorEvent + fallback FinalEvent as the
                # retry-disabled / budget-exhausted text path below, then
                # return. In stream mode `response is None` (a stream has no
                # single response object), so a streamed turn CANNOT go
                # native (D5) and falls through to the text path unchanged.
                # FR-001 D9: the precedence is mode-gated. Legacy and NATIVE
                # keep it; an explicit JSON/PROSE loop turns it off (below).
                if self._native_precedence and response is not None and response.message.tool_calls:
                    try:
                        parsed = self._native_parser.parse(response)
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
                    native_turn = True
                    # Native parse does not retry — leave the inner loop and
                    # continue with the unchanged branching logic below.
                    break
                if (
                    not self._native_precedence
                    and response is not None
                    and response.message.tool_calls
                ):
                    # FR-001 D9: an explicit JSON/PROSE loop never declared
                    # tools, so structured tool_calls are unrequested. Running
                    # them would put a native assistant turn before a
                    # role="assistant"/"user" result — the mirror image of the
                    # half-native 400. Ignore them and parse the text instead.
                    # Log the count only, never content (§10). Unreachable on
                    # the legacy path, where `_native_precedence` is True.
                    _log.warning(
                        "native_tool_calls_ignored",
                        count=len(response.message.tool_calls),
                        run_id=run_id,
                    )
                native_turn = False
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
                        # FR-001 D6: under an explicit tool_mode a BLANK
                        # completion is not echoed (it carries no information
                        # and some backends reject an empty assistant
                        # message). Legacy always echoes, as in 1.7.0.
                        if self._echo_blank_retry or completion.strip():
                            working.append(ChatMessage(role="assistant", content=completion))
                        # FR-001 D11 (AC-6): the mode-appropriate reminder.
                        # Legacy resolves this to `safety.parser_retry_reminder`.
                        working.append(
                            ChatMessage(
                                role="user",
                                content=self._parser_retry_reminder,
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

            # BR-006: a native response carrying MORE THAN ONE tool call yields
            # a MultiAction. Dispatch all N calls concurrently under a
            # Semaphore-bounded gather, feed their observations back in CALL
            # order, and emit per-call events with monotonic sequence — then
            # `continue` (the batch counts as ONE iteration unit). The single-
            # call path (ThoughtAction, below) is UNCHANGED: a single native
            # call returns ThoughtAction, never MultiAction, so this branch is
            # dead code for every text/JSON/single-call turn.
            if isinstance(parsed, MultiAction):
                calls = parsed.tool_calls
                # Mint N DISTINCT call_ids BEFORE the assistant-turn append so
                # each entry on the replayed assistant turn carries its own
                # pairing key (learning #934: minting after would share one id
                # across N entries → the provider cannot pair N distinct tool
                # replies → HTTP 400). Each id is reused for the matching
                # ToolStartedEvent, registry.invoke, and tool-reply message.
                call_ids = [uuid4().hex for _ in calls]

                # One ThoughtEvent for the batch (the MultiAction's thought,
                # mirroring the single-call ThoughtEvent below).
                yield self._make_event(ThoughtEvent, sequence_box, text=parsed.thought)

                # One ActionEvent PER call, in CALL order, via _make_event so
                # sequence stays monotonic and dense.
                for tc in calls:
                    yield self._make_event(
                        ActionEvent,
                        sequence_box,
                        tool_name=tc.name,
                        args=dict(tc.args),
                    )

                # Append the assistant turn carrying ALL N tool_calls (each
                # with its minted id). ChatMessage.tool_call_id is left None —
                # it is single-valued and cannot carry N ids; the per-call ids
                # live on tool_calls[].id and _serialize_message sources each
                # wire id from there (falling back to tool_call_id for the
                # single-call path).
                working.append(
                    ChatMessage(
                        role="assistant",
                        content=completion,
                        tool_calls=[
                            ToolCall(name=tc.name, args=dict(tc.args), id=cid)
                            for tc, cid in zip(calls, call_ids, strict=True)
                        ],
                    )
                )

                # BR-036: a tool is being dispatched this run, so a later
                # `final` is now grounded and must NOT trigger the
                # force-reconsider guard.
                tool_invoked_this_run = True

                # One ToolStartedEvent PER call, in CALL order.
                for tc, cid in zip(calls, call_ids, strict=True):
                    yield self._make_event(
                        ToolStartedEvent,
                        sequence_box,
                        tool_name=tc.name,
                        call_id=cid,
                    )
                    _log.debug(
                        "tool_invoked",
                        name=tc.name,
                        call_id=cid,
                        run_id=run_id,
                    )

                # Semaphore-bounded gather (Decision §C). `return_exceptions=
                # True` is MANDATORY: without it, the first failing call
                # cancels all sibling tasks (learning #886 — per-call
                # isolation). NO asyncio.wait_for is added (learning #819) —
                # Registry.invoke already enforces the per-tool timeout via
                # `async with asyncio.timeout`, so the gather inherits correct
                # inline-task timeout/cancellation semantics.
                sem = asyncio.Semaphore(self._safety.max_concurrent_tool_calls)

                async def _run_one(
                    cid: str, tc: ToolCall, _sem: asyncio.Semaphore = sem
                ) -> ToolResult:
                    async with _sem:
                        return await self._registry.invoke(
                            tc.name,
                            dict(tc.args),
                            timeout=self._safety.tool_timeout_seconds,
                        )

                raw_results = await asyncio.gather(
                    *[_run_one(cid, tc) for cid, tc in zip(call_ids, calls, strict=True)],
                    return_exceptions=True,
                )

                # Per-call classification loop, in CALL order (NOT completion
                # order — zip preserves the call index, so observations are
                # appended deterministically regardless of which call finished
                # first). Fatal BaseExceptions / AgentSdkError returned by
                # return_exceptions=True are RE-RAISED here (learning #886):
                # a genuinely fatal error must still terminate; recoverable
                # tool failures (ToolNotFound/ToolTimeout/is_error) yield a
                # ToolFailedEvent + observation for THAT call only.
                for tc, cid, raw in zip(calls, call_ids, raw_results, strict=True):
                    tool_name = tc.name
                    if isinstance(raw, ToolNotFound):
                        yield self._make_event(
                            ToolFailedEvent,
                            sequence_box,
                            tool_name=tool_name,
                            call_id=cid,
                            error=f"ToolNotFound: {raw.message}",
                        )
                        working.append(
                            self._build_tool_message(
                                tool_name=tool_name,
                                call_id=cid,
                                content_for_tool_role=(
                                    f"ToolNotFound: tool '{tool_name}' is not registered."
                                ),
                                content_for_other_role=(
                                    f"Tool {tool_name} failed: "
                                    f"ToolNotFound: tool '{tool_name}' is not registered."
                                ),
                            )
                        )
                    elif isinstance(raw, ToolTimeout):
                        yield self._make_event(
                            ToolFailedEvent,
                            sequence_box,
                            tool_name=tool_name,
                            call_id=cid,
                            error=f"ToolTimeout: {raw.message}",
                        )
                        working.append(
                            self._build_tool_message(
                                tool_name=tool_name,
                                call_id=cid,
                                content_for_tool_role=f"ToolTimeout: {raw.message}",
                                content_for_other_role=(
                                    f"Tool {tool_name} failed: ToolTimeout: {raw.message}"
                                ),
                            )
                        )
                    elif isinstance(
                        raw, (AgentSdkError, asyncio.CancelledError, KeyboardInterrupt, SystemExit)
                    ):
                        # Fatal: do NOT swallow as a recoverable observation.
                        # Registry.invoke re-raises these untouched; their
                        # presence here means they escaped as a returned
                        # value via return_exceptions=True — re-raise so the
                        # run terminates (and consumer CancelledError
                        # propagates out of gather untouched).
                        #
                        # ORDER IS LOAD-BEARING: this arm MUST stay AFTER the
                        # ToolNotFound/ToolTimeout arms above — both are
                        # AgentSdkError subclasses, so checking AgentSdkError
                        # first would silently turn a recoverable tool error
                        # into a fatal run termination. test_multi_call_failure_isolation
                        # locks the recoverable contract.
                        raise raw
                    elif isinstance(raw, ToolResult):
                        if raw.is_error:
                            error_text = (
                                raw.error
                                if raw.error is not None
                                else "tool reported error with no message"
                            )
                            yield self._make_event(
                                ToolFailedEvent,
                                sequence_box,
                                tool_name=tool_name,
                                call_id=cid,
                                error=error_text,
                            )
                            working.append(
                                self._build_tool_message(
                                    tool_name=tool_name,
                                    call_id=cid,
                                    content_for_tool_role=f"Tool error: {error_text}",
                                    content_for_other_role=(
                                        f"Tool {tool_name} failed: {error_text}"
                                    ),
                                )
                            )
                        else:
                            yield self._make_event(
                                ObservationEvent,
                                sequence_box,
                                tool_name=tool_name,
                                call_id=cid,
                                result=raw,
                            )
                            working.append(
                                self._build_tool_message(
                                    tool_name=tool_name,
                                    call_id=cid,
                                    content_for_tool_role=_serialize_tool_output(raw.output),
                                    content_for_other_role=(
                                        f"Tool {tool_name} returned: "
                                        f"{_serialize_tool_output(raw.output)}"
                                    ),
                                )
                            )
                    elif isinstance(raw, Exception):
                        # Defensive: a plain non-SDK Exception should not
                        # occur post-Registry.invoke (the registry catches
                        # Exception → ToolResult(is_error=True)). Treat as a
                        # recoverable tool error rather than terminating.
                        error_text = f"{type(raw).__name__}: {raw}"
                        yield self._make_event(
                            ToolFailedEvent,
                            sequence_box,
                            tool_name=tool_name,
                            call_id=cid,
                            error=error_text,
                        )
                        working.append(
                            self._build_tool_message(
                                tool_name=tool_name,
                                call_id=cid,
                                content_for_tool_role=f"Tool error: {error_text}",
                                content_for_other_role=(f"Tool {tool_name} failed: {error_text}"),
                            )
                        )
                    else:
                        # Unexpected non-Exception, non-ToolResult value —
                        # raise loudly rather than silently dropping a call.
                        raise TypeError(
                            f"unexpected gather result type for tool '{tool_name}': "
                            f"{type(raw).__name__}",
                        )
                # The batch consumed ONE iteration unit; re-enter the outer
                # while loop for the next round (still bounded by
                # max_iterations).
                continue

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
            # BR-007 (D6): a native tool turn MUST carry the structured
            # `tool_calls` on the assistant turn so provider-native replay can
            # pair the subsequent `role="tool"` reply (keyed by `tool_call_id`)
            # to this turn.
            #
            # BR-008 (Decision B — the id-pairing reordering): mint the
            # `call_id` BEFORE the assistant-turn append and embed it on the
            # assistant message's `tool_call_id` (reusing the existing field —
            # `ToolCall` carries only `{name, args}` per BR-007 D7). The SAME
            # `call_id` is then reused for the `ToolStartedEvent`,
            # `registry.invoke`, and the `_build_tool_message(call_id=...)`
            # reply, so the assistant turn's id and the tool reply's
            # `tool_call_id` match — a strict OpenAI endpoint requires the
            # pairing key on BOTH sides or it returns 400 on turn 2. The
            # text-path `else` branch never read `call_id`; hoisting the mint
            # above both branches is a no-op for text.
            tool_name = parsed.tool_call.name
            call_id = uuid4().hex
            if native_turn:
                working.append(
                    ChatMessage(
                        role="assistant",
                        content=completion,
                        tool_call_id=call_id,
                        tool_calls=[
                            ToolCall(
                                name=parsed.tool_call.name,
                                args=dict(parsed.tool_call.args),
                            )
                        ],
                    )
                )
            else:
                working.append(ChatMessage(role="assistant", content=completion))

            # BR-036: a tool is being dispatched this run, so a later `final`
            # is now grounded (from the loop's structural view) and must NOT
            # trigger the force-reconsider guard.
            tool_invoked_this_run = True

            # ----- 4. Invoke tool ------------------------------------------
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
