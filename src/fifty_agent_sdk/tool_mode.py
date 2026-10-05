"""How the model calls tools: one ``ToolMode`` instead of four loose knobs (FR-001).

Before FR-001 a consumer chose a tool-calling protocol through four
independent :class:`fifty_agent_sdk.loop.AgentLoop` knobs —
``SafetyConfig.native_tools_enabled``, the injected ``parser``,
``output_format`` and ``tool_message_role`` — and nothing kept them consistent.
A loop could be half native and half text: tools declared natively, but a text
tool call still dispatched and its result sent back as ``role="tool"`` after an
assistant turn with no ``tool_calls``, which strict endpoints reject with
HTTP 400.

:class:`ToolMode` is the single switch. ``AgentLoop(tool_mode=...)`` resolves it
into every owned knob at construction:

============================  =====================  =====================  ==========================
                              ``JSON``               ``PROSE``              ``NATIVE``
============================  =====================  =====================  ==========================
tools declared via ``tools``  no                     no                     yes, ``tool_choice="auto"``
prompt tool block             rendered               rendered               suppressed
text parser                   ``JsonModeParser``     ``ProseModeParser``    final-only (text = answer)
output format                 JSON envelope          Thought/Action prose   none (plain-text final)
tool-result role              ``"assistant"``        ``"assistant"``        ``"tool"``, paired by id
                              (``"user"`` allowed)   (``"user"`` allowed)   (not configurable)
``stream=True``               allowed                allowed                rejected
============================  =====================  =====================  ==========================

Conflict rule
    With an explicit mode, each owned knob the caller ALSO passes is either
    compatible (honoured, e.g. a custom parser wrapper under ``JSON``) or a
    conflict, which raises :class:`ValueError` at construction and names the
    mode, the knob and the fix. A conflict is a value that would mix
    protocols: another mode's shipped artifact (``ProseModeParser`` under
    ``JSON``, ``JSON_MODE_OUTPUT_FORMAT`` under ``NATIVE``) or anything that can
    re-open a second tool channel (any ``parser=`` under ``NATIVE``). The
    loop never silently picks one side.

Legacy path
    When ``tool_mode`` is omitted the loop resolves exactly as 1.7.0 did
    and sends the same request bodies (same keys, values and JSON types),
    for the runs :mod:`fifty_agent_sdk.loop` scopes that claim to ("Non-ASCII
    text (BR-020)", "Error-path final text (BR-021)", "Tool-argument
    nesting (BR-019)", "Tool-result text (BR-022)" and "Model-written text
    in requests (BR-024)").
    That includes the half-native combination ``native_tools_enabled=True``
    + a text parser, kept for compatibility.
    ``tool_mode=ToolMode.NATIVE`` is the documented way to close that hole.

Only :class:`ToolMode` is public. The resolver, its result type and the
mode-specific reminder texts are private implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from fifty_agent_sdk.parser.base import Parser
from fifty_agent_sdk.parser.final_only import _FinalOnlyParser
from fifty_agent_sdk.parser.json_mode import JsonModeParser
from fifty_agent_sdk.parser.prose_mode import ProseModeParser
from fifty_agent_sdk.prompts import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    PromptSections,
)
from fifty_agent_sdk.safety import SafetyConfig

_ToolMessageRole = Literal["tool", "user", "assistant"]
"""Wire role for the synthetic message the loop appends after a tool call."""


class ToolMode(StrEnum):
    """The protocol the model uses to call tools. Pass as ``AgentLoop(tool_mode=...)``.

    A ``StrEnum``, so configuration-driven callers may pass the plain string
    (``"json"``, ``"prose"``, ``"native"``). An unknown value raises
    :class:`ValueError` at loop construction.

    Attributes:
        JSON: Text mode. The model answers with the JSON envelope from
            :data:`fifty_agent_sdk.prompts.JSON_MODE_OUTPUT_FORMAT`
            (``"action": "tool" | "final"``), parsed by
            :class:`fifty_agent_sdk.parser.json_mode.JsonModeParser`. Tools
            are described in the system prompt. Tool results go back as
            ``role="assistant"`` by default (``"user"`` allowed).
        PROSE: Text mode. Classic ReACT ``Thought / Action / Action Input``
            or ``Thought / Final Answer`` from
            :data:`fifty_agent_sdk.prompts.PROSE_MODE_OUTPUT_FORMAT`, parsed by
            :class:`fifty_agent_sdk.parser.prose_mode.ProseModeParser`. Same
            tool-description and result-role rules as ``JSON``.
        NATIVE: Provider-native function calling. Tools are declared through
            the OpenAI ``tools`` request param with ``tool_choice="auto"``
            (omitted when the registry is empty) and the prompt tool block is
            suppressed. A response carrying ``tool_calls`` is a tool turn;
            any other non-blank text is the final answer, so a text tool call
            is never dispatched. Results go back as ``role="tool"`` paired by
            ``tool_call_id``. An empty completion triggers the one-shot parser
            retry. Incompatible with ``stream=True``.
    """

    JSON = "json"
    PROSE = "prose"
    NATIVE = "native"


_PROSE_PARSER_RETRY_REMINDER: Final[str] = (
    "Your previous response did not follow the required Thought / Action format.\n"
    "Respond again in exactly one of these two forms and nothing else.\n"
    "To call a tool:\n"
    "Thought: <your reasoning>\n"
    "Action: <tool_name>\n"
    'Action Input: <arguments as an object, for example {"query": "..."}>\n'
    "To give the final answer:\n"
    "Thought: <your reasoning>\n"
    "Final Answer: <answer>"
)
"""Default parser-retry reminder under ``ToolMode.PROSE`` (FR-001 AC-6).

Restates the prose ReACT form instead of the JSON-envelope wording of the
:class:`SafetyConfig` default. It deliberately avoids the word "JSON" so a
prose loop is never told to emit a JSON envelope; the example object still shows the
``Action Input`` shape the prose parser decodes.
"""

_NATIVE_PARSER_RETRY_REMINDER: Final[str] = (
    "Your previous response was empty. Call one of the declared tools, or reply "
    "with your final answer as plain text."
)
"""Default parser-retry reminder under ``ToolMode.NATIVE`` (FR-001 AC-6, D6).

Under NATIVE the final-only parser fails ONLY on an empty completion, so the
reminder addresses exactly that case.
"""


@dataclass(frozen=True)
class _ResolvedToolMode:
    """Every loop setting a tool mode owns, resolved once at construction.

    Attributes:
        mode: The explicit mode, or ``None`` on the legacy path.
        parser: The text parser for responses without native ``tool_calls``.
        output_format: The value handed to ``AgentLoop._build_system_prompt``.
            On the legacy path it is the raw ``output_format`` kwarg, so the
            1.7.0 ``output_format or prompts.output_format`` fallback runs
            unchanged. Under an explicit mode it is the resolved effective
            format (``""`` means no Output Format section).
        tool_message_role: Role of the synthetic post-tool message.
        declare_native_tools: Declare tools through the ``tools`` param and
            suppress the prompt tool block.
        native_precedence: Route a response carrying ``tool_calls`` to the
            native parser. When ``False`` such calls are ignored (logged) and
            the text is parsed instead.
        omit_empty_tools: Send no ``tools`` / ``tool_choice`` when the
            registry is empty, instead of ``tools=[]``.
        parser_retry_reminder: ``user``-role text injected on a parser retry.
        echo_blank_retry: Echo a blank completion back as an assistant turn
            before the retry reminder (the 1.7.0 behaviour).
    """

    mode: ToolMode | None
    parser: Parser
    output_format: str
    tool_message_role: _ToolMessageRole
    declare_native_tools: bool
    native_precedence: bool
    omit_empty_tools: bool
    parser_retry_reminder: str
    echo_blank_retry: bool


def _conflict(mode: ToolMode, knob: str, why: str, fix: str) -> ValueError:
    """Build the one conflict message shape: mode, knob, reason, fix."""
    return ValueError(f"tool_mode=ToolMode.{mode.name} conflicts with {knob}: {why}. {fix}.")


def _format_summary(value: str, constant_name: str) -> str:
    """Describe an output format without echoing a (possibly long) custom string."""
    exact = {
        JSON_MODE_OUTPUT_FORMAT: "JSON_MODE_OUTPUT_FORMAT",
        PROSE_MODE_OUTPUT_FORMAT: "PROSE_MODE_OUTPUT_FORMAT",
    }
    if value in exact:
        return exact[value]
    return f"<custom format containing {constant_name}>"


def _resolve_legacy(
    *,
    parser: Parser | None,
    output_format: str,
    tool_message_role: _ToolMessageRole | None,
    safety: SafetyConfig,
    stream: bool,
) -> _ResolvedToolMode:
    """Resolve the ``tool_mode``-omitted path to exactly the 1.7.0 knob values (FR-001 D2)."""
    if parser is None:
        raise TypeError(
            "AgentLoop() requires parser= when tool_mode= is omitted. Pass a "
            "parser (for example JsonModeParser()), or pass tool_mode=ToolMode.JSON, "
            "ToolMode.PROSE or ToolMode.NATIVE and let the mode supply it."
        )
    if stream and safety.native_tools_enabled:
        # Message kept verbatim from 1.7.0 (loop.py) — existing tests match it.
        raise ValueError(
            "stream=True is incompatible with "
            "SafetyConfig(native_tools_enabled=True): a streamed "
            "completion carries no structured tool_calls, so native "
            "tool dispatch can never fire. Disable streaming or turn "
            "native_tools_enabled off."
        )
    return _ResolvedToolMode(
        mode=None,
        parser=parser,
        output_format=output_format,
        tool_message_role=tool_message_role if tool_message_role is not None else "tool",
        declare_native_tools=safety.native_tools_enabled,
        native_precedence=True,
        omit_empty_tools=False,
        parser_retry_reminder=safety.parser_retry_reminder,
        echo_blank_retry=True,
    )


def _resolve_parser(mode: ToolMode, parser: Parser | None) -> Parser:
    """Pick the mode's text parser, honouring a compatible explicit one (FR-001 D3)."""
    if mode is ToolMode.NATIVE:
        if parser is not None:
            raise _conflict(
                mode,
                f"parser={type(parser).__name__}()",
                "under NATIVE a tool call is only ever a structured tool_calls "
                "entry, and a text parser could dispatch a text tool call",
                "Drop parser=; NATIVE supplies its own final-only text parser",
            )
        return _FinalOnlyParser()
    if mode is ToolMode.JSON:
        if isinstance(parser, ProseModeParser):
            raise _conflict(
                mode,
                "parser=ProseModeParser()",
                "ProseModeParser parses the PROSE format, not the JSON envelope",
                "Drop parser=, pass a JSON parser, or use tool_mode=ToolMode.PROSE",
            )
        return parser if parser is not None else JsonModeParser()
    if isinstance(parser, JsonModeParser):
        raise _conflict(
            mode,
            "parser=JsonModeParser()",
            "JsonModeParser parses the JSON envelope, not the PROSE format",
            "Drop parser=, pass a prose parser, or use tool_mode=ToolMode.JSON",
        )
    return parser if parser is not None else ProseModeParser()


def _resolve_role(mode: ToolMode, role: _ToolMessageRole | None) -> _ToolMessageRole:
    """Pick the tool-result role for the mode (FR-001 D3, D4)."""
    if mode is ToolMode.NATIVE:
        if role is None or role == "tool":
            return "tool"
        raise _conflict(
            mode,
            f"tool_message_role={role!r}",
            "native tool results must be role='tool' messages paired to the "
            "assistant turn's tool_calls by tool_call_id",
            "Drop tool_message_role=",
        )
    if role == "tool":
        raise _conflict(
            mode,
            "tool_message_role='tool'",
            "a text-mode tool call has no tool_calls id to pair a role='tool' "
            "reply with, and strict endpoints reject that shape with HTTP 400",
            "Drop tool_message_role= (defaults to 'assistant') or pass 'user'",
        )
    return role if role is not None else "assistant"


def _resolve_output_format(mode: ToolMode, *, output_format: str, prompts: PromptSections) -> str:
    """Pick the effective output format for the mode (FR-001 D3).

    The conflict check runs on the EFFECTIVE value
    (``output_format or prompts.output_format``), so a shipped constant of the
    wrong mode is caught whichever slot it arrives through, including when it
    is only a substring of a longer custom format.
    """
    effective = output_format or prompts.output_format
    knob = "output_format" if output_format else "prompts.output_format"
    if mode is ToolMode.NATIVE:
        forbidden: tuple[str, ...] = ("JSON_MODE_OUTPUT_FORMAT", "PROSE_MODE_OUTPUT_FORMAT")
    elif mode is ToolMode.JSON:
        forbidden = ("PROSE_MODE_OUTPUT_FORMAT",)
    else:
        forbidden = ("JSON_MODE_OUTPUT_FORMAT",)
    constants = {
        "JSON_MODE_OUTPUT_FORMAT": JSON_MODE_OUTPUT_FORMAT,
        "PROSE_MODE_OUTPUT_FORMAT": PROSE_MODE_OUTPUT_FORMAT,
    }
    for name in forbidden:
        if constants[name] in effective:
            fix = (
                "Drop the output format (NATIVE needs none) or use a custom one "
                "that does not teach a text tool-call format"
                if mode is ToolMode.NATIVE
                else "Drop the output format so the mode supplies its own, or switch tool_mode"
            )
            raise _conflict(
                mode,
                f"{knob}={_format_summary(effective, name)}",
                f"{name} teaches a different tool-calling protocol than {mode.name}",
                fix,
            )
    if effective:
        return effective
    if mode is ToolMode.JSON:
        return JSON_MODE_OUTPUT_FORMAT
    if mode is ToolMode.PROSE:
        return PROSE_MODE_OUTPUT_FORMAT
    return ""


def _resolve_tool_mode(
    *,
    tool_mode: ToolMode | None,
    parser: Parser | None,
    prompts: PromptSections,
    output_format: str,
    tool_message_role: _ToolMessageRole | None,
    safety: SafetyConfig,
    stream: bool,
) -> _ResolvedToolMode:
    """Resolve ``AgentLoop``'s tool-calling knobs into one consistent setting (FR-001).

    ``tool_mode=None`` takes the frozen legacy path (1.7.0 semantics). Any
    other value is normalised with ``ToolMode(tool_mode)`` and resolved per
    the table in this module's docstring.

    Explicitness of :class:`SafetyConfig` fields is read from
    ``safety.model_fields_set``: a field counts as set when it was passed to
    the constructor, to ``model_validate``, or through
    ``model_copy(update=...)``. The consumer's ``safety`` object is never
    copied or mutated; derived values live on the returned
    :class:`_ResolvedToolMode`.

    Args:
        tool_mode: The requested mode, a mode string, or ``None`` for legacy.
        parser: The caller's ``parser=`` kwarg, or ``None``.
        prompts: The caller's prompt sections (read for ``output_format``).
        output_format: The caller's ``output_format=`` kwarg (``""`` = unset).
        tool_message_role: The caller's ``tool_message_role=`` kwarg, or
            ``None`` for unset.
        safety: The caller's :class:`SafetyConfig`.
        stream: The caller's ``stream=`` kwarg.

    Returns:
        The resolved settings.

    Raises:
        TypeError: Legacy path only, when ``parser`` is ``None``.
        ValueError: On an unknown mode, a mode/knob conflict,
            ``ToolMode.NATIVE`` with ``stream=True``, or (legacy) ``stream=True``
            with ``native_tools_enabled=True``.
    """
    if tool_mode is None:
        return _resolve_legacy(
            parser=parser,
            output_format=output_format,
            tool_message_role=tool_message_role,
            safety=safety,
            stream=stream,
        )

    try:
        mode = ToolMode(tool_mode)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in ToolMode)
        raise ValueError(
            f"unknown tool_mode={tool_mode!r}; expected a ToolMode or one of {valid}"
        ) from None

    if mode is ToolMode.NATIVE and stream:
        raise _conflict(
            mode,
            "stream=True",
            "a streamed completion carries no structured tool_calls, so native "
            "tool dispatch can never fire",
            "Set stream=False, or use tool_mode=ToolMode.JSON or ToolMode.PROSE to stream",
        )

    if mode is ToolMode.NATIVE:
        # The field defaults to False, so only an EXPLICIT False is a conflict.
        if "native_tools_enabled" in safety.model_fields_set and not safety.native_tools_enabled:
            raise _conflict(
                mode,
                "SafetyConfig(native_tools_enabled=False)",
                "NATIVE declares tools natively, which that flag explicitly turns off",
                "Drop native_tools_enabled from SafetyConfig or set it True",
            )
    elif safety.native_tools_enabled:
        raise _conflict(
            mode,
            "SafetyConfig(native_tools_enabled=True)",
            "that flag declares tools natively, but a text mode calls tools in text",
            "Drop native_tools_enabled, or use tool_mode=ToolMode.NATIVE",
        )

    resolved_parser = _resolve_parser(mode, parser)
    role = _resolve_role(mode, tool_message_role)
    effective_format = _resolve_output_format(mode, output_format=output_format, prompts=prompts)

    # FR-001 D11: a reminder the consumer set always wins; otherwise the
    # mode supplies one that matches its protocol (AC-6).
    if "parser_retry_reminder" in safety.model_fields_set or mode is ToolMode.JSON:
        reminder = safety.parser_retry_reminder
    elif mode is ToolMode.PROSE:
        reminder = _PROSE_PARSER_RETRY_REMINDER
    else:
        reminder = _NATIVE_PARSER_RETRY_REMINDER

    native = mode is ToolMode.NATIVE
    return _ResolvedToolMode(
        mode=mode,
        parser=resolved_parser,
        output_format=effective_format,
        tool_message_role=role,
        declare_native_tools=native,
        native_precedence=native,
        omit_empty_tools=native,
        parser_retry_reminder=reminder,
        # FR-001 D6: under an explicit mode a blank completion is not echoed
        # back — it tells the model nothing and some backends 400 on an
        # empty assistant message. Legacy keeps the echo.
        echo_blank_retry=False,
    )


__all__ = ["ToolMode"]
