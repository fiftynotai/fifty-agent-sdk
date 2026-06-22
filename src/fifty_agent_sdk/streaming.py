"""Typed event stream for the agent loop.

Every step of the ReACT cycle emits exactly one :class:`AgentEvent` subclass.
The union is discriminated on the literal ``event_type`` field — callers can
``match event: case ThoughtEvent(): ...`` or branch with :func:`isinstance`.

All events are frozen Pydantic v2 models with ``extra="forbid"``. The loop
assigns ``sequence`` (a monotonic per-run counter starting at 0) and
``timestamp`` (UTC) so consumers can detect drops/reorders and reconstruct
timelines.

Reserved future events:
    :class:`ToolProgressEvent` ships in the union so consumers can write
    exhaustive ``match`` statements, but the v1 loop never emits it —
    :class:`fifty_agent_sdk.tools.protocol.Tool` has no progress channel. A future
    brief that widens the tool contract (or adds a sibling streaming method)
    will be the integration point.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from fifty_agent_sdk.tools.protocol import ToolResult


class _EventBase(BaseModel):
    """Shared frozen base for every :class:`AgentEvent` member.

    Not part of the public API — exists only to share the
    ``model_config``, ``sequence``, and ``timestamp`` fields across every
    event class. Consumers should branch on the concrete event types or on
    ``event_type`` directly, not on this class.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0)
    timestamp: datetime


class ThoughtEvent(_EventBase):
    """The model produced reasoning text on this iteration.

    Emitted exactly once per successful parse, regardless of whether the
    parse resolved to a tool call or a final answer.

    Attributes:
        event_type: Literal discriminator; always ``"thought"``.
        text: The model's reasoning string, taken verbatim from the parser.
    """

    event_type: Literal["thought"] = "thought"
    text: str


class ActionEvent(_EventBase):
    """The model chose to invoke a tool.

    Emitted immediately after a :class:`ThoughtEvent` when the parser
    returned a :class:`fifty_agent_sdk.parser.base.ThoughtAction`.

    Attributes:
        event_type: Literal discriminator; always ``"action"``.
        tool_name: Name of the tool the model asked to invoke.
        args: Arguments for the invocation. Defaults to ``{}``.
    """

    event_type: Literal["action"] = "action"
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolStartedEvent(_EventBase):
    """The loop is about to dispatch a tool call.

    Emitted right before :meth:`fifty_agent_sdk.tools.registry.Registry.invoke`
    is awaited. Pairs with a later :class:`ObservationEvent` or
    :class:`ToolFailedEvent` carrying the same ``call_id``.

    Attributes:
        event_type: Literal discriminator; always ``"tool_started"``.
        tool_name: Name of the tool being invoked.
        call_id: Opaque correlation id (UUID4 hex). Matches the id on the
            corresponding :class:`ObservationEvent` / :class:`ToolFailedEvent`.
    """

    event_type: Literal["tool_started"] = "tool_started"
    tool_name: str
    call_id: str


class ToolProgressEvent(_EventBase):
    """Reserved — the v1 loop does not emit this event.

    Included in the discriminated union so consumers can write exhaustive
    ``match`` statements and forward-compatible code. A later brief that
    widens the :class:`fifty_agent_sdk.tools.protocol.Tool` contract to support
    in-flight progress will be the integration point.

    Attributes:
        event_type: Literal discriminator; always ``"tool_progress"``.
        tool_name: Name of the in-flight tool.
        call_id: Correlation id of the originating
            :class:`ToolStartedEvent`.
        message: Human-readable progress description provided by the tool.
    """

    event_type: Literal["tool_progress"] = "tool_progress"
    tool_name: str
    call_id: str
    message: str


class ObservationEvent(_EventBase):
    """A tool returned a successful :class:`ToolResult`.

    Emitted when ``registry.invoke`` resolved to a result with
    ``is_error=False``. The full :class:`ToolResult` is carried so consumers
    can inspect the entire return value (output, is_error, error).

    Attributes:
        event_type: Literal discriminator; always ``"observation"``.
        tool_name: Name of the tool that returned.
        call_id: Correlation id of the originating
            :class:`ToolStartedEvent`.
        result: The full :class:`ToolResult` returned by the tool.
    """

    event_type: Literal["observation"] = "observation"
    tool_name: str
    call_id: str
    result: ToolResult


class ToolFailedEvent(_EventBase):
    """A tool invocation failed in a recoverable way.

    Emitted in any of the following situations:

    * The tool returned ``ToolResult(is_error=True)``.
    * :class:`fifty_agent_sdk.errors.ToolNotFound` raised (hallucinated tool name).
    * :class:`fifty_agent_sdk.errors.ToolTimeout` raised (per-tool timeout).

    The loop appends a synthesized ``role="tool"`` message to the working
    history so the model can reason about the failure on the next
    iteration. Unrecoverable SDK errors are NOT mapped to this event —
    they propagate out of the loop.

    Attributes:
        event_type: Literal discriminator; always ``"tool_failed"``.
        tool_name: Name of the tool that failed.
        call_id: Correlation id of the originating
            :class:`ToolStartedEvent`.
        error: Human-readable description of the failure.
    """

    event_type: Literal["tool_failed"] = "tool_failed"
    tool_name: str
    call_id: str
    error: str


class TokenEvent(_EventBase):
    """A single streamed delta of the FINAL answer.

    Only emitted when :class:`fifty_agent_sdk.loop.AgentLoop` was constructed with
    ``stream=True`` AND the iteration's parse resolved to a
    :class:`fifty_agent_sdk.parser.base.FinalAnswer`. Intermediate
    thought/action chunks are never token-streamed (the parser requires
    the full structured completion to disambiguate).

    Attributes:
        event_type: Literal discriminator; always ``"token"``.
        text: The streamed delta as produced by the upstream LLM. Empty
            deltas are dropped by the loop and never reach consumers.
    """

    event_type: Literal["token"] = "token"
    text: str


class FinalEvent(_EventBase):
    """The terminal answer for this run.

    Every successful or safety-terminated ``run()`` ends with exactly one
    :class:`FinalEvent`. Consumers may rely on this as their "iteration
    done" signal.

    Attributes:
        event_type: Literal discriminator; always ``"final"``.
        text: The terminal answer text. On safety-cap / parser error / LLM
            error termination this is :attr:`fifty_agent_sdk.safety.SafetyConfig.
            fallback_message`.
        raw_completion: The raw LLM completion that produced this answer
            (the JSON envelope under JSON-mode). Set ONLY on the happy-path
            :class:`fifty_agent_sdk.parser.base.FinalAnswer` branch; ``None`` on
            every safety-fallback path (LLM error, parser error, iteration
            cap). This is what the
            :class:`fifty_agent_sdk.runner.AgentRunner` persists when present,
            so multi-turn sessions get a faithful assistant turn — turn
            ``N+1`` then sees the same structured envelope shape the
            provider produced on turn ``N``, which is what JSON-mode
            parsers (and provider format detectors) rely on. The default
            of ``None`` keeps the model backward-compatible for callers
            that ignore the field.
    """

    event_type: Literal["final"] = "final"
    text: str
    raw_completion: str | None = None


class ErrorEvent(_EventBase):
    """A non-recoverable failure occurred and the run is about to terminate.

    Always followed by a :class:`FinalEvent` carrying
    :attr:`fifty_agent_sdk.safety.SafetyConfig.fallback_message`. Consumers can
    pattern-match on ``error_type`` to take application-specific action
    (for instance, re-raising :class:`fifty_agent_sdk.errors.MaxIterationsExceeded`
    when ``error_type == "MaxIterationsExceeded"``).

    Attributes:
        event_type: Literal discriminator; always ``"error"``.
        error_type: Short tag identifying the kind of failure (e.g.
            ``"LLMError"``, ``"ParserError"``, ``"MaxIterationsExceeded"``).
        message: Human-readable failure description.
        context: Structured debugging payload forwarded verbatim from the
            originating error (when applicable). Always a ``dict``,
            defaults to ``{}``.
    """

    event_type: Literal["error"] = "error"
    error_type: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


AgentEvent = Annotated[
    ThoughtEvent
    | ActionEvent
    | ToolStartedEvent
    | ToolProgressEvent
    | ObservationEvent
    | ToolFailedEvent
    | TokenEvent
    | FinalEvent
    | ErrorEvent,
    Field(discriminator="event_type"),
]
"""Discriminated union of every event the loop may emit.

Use with :class:`pydantic.TypeAdapter` for programmatic round-trip
validation, or branch on :func:`isinstance` against the concrete event
classes for ergonomic match-style consumption.
"""


__all__ = [
    "ActionEvent",
    "AgentEvent",
    "ErrorEvent",
    "FinalEvent",
    "ObservationEvent",
    "ThoughtEvent",
    "TokenEvent",
    "ToolFailedEvent",
    "ToolProgressEvent",
    "ToolStartedEvent",
]
