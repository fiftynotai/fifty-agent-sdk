"""Iteration cap and timeout configuration for the agent loop.

:class:`SafetyConfig` is the loop's only knob for production safety. It is
intentionally narrow: an iteration cap, a per-tool timeout, and a fallback
message the loop emits when iteration is exhausted. The iteration counter
itself is a plain ``int`` inside :func:`agent_sdk.loop.AgentLoop.run` —
introducing a dedicated counter class for a single integer would be
over-abstraction.

When the loop terminates due to safety exhaustion it emits an
:class:`agent_sdk.streaming.ErrorEvent` with ``error_type=
"MaxIterationsExceeded"`` followed by a :class:`agent_sdk.streaming.
FinalEvent` carrying :attr:`SafetyConfig.fallback_message`, then returns
cleanly from the async generator. The exception class
:class:`agent_sdk.errors.MaxIterationsExceeded` is retained for callers
that want to raise from the consumer side of the event stream.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SafetyConfig(BaseModel):
    """Loop safety knobs.

    Frozen so a configured policy cannot be mutated mid-run. ``extra="forbid"``
    so typos raise validation errors instead of silently passing through.

    Attributes:
        max_iterations: Hard upper bound on ReACT cycles before the loop
            emits a fallback :class:`agent_sdk.streaming.FinalEvent`. Must
            be ``>= 1``.
        tool_timeout_seconds: Per-tool timeout passed verbatim to
            :meth:`agent_sdk.tools.registry.Registry.invoke`. ``None``
            disables the timeout; any positive float enforces it. Note:
            this does NOT cover the LLM call itself — wrap a higher-level
            runner with its own request timeout for that.
        fallback_message: Text used as the :class:`agent_sdk.streaming.
            FinalEvent` payload when the iteration cap is hit (or when a
            parser / LLM error terminates the run). Must be non-empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=10, ge=1)
    tool_timeout_seconds: float | None = Field(default=30.0, gt=0.0)
    fallback_message: str = Field(
        default="I was unable to complete the task within the allowed steps.",
        min_length=1,
    )


__all__ = ["SafetyConfig"]
