"""Iteration cap and timeout configuration for the agent loop.

:class:`SafetyConfig` is the loop's only knob for production safety. It is
intentionally narrow: an iteration cap, a per-tool timeout, a fallback
message the loop emits when iteration is exhausted, and the BR-018 parser
one-shot retry knobs. The iteration counter itself is a plain ``int``
inside :func:`agent_sdk.loop.AgentLoop.run` — introducing a dedicated
counter class for a single integer would be over-abstraction.

When the loop terminates due to safety exhaustion it emits an
:class:`agent_sdk.streaming.ErrorEvent` with ``error_type=
"MaxIterationsExceeded"`` followed by a :class:`agent_sdk.streaming.
FinalEvent` carrying :attr:`SafetyConfig.fallback_message`, then returns
cleanly from the async generator. The exception class
:class:`agent_sdk.errors.MaxIterationsExceeded` is retained for callers
that want to raise from the consumer side of the event stream.

Parser-error retry (BR-018)
    On a :class:`agent_sdk.errors.ParserError` mid-iteration the loop now
    performs a one-shot retry: it re-issues the LLM call with the
    malformed completion echoed back as an ``assistant`` turn and a
    ``user`` turn carrying :attr:`SafetyConfig.parser_retry_reminder`.
    The retry budget is per-iteration (NOT cumulative across iterations)
    so a long multi-tool run does not silently exhaust it on the first
    drift. The retry can be disabled globally with
    :attr:`SafetyConfig.parser_retry_enabled` ``= False`` — useful for
    callers that prefer fast-fail semantics or for golden-path tests that
    want the original terminal-ParserError behavior.
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
        parser_retry_enabled: Kill-switch for the BR-018 one-shot
            parser-error retry. When ``True`` (the default) the loop
            transparently re-issues the LLM call on a mid-iteration
            :class:`agent_sdk.errors.ParserError`, echoing the malformed
            completion back to the model and injecting
            :attr:`parser_retry_reminder` as a ``user`` turn. When
            ``False`` the loop preserves pre-BR-018 fast-fail semantics —
            a single parse failure terminates the run with the existing
            ``ErrorEvent`` + fallback ``FinalEvent`` pair.
        parser_retry_reminder: ``user``-role message body injected on the
            retry to remind the model of the JSON envelope schema. Must
            be non-empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=10, ge=1)
    tool_timeout_seconds: float | None = Field(default=30.0, gt=0.0)
    fallback_message: str = Field(
        default="I was unable to complete the task within the allowed steps.",
        min_length=1,
    )
    parser_retry_enabled: bool = Field(default=True)
    parser_retry_reminder: str = Field(
        default=(
            "Your previous response was not valid JSON matching the required envelope.\n"
            "Respond again with a single JSON object exactly matching the schema in "
            "the system prompt. Output ONLY the JSON object — no prose, no Markdown, "
            "no code fences. If the answer is a list or explanation, put it as a "
            "string inside the `answer` field with action=\"final\"."
        ),
        min_length=1,
    )


__all__ = ["SafetyConfig"]
