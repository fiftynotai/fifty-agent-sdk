"""Iteration cap and timeout configuration for the agent loop.

:class:`SafetyConfig` is the loop's only knob for production safety. It is
intentionally narrow: an iteration cap, a per-tool timeout, a fallback
message the loop emits when iteration is exhausted, and the BR-018 parser
one-shot retry knobs. The iteration counter itself is a plain ``int``
inside :func:`fifty_agent_sdk.loop.AgentLoop.run` — introducing a dedicated
counter class for a single integer would be over-abstraction.

When the loop terminates due to safety exhaustion it emits an
:class:`fifty_agent_sdk.streaming.ErrorEvent` with ``error_type=
"MaxIterationsExceeded"`` followed by a :class:`fifty_agent_sdk.streaming.
FinalEvent` carrying :attr:`SafetyConfig.fallback_message`, then returns
cleanly from the async generator. The exception class
:class:`fifty_agent_sdk.errors.MaxIterationsExceeded` is retained for callers
that want to raise from the consumer side of the event stream.

Parser-error retry (BR-018)
    On a :class:`fifty_agent_sdk.errors.ParserError` mid-iteration the loop now
    performs a one-shot retry: it re-issues the LLM call with the
    malformed completion echoed back as an ``assistant`` turn and a
    ``user`` turn carrying :attr:`SafetyConfig.parser_retry_reminder`.
    The retry budget is per-iteration (NOT cumulative across iterations)
    so a long multi-tool run does not silently exhaust it on the first
    drift. The retry can be disabled globally with
    :attr:`SafetyConfig.parser_retry_enabled` ``= False`` — useful for
    callers that prefer fast-fail semantics or for golden-path tests that
    want the original terminal-ParserError behavior.

Require-tool-before-final force-reconsider (BR-036)
    When :attr:`SafetyConfig.require_tool_before_final` is ``True`` (it is
    ``False`` by default, so every existing agent and test is unaffected),
    the loop refuses to accept the FIRST ``final`` answer of a run if NO
    tool has been invoked yet on that run. Instead — mirroring the BR-018
    parser-retry mechanism exactly — it echoes the model's completion back
    as an ``assistant`` turn, injects :attr:`tool_required_reminder` as a
    ``user`` turn, and re-prompts ONCE. The force is strictly one-shot per
    ``run()`` (a per-run budget, analogous to the parser-retry per-iteration
    budget): after the single forced reconsideration the loop accepts the
    next ``final`` regardless. The reminder is intentionally PERMISSIVE — it
    asks the model to call a tool only if the task needs one, and to simply
    re-answer otherwise — so a genuine greeting or capability question is
    never coerced into a tool call, it just costs one extra round-trip. The
    "tool ran this run" signal is per-``run()`` (one conversational turn), so
    a multi-turn session that searched on a PRIOR turn does NOT exempt the
    current turn. This is OFF by default to keep the loop domain-generic; the
    policy-specific reminder text and any hard grounding guarantee live in
    the consuming application, not here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SafetyConfig(BaseModel):
    """Loop safety knobs.

    Frozen so a configured policy cannot be mutated mid-run. ``extra="forbid"``
    so typos raise validation errors instead of silently passing through.

    Attributes:
        max_iterations: Hard upper bound on ReACT cycles before the loop
            emits a fallback :class:`fifty_agent_sdk.streaming.FinalEvent`. Must
            be ``>= 1``.
        tool_timeout_seconds: Per-tool timeout passed verbatim to
            :meth:`fifty_agent_sdk.tools.registry.Registry.invoke`. ``None``
            disables the timeout; any positive float enforces it. Note:
            this does NOT cover the LLM call itself — wrap a higher-level
            runner with its own request timeout for that.
        fallback_message: Text used as the :class:`fifty_agent_sdk.streaming.
            FinalEvent` payload when the iteration cap is hit (or when a
            parser / LLM error terminates the run). Must be non-empty.
        parser_retry_enabled: Kill-switch for the BR-018 one-shot
            parser-error retry. When ``True`` (the default) the loop
            transparently re-issues the LLM call on a mid-iteration
            :class:`fifty_agent_sdk.errors.ParserError`, echoing the malformed
            completion back to the model and injecting
            :attr:`parser_retry_reminder` as a ``user`` turn. When
            ``False`` the loop preserves pre-BR-018 fast-fail semantics —
            a single parse failure terminates the run with the existing
            ``ErrorEvent`` + fallback ``FinalEvent`` pair.
        parser_retry_reminder: ``user``-role message body injected on the
            retry to remind the model of the JSON envelope schema. Must
            be non-empty.
        require_tool_before_final: BR-036 opt-in force-reconsider knob.
            When ``True``, the loop refuses to accept the FIRST ``final``
            answer of a run if no tool has been invoked yet that run,
            injecting :attr:`tool_required_reminder` and re-prompting ONCE
            (one-shot per run). ``False`` by default — the whole guard
            branch is skipped unless a consumer opts in, so existing agents
            and tests see byte-for-byte unchanged behavior.
        tool_required_reminder: ``user``-role message body injected on the
            one-shot force-reconsider when
            :attr:`require_tool_before_final` fires. Must be non-empty. The
            default is domain-neutral; consumers that need a domain-specific
            prompt (e.g. "call policy_search first") override it. The text
            should be PERMISSIVE — it must allow the model to simply
            re-answer when no tool is actually needed (greetings, capability
            questions), or the guard would wrongly coerce those into tool
            calls.
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
            'string inside the `answer` field with action="final".'
        ),
        min_length=1,
    )
    require_tool_before_final: bool = Field(default=False)
    tool_required_reminder: str = Field(
        default=(
            "You produced a final answer without calling any tool. If this task "
            "requires a tool, call it first; if it does not, reply again with your "
            "final answer."
        ),
        min_length=1,
    )


__all__ = ["SafetyConfig"]
