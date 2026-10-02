"""Intervention hooks: host callables whose return values the loop honours (FR-003).

:class:`Interventions` is the counterpart of the observability
:class:`~fifty_agent_sdk.observability.hooks.Hooks`. ``Hooks`` only observe:
their return values are ignored and a failure is swallowed. An intervention
hook's return value changes the run (what the model reads, which call runs,
with which arguments), so each hook has a validated failure fallback instead.
The two containers are separate on purpose: one class with two failure
contracts is what BR-010 rejected for ``on_tool_error``.

Wiring
    ``AgentLoop(..., interventions=Interventions(...))``, and nowhere else.
    The loop builds every model-facing observation and dispatches every tool
    call, so :class:`~fifty_agent_sdk.runner.AgentRunner` takes no
    ``interventions``: it already passes its ``session_id`` into
    :meth:`~fifty_agent_sdk.loop.AgentLoop.run`, and the loop forwards it to
    both hooks. ``session_id`` is ``None`` when the loop runs without a
    Runner. With ``interventions=None`` (the default) the loop sends the same
    request bodies (same keys, values and JSON types) and emits the same event
    stream as 1.9.0 (timestamps and minted ids aside), for the runs
    :mod:`fifty_agent_sdk.loop` scopes that claim to ("Non-ASCII text
    (BR-020)", "Error-path final text (BR-021)" and "Tool-argument nesting
    (BR-019)").

``before_tool(session_id, call_id, tool_name, args)``
    Runs once per model tool call (the single call, and each member of a
    native batch in call order), before that call's ``ActionEvent``,
    including calls to names the registry does not know. ``args`` is the
    hook's own deep copy of the model's arguments (see "Argument copies"). It
    returns one of:

    * ``None``: proceed with the model's arguments. Editing the copy changes
      nothing (see "Argument copies"); changing arguments takes
      :class:`ReplaceToolArgs`.
    * :class:`ReplaceToolArgs`: run the call with these arguments instead.
      ``ActionEvent.args`` (and so a Runner's ``on_tool_start`` and
      ``tool_invocation`` audit payload) and ``after_tool`` see the
      replacement. It is never written into the model's own assistant turn.
      The tool name cannot change: deny, and let the model call the other
      tool.
    * :class:`DenyToolCall`: do not run the call. It keeps the per-call event
      shape an unregistered name has: ``ActionEvent`` (the model's
      arguments), ``ToolStartedEvent``, then
      ``ToolFailedEvent(error="Tool call denied: <reason>")``. The model
      reads the same text as that call's observation (``"Tool <name> failed:
      Tool call denied: <reason>"`` in the ``"user"``/``"assistant"``
      tool-result roles), so, like an ``after_tool`` note, the reason takes
      the observation's role: in the ``"user"`` role it carries user
      authority. ``after_tool`` does not run for it.

``after_tool(session_id, call_id, tool_name, args, result)``
    Runs once per dispatched call that returned a
    :class:`~fifty_agent_sdk.tools.protocol.ToolResult`: a success
    (``ObservationEvent``) or ``is_error=True`` (``ToolFailedEvent``: a tool
    that reported failure, an ``@tool`` argument-validation failure, an MCP
    ``isError`` result). Never for ``ToolNotFound``, ``ToolTimeout``, a denied
    call or a fatal error: no ``ToolResult`` exists there. It runs AFTER that
    call's terminal event has been yielded (under a Runner: after the
    consumer handled the event and ``on_tool_end`` fired) and before the
    observation joins the loop's working message list; in a batch, calls are
    handled one at a time in call order. ``args`` is the hook's own deep copy
    of the dispatched arguments (the replacement, when ``before_tool``
    returned a valid one), taken after the tool returns, so nested values
    may carry in-place edits the tool made to its own arguments (see
    "Argument copies"). ``result`` is the registry's ``ToolResult`` by
    identity, the same object ``ObservationEvent.result`` carries: the SDK
    neither copies it nor writes to it. Treat ``result`` as READ-ONLY.

    A non-blank ``str`` return is appended verbatim (not stripped, not
    truncated) to that call's model-facing observation, after a blank line.
    There is one append point, after the tool-result role is chosen, so the
    same suffix lands in every tool mode and role: the ``role="tool"`` reply
    (``ToolMode.NATIVE`` and the legacy default) and the collapsed
    ``"user"``/``"assistant"`` message (``ToolMode.JSON``/``PROSE`` and the
    legacy option). ``None`` or a blank string adds nothing. The SDK never
    writes to the ``ToolResult``, and the events are the ones the run emits
    without the hook. The note inherits the
    observation's role, so phrase it neutrally and factually: in the
    ``"user"`` role it carries user authority, and in the ``"assistant"`` role
    the model reads it as its own words. Bounding its length is the host's
    job.

Argument copies
    Each hook gets its own deep copy of the arguments. ``before_tool``
    copies the model's arguments before dispatch. ``after_tool`` copies the
    dispatched arguments after the tool returns; the tool itself received a
    one-level copy, as in 1.9.0, so nested values may carry in-place edits the
    tool made to its own arguments. Nothing a hook does to its copy, at any
    depth, reaches the dispatch, the events, the model's assistant turn or the
    other hook. The ``ToolResult`` is not copied.

    The copy walks exact ``dict`` and ``list`` containers without recursion,
    however deeply they nest, when each is reached from the arguments through
    other exact dicts and lists. Exact ``str``, ``int``, ``float``, ``bool``
    and ``None`` values are kept as they are, since they are immutable. Any
    other value (a ``dict`` or ``list`` subclass, a tuple, any object) is
    copied with ``copy.deepcopy``, together with everything inside it, and
    shared references and cycles come out as ``copy.deepcopy`` makes them. The
    shipped parsers and LLM client decode arguments with ``json.loads``, which
    builds only exact dicts, lists and those scalars, so for them how deeply
    the model nests its arguments does not matter to the copy. Apart from
    running out of memory, only a non-JSON value that cannot be copied falls
    back, and only host code supplies one: a custom parser or a custom
    ``LLMClient`` that puts it into ``ToolCall.args``, or, for
    ``after_tool``'s copy, also a :class:`ReplaceToolArgs` replacement, a tool
    that stores one into its own nested arguments, or other code that stores
    one into the arguments' nested values (an event consumer writing into
    ``ActionEvent.args``, say). That hook then gets a shallow ``dict(args)``
    instead, isolated at the top level only, and one WARNING
    ``intervention.args_not_copyable`` is logged. That is not a hook failure:
    the hook still runs and no fallback applies.

    Change arguments with :class:`ReplaceToolArgs`, never by editing them in
    place, and alert on ``intervention.args_not_copyable``: it means host code
    supplied a value that cannot be copied, or the copy ran out of memory, so
    that hook's nested edits were not isolated.

Failure policy
    ``after_tool`` always fails soft: if it raises or returns something other
    than ``str`` or ``None``, the observation is left unaugmented. A failure
    costs only the note, so there is nothing to configure.

    ``before_tool`` follows :attr:`Interventions.before_tool_fallback` when
    it raises, returns something that is not ``None``, a
    :class:`DenyToolCall` or a :class:`ReplaceToolArgs`, or returns a
    :class:`ReplaceToolArgs` whose ``args`` is not a ``dict`` with ``str``
    keys:

    * :attr:`BeforeToolFallback.DENY` (the default) fails closed: the call is
      denied with the SDK's own reason ("this call was not approved, so it
      was not run.").
    * :attr:`BeforeToolFallback.ALLOW` fails open: the call runs with the
      model's ORIGINAL arguments, never the hook's (possibly mutated) copy and
      never an invalid replacement. The model sees the tool's real output and
      nothing about the failed check.

    A returned :class:`DenyToolCall` is honoured under both values. If its
    reason is unusable (reachable only through ``model_construct``), the SDK
    supplies its own reason and the call is still denied: the fallback covers
    only a decision that cannot be carried out as returned.

    Choose ``DENY`` for guards (block out-of-policy or dangerous calls) and
    for argument-scoping hooks (inject the current tenant, clamp a limit): a
    scoping hook that fails open runs the model's UNSCOPED arguments, which is
    a data-exposure path. Choose ``ALLOW`` only for advisory hooks, where
    availability matters more than the check (normalising argument formats,
    skipping calls the host already has results for). Never choose ``ALLOW``
    for a hook whose decision protects data.

    :class:`asyncio.CancelledError` propagates untouched from either hook, as
    do ``KeyboardInterrupt`` and ``SystemExit``.

Logging
    Under the fixed logger ``fifty_agent_sdk.interventions``, at ``WARNING``:
    ``intervention.hook_failed`` (``hook_name``, ``call_id``,
    ``error_type``, ``fallback``), ``intervention.hook_invalid``
    (``hook_name``, ``call_id``, ``returned_type``, ``reason``,
    ``fallback``) and ``intervention.args_not_copyable`` (``hook_name``,
    ``call_id``, ``error_type``; see "Argument copies"), where ``reason`` is
    ``not_a_string``,
    ``unrecognised_decision``, ``invalid_args`` or ``invalid_reason`` and
    ``fallback`` is ``observation_unaugmented``, ``call_denied`` or
    ``call_allowed``. A successful intervention logs ``intervention.applied``
    at ``DEBUG``. No line carries the exception text, the arguments, the
    result, the note, the reason text or the tool name (which the model may
    have invented); ``call_id`` joins a line to the call's
    ``ToolStartedEvent`` and audit payload. Alert on
    ``fallback="call_allowed"``: under ``ALLOW`` that line is the only sign
    that a check was skipped.

Persistence
    A note or a denial text is part of that call's observation in the loop's
    working list, so every later request IN THE SAME RUN carries it and the
    model never sees two versions of one observation. It is never written to
    the state store, and neither is any other tool observation
    (:mod:`fifty_agent_sdk.runner` persists the user message and the final
    answer only). A later turn sees only the final answer the model wrote
    after reading it.

Latency and sharing
    Hooks are awaited inline and one at a time, also within a batch, where N
    calls pay N hook latencies. Keep them fast (enqueue and return). A call's
    own terminal event is delivered before its ``after_tool`` runs; later
    events (the next batch member's terminal event, the next iteration) wait
    for it. One :class:`Interventions` instance may serve concurrent runs,
    so hook bodies must be safe for that.

What they cannot do
    ``after_tool`` cannot replace or remove observation text, change
    ``is_error`` or ``output``, or change an event. ``before_tool`` cannot
    rename a tool; deny instead. Screening MCP ``isError`` text stays the job
    of ``MCPClient(on_tool_error=...)``, which replaces that text;
    ``after_tool`` only appends, and the two compose.

Positional signatures
    Both hooks are called positionally with their identifiers first, in the
    same positions (``session_id, call_id, tool_name, args``). A signature
    therefore does not grow within 1.x: an extra parameter would break every
    implementor.
    New data arrives as a new hook or a new field, defaulting to ``None``.

No ``before_llm_call``
    There is no hook that rewrites the ``ChatRequest``. To change a request,
    wrap the :class:`~fifty_agent_sdk.llm.protocol.LLMClient` (for example
    with ``request.model_copy(update=...)``); a whole-request rewrite cannot
    be validated the way these two decisions are.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, TypeAlias, TypeVar, cast

import structlog
from pydantic import BaseModel, ConfigDict, field_validator

from fifty_agent_sdk.observability.hooks import _call_hook
from fifty_agent_sdk.tools.protocol import ToolResult

_log: Final = structlog.get_logger("fifty_agent_sdk.interventions")
"""Structured logger bound to the fixed name ``fifty_agent_sdk.interventions``.

Fixed (not ``__name__``) so a host can route or alert on intervention
fallbacks, which change a run, separately from ``hook.invoke_failed`` under
``fifty_agent_sdk.observability``, which does not.
"""

_AFTER_TOOL_SEPARATOR: Final = "\n\n"
"""Placed between an observation and the ``after_tool`` note (one blank line)."""

_DENIAL_PREFIX: Final = "Tool call denied: "
"""Prefix of every denial text: the ``ToolFailedEvent.error`` and the observation."""

_SDK_DENIAL_REASON: Final = "this call was not approved, so it was not run."
"""The reason the SDK supplies when ``before_tool`` fails under ``DENY``, or when a
returned :class:`DenyToolCall` carries an unusable reason."""


class BeforeToolFallback(StrEnum):
    """What happens when ``before_tool`` fails. Pass as ``Interventions(before_tool_fallback=...)``.

    A ``StrEnum``, so configuration-driven callers may pass the plain strings
    ``"deny"`` or ``"allow"``. Any other value raises :class:`ValueError` when
    the :class:`Interventions` is constructed.

    "Fails" means ``before_tool`` raised an :class:`Exception`, returned
    something that is not ``None``, a :class:`DenyToolCall` or a
    :class:`ReplaceToolArgs`, or returned a :class:`ReplaceToolArgs` whose
    ``args`` is not a ``dict`` with ``str`` keys: a decision that cannot be
    carried out as returned. A returned :class:`DenyToolCall` is honoured
    under both values, even one whose reason is unusable (the SDK then
    supplies the reason). Every fallback logs a WARNING whose ``fallback``
    field is ``call_denied`` or ``call_allowed``.

    Attributes:
        DENY: The default. Fails closed: the call is not run, the model reads
            ``"Tool call denied: this call was not approved, so it was not
            run."`` and the host gets a ``ToolFailedEvent``. Use it for
            guards and for argument-scoping hooks. A scoping hook (inject the
            tenant, clamp a limit) that failed open would run the model's
            unscoped arguments.
        ALLOW: Fails open: the call runs with the model's ORIGINAL arguments
            (never the hook's copy, never an invalid replacement), the model
            sees the tool's real output, and ``after_tool`` runs as usual. Use
            it only for advisory hooks, where availability matters more than
            the check, and never for a hook that protects data. The WARNING
            with ``fallback="call_allowed"`` is the only signal, so alert on
            it.
    """

    DENY = "deny"
    ALLOW = "allow"


class DenyToolCall(BaseModel):
    """A ``before_tool`` decision: do not run this call, and tell the model why.

    The model reads ``"Tool call denied: <reason>"`` as the call's
    observation, and the host sees a ``ToolFailedEvent`` carrying the same
    text. Honoured under both :class:`BeforeToolFallback` values.

    Attributes:
        reason: Model-facing explanation. Must be a non-blank string; a blank
            or whitespace-only reason raises a pydantic ``ValidationError``
            here, at the hook's return site.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("DenyToolCall.reason must be a non-blank string")
        return value


class ReplaceToolArgs(BaseModel):
    """A ``before_tool`` decision: run this call with these arguments instead.

    The loop dispatches a one-level copy of ``args``, as it does the model's
    arguments. ``ActionEvent.args`` and ``after_tool`` see the replacement; it
    is never written into the model's own assistant turn. The tool name cannot
    be changed this way.

    Attributes:
        args: The arguments to dispatch, the same type as
            :attr:`fifty_agent_sdk.llm.types.ToolCall.args` (``str`` keys).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    args: dict[str, Any]


BeforeToolHook: TypeAlias = Callable[
    [str | None, str, str, dict[str, Any]],
    DenyToolCall | ReplaceToolArgs | None | Awaitable[DenyToolCall | ReplaceToolArgs | None],
]
"""Public type of ``Interventions.before_tool`` (FR-003).

Called positionally as ``hook(session_id, call_id, tool_name, args)`` once per
model tool call, before dispatch:

* ``session_id``: the Runner's session id, or ``None`` when the loop runs
  without a :class:`~fifty_agent_sdk.runner.AgentRunner` (hence
  ``str | None``).
* ``call_id``: the id the call's ``ToolStartedEvent`` and terminal event
  carry, and, on the ``role="tool"`` paths, the reply's ``tool_call_id``.
* ``tool_name``: the name the model asked for. It may not be registered.
* ``args``: the hook's own deep copy of the model's arguments, taken before
  dispatch. Nothing the hook does to it, at any depth, reaches the dispatch,
  the events, the model's turn or ``after_tool`` (top-level only when a
  non-JSON value in them cannot be copied or the copy runs out of memory; see
  the module docstring, "Argument copies").

Returns ``None`` (proceed), a :class:`ReplaceToolArgs` or a
:class:`DenyToolCall`, or an awaitable of one of those. The RETURN VALUE is
inspected with :func:`inspect.isawaitable`, so ``async def``, plain ``def``,
callable objects and :func:`functools.partial` all work.
"""

AfterToolHook: TypeAlias = Callable[
    [str | None, str, str, dict[str, Any], ToolResult],
    str | None | Awaitable[str | None],
]
"""Public type of ``Interventions.after_tool`` (FR-003).

Called positionally as ``hook(session_id, call_id, tool_name, args, result)``
once per dispatched call that returned a :class:`~fifty_agent_sdk.tools.
protocol.ToolResult`, after that call's terminal event:

* ``session_id``: as for :data:`BeforeToolHook` (``None`` without a Runner).
* ``call_id``: the id the call's ``ObservationEvent`` or ``ToolFailedEvent``
  carried, so a host can key "did my UI render this call" on it.
* ``tool_name``: the registered tool that ran.
* ``args``: the hook's own deep copy of the dispatched arguments, taken after
  the tool returns, so nested values may carry in-place edits the tool made to
  its own arguments. Changing it affects nothing else (one level only when a
  non-JSON value in them cannot be copied or the copy runs out of memory).
* ``result``: the registry's ``ToolResult`` by identity (the object the
  ``ObservationEvent`` carries). READ-ONLY; the SDK neither copies it nor
  writes to it.

Returns a note to append to the model-facing observation, or ``None``/a blank
string for no note, or an awaitable of that. Sync and async callables both
work, as for :data:`BeforeToolHook`.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Interventions:
    """Optional hooks whose return values the loop honours. Pass as ``AgentLoop(interventions=...)``.

    A frozen, keyword-only dataclass like
    :class:`~fifty_agent_sdk.observability.hooks.Hooks`, so one instance can
    be shared across concurrent runs. Every hook defaults to ``None``; with
    none set the loop behaves as without ``interventions``. A field added
    later also defaults to ``None``, so adding one is a minor change. See the
    module docstring for the full contract.

    Attributes:
        before_tool: A :data:`BeforeToolHook`, or ``None``. Runs before each
            model tool call and may deny it (:class:`DenyToolCall`) or
            replace its arguments (:class:`ReplaceToolArgs`). Its failure
            behaviour is ``before_tool_fallback``.
        after_tool: An :data:`AfterToolHook`, or ``None``. Runs after each
            dispatched call that returned a ``ToolResult``, once its terminal
            event has been yielded, and may return a note that is appended to
            that call's model-facing observation. It always fails soft: a
            raise or an invalid return adds no note.
        before_tool_fallback: What a failed ``before_tool`` does, as a
            :class:`BeforeToolFallback` or its string value. The default
            ``DENY`` fails closed (the call is denied); ``ALLOW`` fails open
            (the call runs with the model's original arguments). It has no
            effect without ``before_tool``, which is not an error, so a host
            may set it globally and install the hook conditionally.

    Raises:
        TypeError: When ``before_tool`` or ``after_tool`` is neither ``None``
            nor callable.
        ValueError: When ``before_tool_fallback`` is not a
            :class:`BeforeToolFallback` or one of ``"deny"``, ``"allow"``.
    """

    before_tool: BeforeToolHook | None = None
    after_tool: AfterToolHook | None = None
    before_tool_fallback: BeforeToolFallback = BeforeToolFallback.DENY

    def __post_init__(self) -> None:
        # A hook that cannot be called would fail on every tool call at run
        # time, so reject it at construction.
        for field_name in ("before_tool", "after_tool"):
            hook = getattr(self, field_name)
            if hook is not None and not callable(hook):
                raise TypeError(
                    f"Interventions.{field_name} must be callable or None; "
                    f"got {type(hook).__name__}"
                )
        # Normalise the fallback to its member (plain strings are accepted,
        # like ToolMode), so the loop reads one resolved value. Frozen, hence
        # object.__setattr__.
        value = self.before_tool_fallback
        try:
            fallback = BeforeToolFallback(value)
        except ValueError:
            valid = ", ".join(repr(member.value) for member in BeforeToolFallback)
            raise ValueError(
                f"unknown before_tool_fallback={value!r}; "
                f"expected a BeforeToolFallback or one of {valid}"
            ) from None
        object.__setattr__(self, "before_tool_fallback", fallback)


@dataclass(frozen=True, slots=True)
class _BeforeToolOutcome:
    """What the loop does with one tool call after ``before_tool`` (FR-003 D5).

    Attributes:
        args: The arguments to dispatch, and to show on ``ActionEvent``. The
            loop's own (original) object when the call proceeds, falls open or
            is denied; a one-level copy of the replacement for
            :class:`ReplaceToolArgs`.
        denial: The full model-facing denial text (``"Tool call denied:
            ..."``) when the call must not run; ``None`` otherwise.
    """

    args: dict[str, Any]
    denial: str | None = None


def _fallback_outcome(fallback: BeforeToolFallback, args: dict[str, Any]) -> _BeforeToolOutcome:
    """The single policy branch for a ``before_tool`` decision that cannot be carried out.

    ``ALLOW`` dispatches ``args``, the loop's ORIGINAL object (never the
    hook's copy, never an invalid replacement). ``DENY`` denies with the
    SDK's own reason.
    """
    if fallback is BeforeToolFallback.ALLOW:
        return _BeforeToolOutcome(args=args)
    return _BeforeToolOutcome(args=args, denial=_DENIAL_PREFIX + _SDK_DENIAL_REASON)


def _fallback_label(outcome: _BeforeToolOutcome) -> str:
    """The WARNING's ``fallback`` value: what actually happened to the call."""
    return "call_denied" if outcome.denial is not None else "call_allowed"


_T = TypeVar("_T")

_JSON_SCALAR_TYPES: Final[frozenset[type[object]]] = frozenset({str, int, float, bool, type(None)})
"""The exact types of the five JSON scalars, which :func:`_iterative_deepcopy` keeps as they are (TD-010).

Matched by exact type, as ``copy.deepcopy`` matches its atomic types: an
instance of a subclass (a ``str`` subclass with instance attributes, say) can
carry mutable state, so it is copied like any other value.
"""


def _iterative_deepcopy(value: _T) -> _T:
    """Return a deep copy of ``value`` that walks exact ``dict`` and ``list`` nesting without recursion (TD-010).

    ``copy.deepcopy`` recurses for every nesting level, so JSON nested 500
    levels deep, which ``json.loads`` decodes, already exhausts the default
    recursion limit (measured on Python 3.14.3). This copy never recurses for
    the containers ``json.loads`` builds:

    * A container whose type is exactly ``dict`` or exactly ``list``, reached
      from ``value`` through such containers, is copied on an explicit stack,
      so its nesting depth is limited by memory, not by the recursion limit.
      As in ``copy.deepcopy``, each copy starts empty, gets its memo entry
      before any child and is filled in order. Unlike ``copy.deepcopy``, a
      nested container is filled only after the rest of its parent, so a
      non-JSON value whose copy reads another container of the tree (from
      ``__deepcopy__`` or ``__setstate__``) may see that container's copy
      still empty.
    * A value whose type is exactly ``str``, ``int``, ``float``, ``bool`` or
      ``NoneType`` is returned as it is, because it is immutable.
    * Every other value, including a key of any other type and a ``dict`` or
      ``list`` subclass, is copied with ``copy.deepcopy`` and the walk's memo,
      together with everything inside it: a ``dict`` inside a tuple is copied
      by ``copy.deepcopy``, with recursion. Shared references and cycles come
      out as ``copy.deepcopy`` makes them: an object that gets a copy gets
      exactly one, whether the walk or ``copy.deepcopy`` meets it first.

    The walk's own work and output are linear in what it walks: each
    distinct exact dict and list is processed once and each reference in it
    followed once, so a diamond of any width built from them costs one copy
    per container, and ``json.loads`` builds only trees of them. A value
    handed to ``copy.deepcopy`` costs what ``copy.deepcopy`` costs, which is
    not always linear: it never memoizes a value that copies to itself, such
    as a tuple of immutables, so a diamond of such tuples takes time
    exponential in its depth (measured on Python 3.14.3: about four times as
    long for every two more levels), as in 1.10.0; only host code can supply
    one. There is deliberately no node budget: the walk's output never
    outgrows an input the producer has already built, a budget would only
    add a fallback that wide but valid input could trigger, and it could not
    see the work done inside ``copy.deepcopy``.

    Args:
        value: The value to copy. Never modified: the walk only reads it.

    Returns:
        The copy.

    Raises:
        Exception: Whatever ``copy.deepcopy`` raises for a value that is not
            JSON, including ``RecursionError`` when that value, or JSON inside
            it, nests past the recursion limit (a 5000-level ``OrderedDict``
            chain, or a tuple holding a 5000-level dict). Never
            ``RecursionError`` for ``dict`` or ``list`` nesting reached from
            ``value`` through exact dicts and lists, however deep. Nothing is
            caught here: the fallback for a copy that fails belongs to the
            caller, :func:`_hook_args`.
        RuntimeError: When a non-JSON value's copy resizes a dict that the
            walk is iterating, as ``copy.deepcopy`` raised in 1.10.0.
        MemoryError: When the copy runs out of memory.
    """
    scalar_types = _JSON_SCALAR_TYPES
    value_type = type(value)
    if value_type in scalar_types:
        return value
    # One memo for the whole copy, shared with every copy.deepcopy call below.
    memo: dict[int, Any] = {}
    if value_type is not dict and value_type is not list:
        return copy.deepcopy(value, memo)
    root: dict[Any, Any] | list[Any] = {} if value_type is dict else []
    memo[id(value)] = root
    stack: list[tuple[Any, Any]] = [(value, root)]
    while stack:
        source, target = stack.pop()
        is_dict = type(source) is dict
        for key, child in source.items() if is_dict else enumerate(source):
            child_type = type(child)
            if child_type in scalar_types:
                child_copy = child
            elif child_type is dict or child_type is list:
                # TD-010: the memo entry is made when a container is first
                # met, before its children, as copy.deepcopy does. A second
                # reference or a cycle back to it resolves to this one copy,
                # so each container is pushed once: that keeps the walk
                # linear for a cycle and for a diamond of any width.
                child_id = id(child)
                if child_id in memo:
                    child_copy = memo[child_id]
                else:
                    child_copy = {} if child_type is dict else []
                    memo[child_id] = child_copy
                    stack.append((child, child_copy))
            else:
                # The same memo, so a non-JSON value that references a
                # container of this tree gets that container's copy (and
                # copy.deepcopy keeps its own temporaries alive in it).
                child_copy = copy.deepcopy(child, memo)
            if is_dict:
                # The value is copied before the key, as copy.deepcopy does.
                new_key = key if type(key) in scalar_types else copy.deepcopy(key, memo)
                target[new_key] = child_copy
            else:
                target.append(child_copy)
    return cast(_T, root)


def _hook_args(args: dict[str, Any], *, hook_name: str, call_id: str) -> dict[str, Any]:
    """The hook's own copy of ``args``: a deep copy, so nothing it does reaches the SDK.

    FR-003 review round 1: a shallow copy left nested values shared, so a hook
    editing ``args["filters"]`` in place changed what the tool received,
    ``ActionEvent.args`` and the model's replayed turn. TD-010: the copy is
    :func:`_iterative_deepcopy`, which walks exact ``dict`` and ``list``
    nesting reached from ``args`` through exact dicts and lists without
    recursion, so no JSON nesting depth reaches the fallback below. (1.10.0
    called ``copy.deepcopy``, which raised ``RecursionError`` for JSON nested
    500 levels deep on Python 3.14.3.) The shipped parsers and LLM client
    decode arguments with ``json.loads``, which builds nothing else. Apart
    from a ``MemoryError`` while copying, which nesting depth does not cause,
    the fallback covers only a non-JSON value that cannot be copied, and only
    host code supplies one: an object a custom parser or a custom
    ``LLMClient`` put into ``ToolCall.args``, or, for ``after_tool``'s copy,
    one in a ``ReplaceToolArgs`` replacement or one that a tool or other code
    (an event consumer writing into ``ActionEvent.args``, say) stored into the
    arguments' nested values. Then the hook gets a shallow ``dict(args)``
    instead, so its isolation is top-level only, and one WARNING
    ``intervention.args_not_copyable`` is logged. That is NOT a hook failure:
    the caller computes this outside the hook's ``try``, the hook still runs,
    and no fallback applies.

    Args:
        args: The arguments to copy. Never modified.
        hook_name: ``"before_tool"`` or ``"after_tool"``, for the log line.
        call_id: The call's id, for the log line.

    Returns:
        A deep copy of ``args``, or a one-level one when a non-JSON value in
        it cannot be copied or the copy runs out of memory.
    """
    try:
        # A module global looked up at call time, so a test can spy on it.
        return _iterative_deepcopy(args)
    except Exception as exc:
        # Type only, never str(exc): the message may quote the arguments.
        _log.warning(
            "intervention.args_not_copyable",
            hook_name=hook_name,
            call_id=call_id,
            error_type=type(exc).__name__,
        )
        return dict(args)


async def _apply_after_tool(
    hook: AfterToolHook,
    *,
    session_id: str | None,
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
    result: ToolResult,
) -> str | None:
    """Run ``after_tool`` for one call and return the note to append, or ``None``.

    Fails soft (FR-003 D2, D4): a raising hook or a non-``str`` return yields
    ``None`` and a WARNING; ``None`` or a blank string yields ``None`` with
    nothing logged; any other ``str`` is returned verbatim.

    Args:
        hook: The configured ``after_tool``.
        session_id: Forwarded to the hook.
        call_id: Forwarded to the hook and logged.
        tool_name: Forwarded to the hook. Never logged.
        args: The dispatched arguments, as they are after the tool returned
            (the tool got a one-level copy, so nested values may carry its
            in-place edits); the hook receives its own deep copy
            (:func:`_hook_args`).
        result: The registry's ``ToolResult``; the hook receives it by identity.

    Returns:
        The note, or ``None`` for no note.

    Raises:
        asyncio.CancelledError: Re-raised untouched if the hook raises it.
    """
    # The copy is made OUTSIDE the hook's try: a copy problem is not a hook failure.
    hook_args = _hook_args(args, hook_name="after_tool", call_id=call_id)
    try:
        returned: object = await _call_hook(hook, session_id, call_id, tool_name, hook_args, result)
    # Ordering is load-bearing (coding_guidelines §2, §9b): the CancelledError
    # arm precedes the Exception arm so cancellation propagates instead of
    # falling back.
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Type only, never str(exc): the hook holds the tool's output and may
        # embed it in its own message. Unlike `mcp.tool_error_hook_failed`,
        # this line carries `call_id`, never `tool_name` (FR-003 D4).
        _log.warning(
            "intervention.hook_failed",
            hook_name="after_tool",
            call_id=call_id,
            error_type=type(exc).__name__,
            fallback="observation_unaugmented",
        )
        return None
    if returned is None:
        return None
    if not isinstance(returned, str):
        _log.warning(
            "intervention.hook_invalid",
            hook_name="after_tool",
            call_id=call_id,
            returned_type=type(returned).__name__,
            reason="not_a_string",
            fallback="observation_unaugmented",
        )
        return None
    # Blank is "no note", not an error: this hook supplies an optional
    # addition, unlike `on_tool_error`, which must supply a replacement.
    if not returned.strip():
        return None
    _log.debug(
        "intervention.applied",
        hook_name="after_tool",
        call_id=call_id,
        outcome="augmented",
        text_len=len(returned),
    )
    return returned


async def _apply_before_tool(
    hook: BeforeToolHook,
    *,
    fallback: BeforeToolFallback,
    session_id: str | None,
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
) -> _BeforeToolOutcome:
    """Run ``before_tool`` for one call and return what the loop must do (FR-003 D5).

    Validation is total: the decision is re-checked here even though its type
    validates at construction, because ``model_construct()`` bypasses that,
    and fields are read with a default so a missing one is invalid rather than
    an ``AttributeError``.

    * ``None``: proceed with ``args`` (the loop's own object).
    * :class:`ReplaceToolArgs` with a ``dict`` of ``str`` keys: dispatch a
      one-level copy of it.
    * :class:`DenyToolCall`: deny with ``"Tool call denied: <reason>"``, or
      with the SDK reason when its reason is not a non-blank ``str``, under
      BOTH fallbacks.
    * A raise, any other return, or an invalid :class:`ReplaceToolArgs`:
      :func:`_fallback_outcome` for ``fallback``.

    Args:
        hook: The configured ``before_tool``.
        fallback: The loop's resolved ``Interventions.before_tool_fallback``.
        session_id: Forwarded to the hook.
        call_id: Forwarded to the hook and logged.
        tool_name: Forwarded to the hook. Never logged: the model may have
            invented it.
        args: The model's arguments; the hook receives its own deep copy
            (:func:`_hook_args`), taken before dispatch. Returned by identity
            when the call proceeds or falls open, so the loop goes on from the
            model's own object exactly as it does without a hook.

    Returns:
        The call's outcome.

    Raises:
        asyncio.CancelledError: Re-raised untouched if the hook raises it.
    """
    # The copy is made OUTSIDE the hook's try: a copy problem is not a hook failure.
    hook_args = _hook_args(args, hook_name="before_tool", call_id=call_id)
    try:
        decision: object = await _call_hook(hook, session_id, call_id, tool_name, hook_args)
    # Ordering is load-bearing (coding_guidelines §2, §9b): the CancelledError
    # arm precedes the Exception arm so cancellation propagates instead of
    # being turned into a fallback.
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Type only, never str(exc): the hook holds the call's arguments and
        # may embed them in its own message. Unlike `mcp.tool_error_hook_failed`,
        # this line carries `call_id`, never `tool_name`, which the model may
        # have invented (FR-003 D4).
        outcome = _fallback_outcome(fallback, args)
        _log.warning(
            "intervention.hook_failed",
            hook_name="before_tool",
            call_id=call_id,
            error_type=type(exc).__name__,
            fallback=_fallback_label(outcome),
        )
        return outcome
    if decision is None:
        return _BeforeToolOutcome(args=args)
    if isinstance(decision, DenyToolCall):
        reason = getattr(decision, "reason", None)
        if not isinstance(reason, str) or not reason.strip():
            # The hook's decision is honoured whatever the fallback: the SDK
            # can always carry out a deny by supplying its own reason.
            outcome = _BeforeToolOutcome(args=args, denial=_DENIAL_PREFIX + _SDK_DENIAL_REASON)
            _log.warning(
                "intervention.hook_invalid",
                hook_name="before_tool",
                call_id=call_id,
                returned_type=type(decision).__name__,
                reason="invalid_reason",
                fallback=_fallback_label(outcome),
            )
            return outcome
        _log.debug(
            "intervention.applied",
            hook_name="before_tool",
            call_id=call_id,
            outcome="denied",
            text_len=len(reason),
        )
        return _BeforeToolOutcome(args=args, denial=_DENIAL_PREFIX + reason)
    if isinstance(decision, ReplaceToolArgs):
        replacement = getattr(decision, "args", None)
        if isinstance(replacement, dict) and all(isinstance(key, str) for key in replacement):
            _log.debug(
                "intervention.applied",
                hook_name="before_tool",
                call_id=call_id,
                outcome="replaced",
            )
            return _BeforeToolOutcome(args=dict(replacement))
        # The replacement IS the decision, so it cannot be carried out: the
        # fallback applies, and even ALLOW never dispatches it.
        outcome = _fallback_outcome(fallback, args)
        _log.warning(
            "intervention.hook_invalid",
            hook_name="before_tool",
            call_id=call_id,
            returned_type=type(decision).__name__,
            reason="invalid_args",
            fallback=_fallback_label(outcome),
        )
        return outcome
    # On a gate, every non-None return must be a deliberate, named decision:
    # a stray str/dict/bool is not read as deny or replace.
    outcome = _fallback_outcome(fallback, args)
    _log.warning(
        "intervention.hook_invalid",
        hook_name="before_tool",
        call_id=call_id,
        returned_type=type(decision).__name__,
        reason="unrecognised_decision",
        fallback=_fallback_label(outcome),
    )
    return outcome


__all__ = [
    "AfterToolHook",
    "BeforeToolFallback",
    "BeforeToolHook",
    "DenyToolCall",
    "Interventions",
    "ReplaceToolArgs",
]
