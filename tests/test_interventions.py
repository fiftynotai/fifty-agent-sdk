"""Unit tests for :mod:`fifty_agent_sdk.interventions` (FR-003).

What these pin, below the loop:

* the public types: :class:`Interventions` (frozen, keyword-only, callable
  checks, the ``before_tool_fallback`` default, validation and string
  normalisation), :class:`BeforeToolFallback`, :class:`DenyToolCall`,
  :class:`ReplaceToolArgs`, and the six root exports;
* ``_apply_after_tool``: the note is returned verbatim; ``None`` and blank
  mean no note; a non-``str`` return and a raise fall back to no note with a
  type-only WARNING; cancellation propagates;
* ``_apply_before_tool``: proceed, replace and deny; every failure class under
  BOTH fallbacks, including that fail-open dispatches the ORIGINAL args object
  (never the hook's mutated copy, never an invalid replacement) and that a
  returned deny is honoured under both; cancellation propagates;
* AC-5: every callable shape works for both hooks.
* argument copies (review round 1, TD-010): each hook edits its own deep copy,
  nested values included, and JSON args nested 5000 levels deep are copied in
  full with no WARNING; args holding a non-JSON value ``copy.deepcopy``
  rejects reach the hook as a one-level copy with one
  ``intervention.args_not_copyable`` WARNING, and the hook still runs;
* ``_iterative_deepcopy`` (TD-010): the same result as ``copy.deepcopy`` on
  values ``json.loads`` never builds (types, values and which objects are
  shared), shared references, cycles, a memo shared with ``copy.deepcopy``'s
  copies of such values and of non-``str`` keys, and a 64-level diamond
  copied once per object.

The loop-level behaviour (where the helpers are called, and what reaches the
wire and the event stream) is pinned in ``tests/loop/test_loop_interventions.py``.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import inspect
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
import structlog
from pydantic import ValidationError

import fifty_agent_sdk
from fifty_agent_sdk import (
    AgentLoop,
    ChatMessage,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolFailedEvent,
    ToolMode,
    ToolResult,
)
from fifty_agent_sdk import interventions as interventions_module
from fifty_agent_sdk.interventions import (
    BeforeToolFallback,
    DenyToolCall,
    Interventions,
    ReplaceToolArgs,
    _apply_after_tool,
    _apply_before_tool,
    _iterative_deepcopy,
)
from fifty_agent_sdk.observability import hooks as hooks_module
from tests.loop.conftest import FakeLLMClient, FakeTool, make_response

_SDK_DENIAL = "Tool call denied: this call was not approved, so it was not run."
_SECRET = "SECRET-intervention-payload-DO-NOT-LOG"
_BOTH_FALLBACKS = [
    pytest.param(BeforeToolFallback.DENY, id="deny"),
    pytest.param(BeforeToolFallback.ALLOW, id="allow"),
]
_FAILED_KEYS = {"event", "log_level", "hook_name", "call_id", "error_type", "fallback"}
_INVALID_KEYS = {
    "event",
    "log_level",
    "hook_name",
    "call_id",
    "returned_type",
    "reason",
    "fallback",
}


# --- Helpers ------------------------------------------------------------------


def _intervention_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``intervention.*`` WARNING entries (DEBUG ``intervention.applied`` excluded)."""
    return [
        entry
        for entry in logs
        if str(entry.get("event", "")).startswith("intervention.")
        and entry.get("log_level") == "warning"
    ]


async def _after(hook: Callable[..., Any], **overrides: Any) -> str | None:
    kwargs: dict[str, Any] = {
        "session_id": "s-1",
        "call_id": "call-1",
        "tool_name": "search",
        "args": {"q": "weather"},
        "result": ToolResult(output={"rows": 2}),
    }
    kwargs.update(overrides)
    return await _apply_after_tool(hook, **kwargs)


async def _before(
    hook: Callable[..., Any],
    args: dict[str, Any],
    *,
    fallback: BeforeToolFallback = BeforeToolFallback.DENY,
) -> Any:
    return await _apply_before_tool(
        hook,
        fallback=fallback,
        session_id="s-1",
        call_id="call-1",
        tool_name="search",
        args=args,
    )


def _returning(value: object) -> Callable[..., object]:
    def hook(*_args: object) -> object:
        return value

    return hook


# --- The Interventions container ------------------------------------------------


def test_interventions_fields_default_to_none_and_are_keyword_only() -> None:
    """Both hooks default to ``None``; the dataclass is frozen and keyword-only, and so is the loop kwarg (FR-003 D1)."""
    empty = Interventions()
    assert empty.before_tool is None
    assert empty.after_tool is None
    assert [f.name for f in dataclasses.fields(Interventions)] == [
        "before_tool",
        "after_tool",
        "before_tool_fallback",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        empty.after_tool = _returning(None)  # type: ignore[misc]
    with pytest.raises(TypeError):
        Interventions(_returning(None))  # type: ignore[misc]
    parameter = inspect.signature(AgentLoop).parameters["interventions"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("field_name", ["before_tool", "after_tool"])
@pytest.mark.parametrize("value", ["not callable", 42, {"a": 1}])
def test_interventions_rejects_non_callable_hook(field_name: str, value: object) -> None:
    """A non-callable hook raises ``TypeError`` naming the field, at construction (FR-003 D1)."""
    with pytest.raises(TypeError, match=field_name):
        Interventions(**{field_name: value})


def test_intervention_names_are_exported_from_package_root() -> None:
    """The six public names are root exports, identical to the module's; privates stay private (FR-003 D1)."""
    public = [
        "AfterToolHook",
        "BeforeToolFallback",
        "BeforeToolHook",
        "DenyToolCall",
        "Interventions",
        "ReplaceToolArgs",
    ]
    assert sorted(interventions_module.__all__) == public
    for name in public:
        assert name in fifty_agent_sdk.__all__
        assert getattr(fifty_agent_sdk, name) is getattr(interventions_module, name)
    for private in (
        "_call_hook",
        "_apply_after_tool",
        "_apply_before_tool",
        "_BeforeToolOutcome",
        "_DeniedCall",
    ):
        assert private not in fifty_agent_sdk.__all__
        assert private not in interventions_module.__all__
    assert hooks_module.__all__ == ["Hooks", "invoke_hook"]


# --- Decision types -------------------------------------------------------------


@pytest.mark.parametrize("reason", ["", "  ", "\n"])
def test_deny_tool_call_rejects_blank_reason(reason: str) -> None:
    """A blank or whitespace-only reason fails at the hook's return site; the model is frozen and closed (FR-003 D5)."""
    with pytest.raises(ValidationError):
        DenyToolCall(reason=reason)
    decision = DenyToolCall(reason="not in this tenant")
    with pytest.raises(ValidationError):
        decision.reason = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DenyToolCall(reason="ok", extra="nope")  # type: ignore[call-arg]


def test_replace_tool_args_requires_str_keys() -> None:
    """``args`` must be a dict with ``str`` keys; the model is frozen and closed (FR-003 D5)."""
    assert ReplaceToolArgs(args={"q": 1}).args == {"q": 1}
    with pytest.raises(ValidationError):
        ReplaceToolArgs(args={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(ValidationError):
        ReplaceToolArgs(args=["x"])  # type: ignore[arg-type]
    decision = ReplaceToolArgs(args={"q": 1})
    with pytest.raises(ValidationError):
        decision.args = {}  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ReplaceToolArgs(args={}, extra="nope")  # type: ignore[call-arg]


# --- after_tool -----------------------------------------------------------------


async def test_apply_after_tool_returns_note_verbatim() -> None:
    """A non-blank note is returned verbatim, surrounding whitespace included (FR-003 D2)."""
    note = "  The user can already see these records.\n"
    assert await _after(_returning(note)) == note


@pytest.mark.parametrize("value", [None, "", "   ", "\n\t"], ids=["none", "empty", "spaces", "ws"])
async def test_apply_after_tool_treats_none_and_blank_as_no_note(value: str | None) -> None:
    """``None`` or a blank string means no note, and nothing is logged (FR-003 D2)."""
    with structlog.testing.capture_logs() as logs:
        assert await _after(_returning(value)) is None
    assert [e for e in logs if str(e.get("event", "")).startswith("intervention.")] == []


@pytest.mark.parametrize(
    "value",
    [424242, _SECRET.encode(), [_SECRET], {_SECRET: 1}],
    ids=["int", "bytes", "list", "dict"],
)
async def test_apply_after_tool_rejects_non_string_return(value: object) -> None:
    """A non-``str`` return gives no note and one type-only ``hook_invalid`` WARNING (FR-003 D4)."""
    with structlog.testing.capture_logs() as logs:
        assert await _after(_returning(value)) is None
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.hook_invalid"
    assert entry["hook_name"] == "after_tool"
    assert entry["call_id"] == "call-1"
    assert entry["returned_type"] == type(value).__name__
    assert entry["reason"] == "not_a_string"
    assert entry["fallback"] == "observation_unaugmented"
    assert set(entry) == _INVALID_KEYS
    assert _SECRET not in str(entry)
    assert "424242" not in str(entry)


async def test_apply_after_tool_swallows_raising_hook_and_logs_type_only() -> None:
    """A raising hook gives no note and one ``hook_failed`` WARNING with the type only (FR-003 AC-3)."""

    def hook(*_args: object) -> str:
        raise ValueError(f"cannot annotate {_SECRET}")

    with structlog.testing.capture_logs() as logs:
        assert await _after(hook) is None
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.hook_failed"
    assert entry["hook_name"] == "after_tool"
    assert entry["call_id"] == "call-1"
    assert entry["error_type"] == "ValueError"
    assert entry["fallback"] == "observation_unaugmented"
    assert "error_message" not in entry
    assert set(entry) == _FAILED_KEYS
    assert _SECRET not in str(entry)


async def test_apply_after_tool_reraises_cancelled_error_without_logging() -> None:
    """``CancelledError`` from ``after_tool`` propagates untouched and is not logged (FR-003 AC-3)."""

    async def hook(*_args: object) -> str:
        raise asyncio.CancelledError

    with structlog.testing.capture_logs() as logs:  # noqa: SIM117
        with pytest.raises(asyncio.CancelledError):
            await _after(hook)
    assert _intervention_logs(logs) == []


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("hook_name", ["after_tool", "before_deny", "before_allow"])
async def test_intervention_hooks_propagate_keyboard_interrupt_and_system_exit(
    hook_name: str, exc_type: type[BaseException]
) -> None:
    """``KeyboardInterrupt`` and ``SystemExit`` are not caught by either helper, under any fallback (FR-003 D4).

    This is what tells an ``except Exception`` arm from an ``except
    BaseException`` arm placed after the ``CancelledError`` arm; the
    ``CancelledError`` tests alone cannot.
    """

    def hook(*_args: object) -> object:
        raise exc_type

    with structlog.testing.capture_logs() as logs:  # noqa: SIM117
        with pytest.raises(exc_type):
            if hook_name == "after_tool":
                await _after(hook)
            else:
                fallback = (
                    BeforeToolFallback.DENY
                    if hook_name == "before_deny"
                    else BeforeToolFallback.ALLOW
                )
                await _before(hook, {"q": "x"}, fallback=fallback)
    assert _intervention_logs(logs) == []


# --- Argument copies (review round 1) ------------------------------------------------


class _NotDeepCopyable:
    """An argument value ``copy.deepcopy`` rejects, as a custom parser might produce."""

    def __deepcopy__(self, memo: dict[int, Any]) -> _NotDeepCopyable:
        raise TypeError("this handle cannot be copied")


def _deeply_nested(depth: int) -> dict[str, Any]:
    """A dict nested ``depth`` levels deep, built without recursion."""
    value: dict[str, Any] = {}
    for _ in range(depth):
        value = {"a": value}
    return value


# TD-010 deep fixtures. Each is built fresh for every run, without recursion,
# and is never passed to ``==``, ``repr``, ``json.dumps`` or ``copy.deepcopy``:
# all of them recurse, and would raise ``RecursionError`` inside the test.

_DEEP_LEVELS = 5000
"""How deeply the TD-010 fixtures nest: five times the default recursion limit."""

_MID_LEVEL = 2500
"""The level halfway down, where the TD-010 tests edit a hook's copy."""


def _list_chain(depth: int) -> list[Any]:
    """A list nested ``depth`` levels deep (``[[[...]]]``), built without recursion."""
    value: list[Any] = []
    for _ in range(depth):
        value = [value]
    return value


def _mixed_chain(depth: int) -> Any:
    """Dicts and lists alternating ``depth`` levels deep, built without recursion.

    Every level also holds scalars (``str`` and ``int`` in a dict level,
    ``float``, ``bool`` and ``None`` in a list level), so all five JSON scalar
    types occur all the way down. The child is under key ``"a"`` of a dict
    and at index 0 of a list, as in :func:`_deeply_nested` and
    :func:`_list_chain`.
    """
    value: Any = {}
    for level in range(depth):
        if level % 2:
            value = {"a": value, "s": "text", "n": level}
        else:
            value = [value, 1.5, True, False, None]
    return value


def _json_chain_holding(leaf: object, depth: int) -> dict[str, Any]:
    """A dict chain ``depth`` levels deep whose innermost dict holds ``leaf``, built without recursion."""
    value: dict[str, Any] = {"leaf": leaf}
    for _ in range(depth):
        value = {"a": value}
    return value


def _ordered_dict_chain(depth: int) -> OrderedDict[str, Any]:
    """An ``OrderedDict`` nested ``depth`` levels deep, built without recursion.

    A ``dict`` subclass is not JSON, so it goes to ``copy.deepcopy``: host
    code (a custom parser or ``LLMClient``) could supply one, ``json.loads``
    never does.
    """
    value: OrderedDict[str, Any] = OrderedDict()
    for _ in range(depth):
        value = OrderedDict(a=value)
    return value


def _descend(value: Any, levels: int) -> Any:
    """The container ``levels`` below ``value``: through key ``"a"`` of a dict, index 0 of a list."""
    for _ in range(levels):
        value = value["a"] if type(value) is dict else value[0]
    return value


def _edit_at_three_depths(args: dict[str, Any]) -> None:
    """What a careless hook does to deep args: edit the top level, level 2500 and the innermost container in place."""
    args["q"] = "changed"
    for level in (_MID_LEVEL, _DEEP_LEVELS):
        container = _descend(args["deep"], level)
        if type(container) is dict:
            container["edited"] = True
        else:
            container.append("edited")


def _tree_difference(actual: Any, expected: Any, *, copied: bool) -> str | None:
    """The first difference between two trees, found by a lockstep walk without recursion, or ``None``.

    At every node: the same exact type; for a ``dict`` the same keys in the
    same order, for a ``list`` the same length, for anything else an equal
    value. With ``copied=True``, ``actual`` must also be a full copy of
    ``expected``: no ``dict`` or ``list`` in it is the original object, and
    every other node is the original object (JSON scalars are immutable, so a
    copy keeps them). ``==`` is applied to keys and scalars only, never to a
    container (TD-010).
    """
    stack: list[tuple[Any, Any, int]] = [(actual, expected, 0)]
    while stack:
        got, want, depth = stack.pop()
        if type(got) is not type(want):
            return f"depth {depth}: {type(got).__name__} instead of {type(want).__name__}"
        if type(want) is dict or type(want) is list:
            if copied and got is want:
                return f"depth {depth}: a {type(want).__name__} shared with the original"
            if type(want) is dict:
                if list(got) != list(want):
                    return f"depth {depth}: different keys"
                stack.extend((got[key], want[key], depth + 1) for key in want)
            else:
                if len(got) != len(want):
                    return f"depth {depth}: length {len(got)} instead of {len(want)}"
                stack.extend((g, w, depth + 1) for g, w in zip(got, want, strict=True))
        elif copied and got is not want:
            return f"depth {depth}: a {type(want).__name__} that is not the original object"
        elif got != want:
            return f"depth {depth}: {got!r} instead of {want!r}"
    return None


async def test_apply_after_tool_hands_the_hook_a_deep_copy_of_args() -> None:
    """``after_tool`` gets its own deep copy of the dispatched args: its edits, nested ones included, leave the loop's args untouched (FR-003 review round 1)."""
    args = {"q": "open", "filters": {"tenant": "t-1", "tags": ["a"]}}
    expected = copy.deepcopy(args)
    seen: list[dict[str, Any]] = []

    def hook(*hook_args: Any) -> str:
        own = hook_args[3]
        seen.append(own)
        own["filters"]["tenant"] = "t-2"
        own["filters"]["tags"].append("b")
        del own["q"]
        return "note"

    assert await _after(hook, args=args) == "note"
    assert args == expected
    assert seen[0] is not args
    assert seen[0]["filters"] is not args["filters"]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_deeply_nested, id="dict_chain"),
        pytest.param(_list_chain, id="list_chain"),
        pytest.param(_mixed_chain, id="mixed_chain"),
    ],
)
@pytest.mark.parametrize("hook_name", ["before_tool", "after_tool"])
async def test_hooks_get_a_full_copy_of_json_args_nested_5000_levels(
    hook_name: str, build: Callable[[int], Any]
) -> None:
    """JSON args nested 5000 levels deep reach each hook as a full copy with no WARNING, and the hook's edits at every depth leave the args untouched (TD-010 AC-1).

    In 1.10.0 ``copy.deepcopy`` raised ``RecursionError`` here, so the hook got
    a one-level copy and its nested edits reached the args. The hook edits the
    top level, level 2500 and the innermost container. Trees are compared by a
    lockstep walk (:func:`_tree_difference`), never by ``==``, which recurses.
    Not pinned here: what the loop does with the args (the 5000-level test in
    ``tests/loop/test_loop_interventions.py``), and non-JSON values (the
    fallback test below and the ``_iterative_deepcopy`` tests).
    """
    args: dict[str, Any] = {"q": "x", "deep": build(_DEEP_LEVELS)}
    seen: list[dict[str, Any]] = []
    copy_differences: list[str | None] = []

    def hook(*hook_args: Any) -> str | None:
        own = hook_args[3]
        seen.append(own)
        copy_differences.append(_tree_difference(own, args, copied=True))
        _edit_at_three_depths(own)
        return "note" if hook_name == "after_tool" else None

    with structlog.testing.capture_logs() as logs:
        if hook_name == "after_tool":
            note = await _after(hook, args=args)
            proceeds_with_the_original = True
        else:
            outcome = await _before(hook, args)
            note = "note"
            proceeds_with_the_original = outcome.args is args and outcome.denial is None

    assert _intervention_logs(logs) == []
    assert note == "note"
    assert proceeds_with_the_original
    assert len(seen) == 1
    # Only scalars and short strings reach an assertion: pytest would repr
    # the deep trees if an assertion on them failed.
    deep_shared = seen[0]["deep"] is args["deep"]
    args_difference = _tree_difference(args, {"q": "x", "deep": build(_DEEP_LEVELS)}, copied=False)
    edited: dict[str, Any] = {"q": "x", "deep": build(_DEEP_LEVELS)}
    _edit_at_three_depths(edited)
    edits_difference = _tree_difference(seen[0], edited, copied=False)
    assert not deep_shared
    assert copy_differences == [None]
    assert args_difference is None
    assert edits_difference is None


@pytest.mark.parametrize(
    ("make_value", "error_type"),
    [
        pytest.param(_NotDeepCopyable, "TypeError", id="custom_object"),
        pytest.param(
            lambda: _json_chain_holding(_NotDeepCopyable(), _DEEP_LEVELS),
            "TypeError",
            id="custom_object_under_5000_json_levels",
        ),
        pytest.param(
            lambda: _ordered_dict_chain(_DEEP_LEVELS), "RecursionError", id="deep_dict_subclass"
        ),
    ],
)
@pytest.mark.parametrize("hook_name", ["before_tool", "after_tool"])
async def test_hooks_get_a_shallow_copy_when_args_cannot_be_deep_copied(
    hook_name: str, make_value: Callable[[], object], error_type: str
) -> None:
    """Args holding a non-JSON value ``copy.deepcopy`` rejects reach the hook as a one-level copy with one WARNING; the hook still runs and nothing is treated as a hook failure (FR-003 review round 1, TD-010).

    ``custom_object_under_5000_json_levels`` puts the uncopyable object under
    5000 levels of JSON: the copy reaches it without recursing, so the WARNING
    names its ``TypeError``, not a ``RecursionError``. ``deep_dict_subclass``
    is a 5000-level ``OrderedDict`` chain: not JSON, so ``copy.deepcopy``
    copies it and raises ``RecursionError``, which stays contained. JSON
    nested that deeply no longer falls back (TD-010; 1.10.0 pinned that
    fallback here as ``deep_nesting``). The one-level copy still keeps a
    top-level edit from leaking; a nested value is shared, which is the
    documented degradation.
    """
    value = make_value()
    args = {"q": "x", "handle": value}
    seen: list[dict[str, Any]] = []

    def hook(*hook_args: Any) -> str | None:
        own = hook_args[3]
        seen.append(own)
        own["q"] = "changed"
        return "note" if hook_name == "after_tool" else None

    with structlog.testing.capture_logs() as logs:
        if hook_name == "after_tool":
            note = await _after(hook, args=args)
            proceeds_with_the_original = True
        else:
            outcome = await _before(hook, args)
            note = "note"
            proceeds_with_the_original = outcome.args is args and outcome.denial is None

    # Scalars only in the assertions: some values nest past the recursion
    # limit, and pytest would repr them if an assertion on them failed.
    copy_is_new = seen[0] is not args if seen else False
    handle_shared = seen[0]["handle"] is value if seen else False
    assert note == "note"
    assert proceeds_with_the_original
    assert len(seen) == 1
    assert copy_is_new
    assert handle_shared
    assert args["q"] == "x"
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.args_not_copyable"
    assert entry["hook_name"] == hook_name
    assert entry["call_id"] == "call-1"
    assert entry["error_type"] == error_type
    assert set(entry) == {"event", "log_level", "hook_name", "call_id", "error_type"}


# --- _iterative_deepcopy (TD-010) -------------------------------------------------------


class _Box:
    """A plain object holding a value: not JSON, so ``copy.deepcopy`` copies it with its attributes."""

    def __init__(self, ref: Any) -> None:
        self.ref = ref


class _TaggedStr(str):
    """A ``str`` subclass with an instance attribute: not an exact ``str``, so it is copied, not kept."""

    tags: list[str]


def _mixed_corpus() -> dict[Any, Any]:
    """A shallow corpus of values ``json.loads`` never builds, around a small JSON core (TD-010 D1)."""
    tagged = _TaggedStr("label")
    tagged.tags = ["t1"]
    tagged_key = _TaggedStr("tagged key")
    tagged_key.tags = ["k1"]
    return {
        "json": {"s": "text", "i": 1, "f": 1.5, "b": True, "n": None, "l": ["a", 2]},
        "tuple_with_list": (1, ["inside", "a", "tuple"]),
        "tuple_of_scalars": (1, "a"),
        "ordered": OrderedDict([("k", [1, 2])]),
        "default": defaultdict(list, {"k": ["v"]}),
        "when": datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
        "amount": Decimal("12.50"),
        "raw": b"bytes",
        "buffer": bytearray(b"buffer"),
        "tags": {"x", "y"},
        "box": _Box(["i1", "i2"]),
        "tagged": tagged,
        7: "an int key",
        ("t", 1): "a tuple key",
        frozenset({"f"}): "a frozenset key",
        tagged_key: "a str subclass key",
        "list": [OrderedDict(a=[1]), (2, [3]), {"deeper": [{"k": "v"}]}],
    }


def _parity_difference(ours: Any, theirs: Any, original: Any) -> str | None:
    """Where ``ours`` first differs from ``theirs``, two copies of ``original``, or ``None``.

    At every node, keys included: the same exact type; an equal value (for an
    object with a ``__dict__``, equal attributes, compared node by node); a
    ``defaultdict``'s same ``default_factory``; and the same identity
    pattern: ``ours`` is the original object exactly when ``theirs`` is.
    Descends into dicts, lists, tuples and attributes; compares other values
    with ``==``.
    """
    stack: list[tuple[Any, Any, Any, str]] = [(ours, theirs, original, "root")]
    while stack:
        got, want, orig, where = stack.pop()
        if type(got) is not type(want):
            return f"{where}: {type(got).__name__} instead of {type(want).__name__}"
        if (got is orig) != (want is orig):
            return f"{where}: shared is {got is orig}, copy.deepcopy's shared is {want is orig}"
        if isinstance(orig, dict):
            if getattr(got, "default_factory", None) is not getattr(want, "default_factory", None):
                return f"{where}: a different default_factory"
            if not len(got) == len(want) == len(orig):
                return f"{where}: {len(got)} keys instead of {len(want)}"
            for got_key, want_key, orig_key in zip(got, want, orig, strict=True):
                stack.append((got_key, want_key, orig_key, f"{where} key {orig_key!r}"))
                stack.append(
                    (got[got_key], want[want_key], orig[orig_key], f"{where}[{orig_key!r}]")
                )
        elif isinstance(orig, list | tuple):
            if not len(got) == len(want) == len(orig):
                return f"{where}: length {len(got)} instead of {len(want)}"
            stack.extend(
                (g, w, o, f"{where}[{i}]")
                for i, (g, w, o) in enumerate(zip(got, want, orig, strict=True))
            )
        elif hasattr(orig, "__dict__"):
            if isinstance(orig, str) and got != want:
                return f"{where}: {got!r} instead of {want!r}"
            stack.append((vars(got), vars(want), vars(orig), f"{where}.__dict__"))
        elif got != want:
            return f"{where}: {got!r} instead of {want!r}"
    return None


def test_iterative_deepcopy_copies_a_list_root_nested_5000_levels() -> None:
    """A list nested 5000 levels deep, passed as the root, is copied in full with no ``RecursionError`` and no shared container (TD-010 D1).

    Hook args always have a ``dict`` root, so only a direct call reaches the
    root ``list`` case, and the hook tests cannot see it (the sentinel's
    ``b_root`` mutant). The copy is checked with the lockstep walker, never
    with ``==``. A ``RecursionError`` is caught and asserted on, so a
    regression fails in one line, not with a thousand-frame traceback.
    """
    original = _list_chain(_DEEP_LEVELS)
    raised_recursion_error = False
    copied: Any = None
    try:
        copied = _iterative_deepcopy(original)
    except RecursionError:
        raised_recursion_error = True

    assert not raised_recursion_error
    root_shared = copied is original
    difference = _tree_difference(copied, original, copied=True)
    assert not root_shared
    assert difference is None


def test_iterative_deepcopy_matches_copy_deepcopy_on_mixed_values() -> None:
    """On values ``json.loads`` never builds, the copy equals ``copy.deepcopy``'s in types, values and which objects are shared (TD-010 D1).

    The corpus covers a tuple holding a list and a tuple of scalars (which
    ``copy.deepcopy`` returns as it is), ``OrderedDict``, ``defaultdict``
    (its factory kept), ``datetime``, ``Decimal``, ``bytes``, ``bytearray``,
    ``set``, a plain object holding a list, a ``str`` subclass with an
    attribute, and ``int``, tuple, frozenset and ``str`` subclass keys. Some
    of its values are also copied as the root, since the root, the keys and
    the values are each dispatched on their own. It is shallow, so
    ``copy.deepcopy`` itself can copy it. Not pinned here: deep values (the
    5000-level tests), shared references and cycles (the tests below).
    """
    corpus = _mixed_corpus()
    roots = {
        "corpus": corpus,
        "ordered_root": corpus["ordered"],
        "tagged_root": corpus["tagged"],
        "tuple_root": corpus["tuple_with_list"],
        "str_root": "plain",
    }

    differences = {
        name: _parity_difference(_iterative_deepcopy(root), copy.deepcopy(root), root)
        for name, root in roots.items()
    }
    ours = _iterative_deepcopy(corpus)

    assert differences == dict.fromkeys(roots)
    assert ours is not corpus
    assert ours["default"].default_factory is list
    assert ours["tagged"].tags == ["t1"]
    assert ours["tagged"] is not corpus["tagged"]


def test_iterative_deepcopy_preserves_shared_references() -> None:
    """A dict referenced three times is copied once, and that copy is referenced three times, as ``copy.deepcopy`` does (TD-010 D1)."""
    shared: dict[str, Any] = {"k": ["v"]}
    value = {"a": shared, "b": [shared, {"c": shared}]}

    result = _iterative_deepcopy(value)

    assert result["a"] is result["b"][0]
    assert result["a"] is result["b"][1]["c"]
    assert result["a"] is not shared
    assert result["a"]["k"] is not shared["k"]
    assert result == value


def test_iterative_deepcopy_preserves_cycles() -> None:
    """A list that contains itself and a dict that holds the root copy to the same cycles, each through the copy (TD-010 D1, D2).

    Each container is pushed once, so a cycle ends the walk instead of
    feeding it.
    """
    self_list: list[Any] = ["x"]
    self_list.append(self_list)
    root: dict[str, Any] = {"child": {"items": [1]}}
    root["child"]["root"] = root

    list_copy = _iterative_deepcopy(self_list)
    root_copy = _iterative_deepcopy(root)

    assert list_copy is not self_list
    assert list_copy[1] is list_copy
    assert list_copy[0] == "x"
    assert root_copy is not root
    assert root_copy["child"] is not root["child"]
    assert root_copy["child"]["root"] is root_copy
    assert root_copy["child"]["items"] == [1]
    assert root_copy["child"]["items"] is not root["child"]["items"]


@pytest.mark.parametrize("leaf_first", [True, False], ids=["leaf_first", "dict_first"])
def test_iterative_deepcopy_shares_its_memo_with_leaf_copies(leaf_first: bool) -> None:
    """A non-JSON value that references a dict of the same tree gets that dict's copy, whichever of the two the walk meets first (TD-010 D1).

    ``leaf_first``: ``copy.deepcopy`` copies the dict while copying the leaf,
    and the walk then finds that copy in the shared memo. ``dict_first``: the
    walk registers its copy, and ``copy.deepcopy`` finds it there.
    """
    inner: dict[str, Any] = {"k": ["v"]}
    box = _Box(inner)
    value = {"leaf": box, "inner": inner} if leaf_first else {"inner": inner, "leaf": box}

    result = _iterative_deepcopy(value)

    assert result["leaf"] is not box
    assert result["leaf"].ref is result["inner"]
    assert result["inner"] is not inner
    assert result["inner"] == {"k": ["v"]}
    assert result["inner"]["k"] is not inner["k"]


@pytest.mark.parametrize("relation", ["key_is_also_a_value", "key_holds_a_dict_of_the_args"])
def test_iterative_deepcopy_shares_its_memo_with_key_copies(relation: str) -> None:
    """A non-``str`` key keeps its identity relation to the rest of the args, as with ``copy.deepcopy`` (TD-010 D1).

    Such a key is copied by ``copy.deepcopy`` with the walk's memo.
    ``key_is_also_a_value``: the key object is also a value in the args, and
    both must come out as one copy. ``key_holds_a_dict_of_the_args``: the key
    object holds a dict that is also in the args, and the key's copy must hold
    that dict's copy. A key copied with a fresh memo breaks both relations
    while every value still compares equal (the sentinel's ``c_key`` mutant).
    """
    inner: dict[str, Any] = {"k": ["v"]}
    if relation == "key_is_also_a_value":
        key = _Box(["x"])
        value: dict[Any, Any] = {"box": key, key: "keyed"}
    else:
        key = _Box(inner)
        value = {"inner": inner, key: "keyed"}

    def related(result: dict[Any, Any]) -> bool:
        copied_key = next(k for k in result if isinstance(k, _Box))
        if relation == "key_is_also_a_value":
            return result["box"] is copied_key
        return copied_key.ref is result["inner"]

    ours = _iterative_deepcopy(value)

    assert related(copy.deepcopy(value))
    assert related(ours)
    assert next(k for k in ours if isinstance(k, _Box)) is not key


def test_iterative_deepcopy_copies_diamond_fan_out_once_per_object() -> None:
    """A 64-level diamond (2**64 paths through 65 objects) copies to 65 new objects that share like the originals (TD-010 D2).

    coding_guidelines §11b: a depth check does not bound width or fan-out.
    Here the memo does: each level's two references resolve to one copy, so
    the work is one step per object and reference. Without the memo the walk
    would follow every path and never finish. The copies are counted by
    walking ``"l"`` only.
    """
    top: dict[str, Any] = {}
    for _ in range(64):
        top = {"l": top, "r": top}

    copy_top = _iterative_deepcopy(top)

    originals: list[dict[str, Any]] = []
    copies: list[dict[str, Any]] = []
    original, copied = top, copy_top
    for _ in range(64):
        originals.append(original)
        copies.append(copied)
        original, copied = original["l"], copied["l"]
    originals.append(original)
    copies.append(copied)
    # Scalars only in the assertions: repr of a diamond is exponential.
    levels_shared = [level["l"] is level["r"] for level in copies[:-1]]
    level_keys = [sorted(level) for level in copies[:-1]]
    copy_ids = {id(level) for level in copies}
    original_ids = {id(level) for level in originals}
    innermost_is_empty = copies[-1] == {}

    assert levels_shared == [True] * 64
    assert level_keys == [["l", "r"]] * 64
    assert len(copy_ids) == 65
    assert not copy_ids & original_ids
    assert innermost_is_empty


# --- before_tool: decisions -----------------------------------------------------


async def test_apply_before_tool_none_dispatches_the_original_args() -> None:
    """``None`` dispatches the loop's own args object; the hook's edits to its deep copy, nested ones included, have no effect (FR-003 D5, review round 1)."""
    args = {"q": "original", "limit": 10, "filters": {"tenant": "t-1", "tags": ["a"]}}
    expected = copy.deepcopy(args)

    def hook(_sid: object, _cid: object, _name: object, hook_args: dict[str, Any]) -> None:
        hook_args["q"] = "mutated"
        hook_args["added"] = True
        hook_args["filters"]["tenant"] = "t-2"
        hook_args["filters"]["tags"].append("b")

    outcome = await _before(hook, args)
    assert outcome.args is args
    assert args == expected
    assert outcome.denial is None


async def test_apply_before_tool_replace_dispatches_a_copy_of_the_replacement() -> None:
    """``ReplaceToolArgs`` dispatches a fresh copy of the replacement; the original args are untouched (FR-003 D5)."""
    args = {"q": "raw", "limit": 500}
    decision = ReplaceToolArgs(args={"q": "clamped", "limit": 50})

    outcome = await _before(_returning(decision), args)
    assert outcome.args == {"q": "clamped", "limit": 50}
    assert outcome.args is not decision.args
    assert outcome.denial is None
    assert args == {"q": "raw", "limit": 500}


async def test_apply_before_tool_deny_formats_the_reason() -> None:
    """``DenyToolCall`` becomes ``"Tool call denied: <reason>"`` and keeps the model's args (FR-003 D5)."""
    args = {"q": "other-tenant"}
    outcome = await _before(_returning(DenyToolCall(reason="not in this tenant")), args)
    assert outcome.denial == "Tool call denied: not in this tenant"
    assert outcome.args is args


# --- before_tool: failures under both fallbacks ------------------------------------


@pytest.mark.parametrize("fallback", _BOTH_FALLBACKS)
async def test_apply_before_tool_raise_follows_the_fallback(
    fallback: BeforeToolFallback,
) -> None:
    """A raising hook denies under ``DENY`` and dispatches the ORIGINAL, unedited args under ``ALLOW``; logged type-only (FR-003 D5).

    The hook edits its copy at the top level and inside a nested value before
    raising; neither edit reaches the args the loop keeps (review round 1).
    """
    args = {"q": "original", "filters": {"tenant": "t-1", "tags": ["a"]}}
    expected = copy.deepcopy(args)

    def hook(_sid: object, _cid: object, _name: object, hook_args: dict[str, Any]) -> None:
        hook_args["q"] = "tampered"
        hook_args["filters"]["tenant"] = "t-2"
        hook_args["filters"]["tags"].append("b")
        raise ValueError(f"guard broke on {_SECRET}")

    with structlog.testing.capture_logs() as logs:
        outcome = await _before(hook, args, fallback=fallback)

    assert outcome.args is args
    assert args == expected
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.hook_failed"
    assert entry["hook_name"] == "before_tool"
    assert entry["call_id"] == "call-1"
    assert entry["error_type"] == "ValueError"
    assert "error_message" not in entry
    assert set(entry) == _FAILED_KEYS
    assert _SECRET not in str(entry)
    if fallback is BeforeToolFallback.DENY:
        assert outcome.denial == _SDK_DENIAL
        assert entry["fallback"] == "call_denied"
    else:
        assert outcome.denial is None
        assert entry["fallback"] == "call_allowed"


@pytest.mark.parametrize("fallback", _BOTH_FALLBACKS)
@pytest.mark.parametrize(
    ("decision", "reason_code"),
    [
        pytest.param("deny", "unrecognised_decision", id="str"),
        pytest.param({"q": 1}, "unrecognised_decision", id="dict"),
        pytest.param(True, "unrecognised_decision", id="bool"),
        pytest.param(ToolResult(), "unrecognised_decision", id="tool_result"),
        pytest.param(
            ReplaceToolArgs.model_construct(args=["x"]), "invalid_args", id="replace_list"
        ),
        pytest.param(
            ReplaceToolArgs.model_construct(args={1: "x"}), "invalid_args", id="replace_int_key"
        ),
        pytest.param(ReplaceToolArgs.model_construct(), "invalid_args", id="replace_missing"),
    ],
)
async def test_apply_before_tool_invalid_decision_follows_the_fallback(
    fallback: BeforeToolFallback, decision: object, reason_code: str
) -> None:
    """An unusable decision denies under ``DENY``; under ``ALLOW`` it dispatches the original args, never the invalid replacement (FR-003 D5)."""
    args = {"q": "original"}
    with structlog.testing.capture_logs() as logs:
        outcome = await _before(_returning(decision), args, fallback=fallback)

    assert outcome.args is args
    assert args == {"q": "original"}
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.hook_invalid"
    assert entry["hook_name"] == "before_tool"
    assert entry["returned_type"] == type(decision).__name__
    assert entry["reason"] == reason_code
    assert set(entry) == _INVALID_KEYS
    if fallback is BeforeToolFallback.DENY:
        assert outcome.denial == _SDK_DENIAL
        assert entry["fallback"] == "call_denied"
    else:
        assert outcome.denial is None
        assert entry["fallback"] == "call_allowed"


@pytest.mark.parametrize("fallback", _BOTH_FALLBACKS)
@pytest.mark.parametrize(
    "decision",
    [
        pytest.param(DenyToolCall.model_construct(reason="  "), id="blank"),
        pytest.param(DenyToolCall.model_construct(reason=5), id="not_str"),
        pytest.param(DenyToolCall.model_construct(), id="missing"),
    ],
)
async def test_apply_before_tool_invalid_deny_reason_denies_under_both_fallbacks(
    fallback: BeforeToolFallback, decision: DenyToolCall
) -> None:
    """A returned deny is honoured under BOTH fallbacks; an unusable reason gets the SDK's (FR-003 D5)."""
    args = {"q": "original"}
    with structlog.testing.capture_logs() as logs:
        outcome = await _before(_returning(decision), args, fallback=fallback)

    assert outcome.denial == _SDK_DENIAL
    assert outcome.args is args
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.hook_invalid"
    assert entry["reason"] == "invalid_reason"
    assert entry["returned_type"] == "DenyToolCall"
    assert entry["fallback"] == "call_denied"


@pytest.mark.parametrize("fallback", _BOTH_FALLBACKS)
async def test_apply_before_tool_reraises_cancelled_error(fallback: BeforeToolFallback) -> None:
    """``CancelledError`` from ``before_tool`` propagates under both fallbacks, unlogged (FR-003 D4)."""

    async def hook(*_args: object) -> None:
        raise asyncio.CancelledError

    with structlog.testing.capture_logs() as logs:  # noqa: SIM117
        with pytest.raises(asyncio.CancelledError):
            await _before(hook, {"q": "x"}, fallback=fallback)
    assert _intervention_logs(logs) == []


# --- AC-5: callable shapes ------------------------------------------------------------


def _shapes(value: object) -> dict[str, Callable[..., Any]]:
    """Five callable shapes that all return (or resolve to) ``value``."""

    def plain(*_args: object) -> object:
        return value

    async def coroutine_function(*_args: object) -> object:
        return value

    def returns_coroutine(*_args: object) -> Awaitable[object]:
        return coroutine_function()

    class CallableObject:
        async def __call__(self, *_args: object) -> object:
            return value

    def keyword_bound(*_args: object, marker: str) -> object:
        assert marker == "bound"
        return value

    return {
        "def": plain,
        "async_def": coroutine_function,
        "sync_returning_coroutine": returns_coroutine,
        "callable_object": CallableObject(),
        "partial": functools.partial(keyword_bound, marker="bound"),
    }


@pytest.mark.parametrize(
    "shape",
    ["def", "async_def", "sync_returning_coroutine", "callable_object", "partial"],
)
async def test_intervention_hooks_accept_every_callable_shape(shape: str) -> None:
    """Both hooks accept every callable shape ``invoke_hook`` does (FR-003 AC-5)."""
    assert await _after(_shapes("the note")[shape]) == "the note"
    outcome = await _before(_shapes(DenyToolCall(reason="nope"))[shape], {"q": "x"})
    assert outcome.denial == "Tool call denied: nope"
    Interventions(before_tool=_shapes(None)[shape], after_tool=_shapes(None)[shape])


# --- before_tool_fallback ---------------------------------------------------------------


async def test_before_tool_fallback_defaults_to_deny() -> None:
    """Omitted, the fallback is ``DENY``: a raising guard means the tool is NOT dispatched (FR-003 D5)."""
    assert Interventions().before_tool_fallback is BeforeToolFallback.DENY

    def broken_guard(*_args: object) -> None:
        raise RuntimeError("guard backend down")

    tool = FakeTool("search")
    registry = Registry()
    registry.register(tool)
    llm = FakeLLMClient(
        [
            make_response(
                '{"thought": "t", "action": "tool", "tool_name": "search", '
                '"tool_args": {"q": "x"}, "answer": null}'
            ),
            make_response(
                '{"thought": "t", "action": "final", "tool_name": null, '
                '"tool_args": null, "answer": "done"}'
            ),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        prompts=PromptSections(persona="p"),
        safety=SafetyConfig(),
        model="m",
        tool_mode=ToolMode.JSON,
        interventions=Interventions(before_tool=broken_guard),
    )
    events = [e async for e in loop.run([ChatMessage(role="user", content="q")])]

    assert tool.call_count == 0
    failed = [e for e in events if isinstance(e, ToolFailedEvent)]
    assert [e.error for e in failed] == [_SDK_DENIAL]


@pytest.mark.parametrize("value", ["open", "DENY", "", None, True, 1])
def test_interventions_rejects_an_unknown_before_tool_fallback(value: object) -> None:
    """An unknown fallback raises ``ValueError`` at construction, in ``ToolMode``'s shape (FR-003 D5)."""
    with pytest.raises(ValueError, match="before_tool_fallback") as info:
        Interventions(before_tool_fallback=value)  # type: ignore[arg-type]
    assert "'deny', 'allow'" in str(info.value)


def test_interventions_normalises_before_tool_fallback_strings() -> None:
    """The plain strings become the enum members, also through ``dataclasses.replace`` (FR-003 D5)."""
    assert Interventions(before_tool_fallback="deny").before_tool_fallback is (  # type: ignore[arg-type]
        BeforeToolFallback.DENY
    )
    allow = Interventions(before_tool_fallback="allow")  # type: ignore[arg-type]
    assert allow.before_tool_fallback is BeforeToolFallback.ALLOW
    replaced = dataclasses.replace(Interventions(), before_tool_fallback="allow")  # type: ignore[arg-type]
    assert replaced.before_tool_fallback is BeforeToolFallback.ALLOW
    assert Interventions(before_tool_fallback=BeforeToolFallback.ALLOW).before_tool_fallback is (
        BeforeToolFallback.ALLOW
    )
