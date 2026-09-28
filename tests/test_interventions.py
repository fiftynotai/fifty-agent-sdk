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
* argument copies (review round 1): each hook edits its own deep copy, nested
  values included; args ``copy.deepcopy`` rejects reach the hook as a
  one-level copy with one ``intervention.args_not_copyable`` WARNING, and the
  hook still runs.

The loop-level behaviour (where the helpers are called, and what reaches the
wire and the event stream) is pinned in ``tests/loop/test_loop_interventions.py``.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import inspect
from collections.abc import Awaitable, Callable
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
    ("value", "error_type"),
    [
        pytest.param(_NotDeepCopyable(), "TypeError", id="custom_object"),
        pytest.param(_deeply_nested(5000), "RecursionError", id="deep_nesting"),
    ],
)
@pytest.mark.parametrize("hook_name", ["before_tool", "after_tool"])
async def test_hooks_get_a_shallow_copy_when_args_cannot_be_deep_copied(
    hook_name: str, value: object, error_type: str
) -> None:
    """Args ``copy.deepcopy`` rejects reach the hook as a one-level copy with one WARNING; the hook still runs and nothing is treated as a hook failure (FR-003 review round 1).

    ``deep_nesting`` stands for JSON the model nested deeply enough to exhaust
    the recursion limit. The one-level copy still keeps a top-level edit from
    leaking; a nested value is shared, which is the documented degradation.
    """
    args = {"q": "x", "handle": value}
    seen: list[dict[str, Any]] = []

    def hook(*hook_args: Any) -> str | None:
        own = hook_args[3]
        seen.append(own)
        own["q"] = "changed"
        return "note" if hook_name == "after_tool" else None

    with structlog.testing.capture_logs() as logs:
        if hook_name == "after_tool":
            assert await _after(hook, args=args) == "note"
        else:
            outcome = await _before(hook, args)
            assert outcome.args is args
            assert outcome.denial is None

    assert len(seen) == 1
    assert seen[0] is not args
    assert seen[0]["handle"] is value
    assert args["q"] == "x"
    warnings = _intervention_logs(logs)
    assert len(warnings) == 1
    entry = warnings[0]
    assert entry["event"] == "intervention.args_not_copyable"
    assert entry["hook_name"] == hook_name
    assert entry["call_id"] == "call-1"
    assert entry["error_type"] == error_type
    assert set(entry) == {"event", "log_level", "hook_name", "call_id", "error_type"}


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
