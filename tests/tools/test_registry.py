"""Tests for fifty_agent_sdk.tools.registry — dispatch, timeout, and exception classification."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import structlog

from fifty_agent_sdk.errors import LLMError, ToolNotFound, ToolTimeout
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema
from fifty_agent_sdk.tools.registry import Registry

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _EchoTool:
    """A trivial Tool that echoes its input args back as the output."""

    name = "echo"
    description = "Echoes its arguments."
    schema = ToolSchema()

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output=args, is_error=False, error=None)


class _NamedTool:
    """A configurable Tool whose name is set per instance."""

    description = "Configurable."
    schema = ToolSchema()

    def __init__(self, name: str, output: Any = None) -> None:
        self.name = name
        self._output = output

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output=self._output, is_error=False, error=None)


class _SlowTool:
    """A Tool that sleeps; tracks whether cleanup ran."""

    name = "slow"
    description = "Sleeps for 1 second."
    schema = ToolSchema()

    def __init__(self) -> None:
        self.cleanup_ran = False

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        try:
            await asyncio.sleep(1.0)
            return ToolResult(output="done")
        finally:
            self.cleanup_ran = True


class _RaisingTool:
    """A Tool whose body raises a configurable exception."""

    name = "raises"
    description = "Always raises."
    schema = ToolSchema()

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        raise self._exc


# ---------------------------------------------------------------------------
# register / list / get
# ---------------------------------------------------------------------------


def test_register_then_list_returns_tool() -> None:
    r = Registry()
    t = _EchoTool()
    r.register(t)
    assert r.list() == [t]


def test_register_rejects_non_tool() -> None:
    r = Registry()
    with pytest.raises(TypeError, match="requires a Tool"):
        r.register(object())  # type: ignore[arg-type]


def test_register_overwrites_on_duplicate_name() -> None:
    r = Registry()
    first = _NamedTool("same", output="first")
    second = _NamedTool("same", output="second")
    r.register(first)
    r.register(second)
    assert r.list() == [second]
    assert r.get("same") is second


def test_register_emits_warning_on_duplicate_name() -> None:
    """An overwrite is allowed (tested above) but must emit an operational signal."""
    r = Registry()
    first = _NamedTool("same", output="first")
    second = _NamedTool("same", output="second")
    r.register(first)
    with structlog.testing.capture_logs() as logs:
        r.register(second)
    overwrites = [
        entry
        for entry in logs
        if entry.get("event") == "tool overwritten" and entry.get("name") == "same"
    ]
    assert len(overwrites) == 1, f"expected exactly one overwrite warning, got {logs}"
    assert overwrites[0]["log_level"] == "warning"


def test_register_does_not_warn_on_first_registration() -> None:
    """A first-time registration must not emit the overwrite warning."""
    r = Registry()
    with structlog.testing.capture_logs() as logs:
        r.register(_NamedTool("fresh"))
    overwrites = [entry for entry in logs if entry.get("event") == "tool overwritten"]
    assert overwrites == []


def test_list_preserves_insertion_order() -> None:
    r = Registry()
    a = _NamedTool("a")
    b = _NamedTool("b")
    c = _NamedTool("c")
    r.register(a)
    r.register(b)
    r.register(c)
    assert [t.name for t in r.list()] == ["a", "b", "c"]


def test_list_is_a_snapshot() -> None:
    r = Registry()
    r.register(_EchoTool())
    snapshot = r.list()
    snapshot.clear()
    # Mutating the snapshot must not affect the registry.
    assert len(r.list()) == 1


def test_get_returns_registered_tool() -> None:
    r = Registry()
    t = _EchoTool()
    r.register(t)
    assert r.get("echo") is t


def test_get_raises_tool_not_found_with_context() -> None:
    r = Registry()
    r.register(_NamedTool("alpha"))
    r.register(_NamedTool("beta"))
    with pytest.raises(ToolNotFound) as exc_info:
        r.get("missing")
    assert exc_info.value.context["name"] == "missing"
    assert set(exc_info.value.context["available"]) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# invoke — happy path
# ---------------------------------------------------------------------------


async def test_invoke_happy_path() -> None:
    r = Registry()
    r.register(_EchoTool())
    result = await r.invoke("echo", {"x": 1}, timeout=1.0)
    assert result.is_error is False
    assert result.output == {"x": 1}


async def test_invoke_with_none_timeout_runs_to_completion() -> None:
    r = Registry()
    r.register(_EchoTool())
    result = await r.invoke("echo", {"y": 2}, timeout=None)
    assert result.output == {"y": 2}


# ---------------------------------------------------------------------------
# invoke — timeout
# ---------------------------------------------------------------------------


async def test_invoke_timeout_raises_tool_timeout() -> None:
    r = Registry()
    slow = _SlowTool()
    r.register(slow)
    with pytest.raises(ToolTimeout) as exc_info:
        await r.invoke("slow", {}, timeout=0.01)
    assert exc_info.value.context["name"] == "slow"
    assert exc_info.value.context["timeout"] == 0.01


async def test_invoke_timeout_cancels_tool_coroutine() -> None:
    """The tool's `finally` block must run, proving cancellation propagated."""
    r = Registry()
    slow = _SlowTool()
    r.register(slow)
    with pytest.raises(ToolTimeout):
        await r.invoke("slow", {}, timeout=0.01)
    assert slow.cleanup_ran is True


# ---------------------------------------------------------------------------
# invoke — unknown tool
# ---------------------------------------------------------------------------


async def test_invoke_unknown_tool_raises_tool_not_found() -> None:
    r = Registry()
    with pytest.raises(ToolNotFound) as exc_info:
        await r.invoke("ghost", {}, timeout=1.0)
    assert exc_info.value.context["name"] == "ghost"


# ---------------------------------------------------------------------------
# invoke — tool exceptions
# ---------------------------------------------------------------------------


async def test_invoke_wraps_value_error_into_tool_result() -> None:
    r = Registry()
    r.register(_RaisingTool(ValueError("bad input")))
    result = await r.invoke("raises", {}, timeout=1.0)
    assert result.is_error is True
    assert result.error == "ValueError: bad input"
    assert result.output is None


async def test_invoke_wraps_runtime_error_into_tool_result() -> None:
    r = Registry()
    r.register(_RaisingTool(RuntimeError("oops")))
    result = await r.invoke("raises", {}, timeout=1.0)
    assert result.is_error is True
    assert result.error == "RuntimeError: oops"


async def test_invoke_propagates_agent_sdk_error() -> None:
    r = Registry()
    r.register(_RaisingTool(LLMError("provider blew up")))
    with pytest.raises(LLMError):
        await r.invoke("raises", {}, timeout=1.0)


async def test_invoke_propagates_cancelled_error() -> None:
    r = Registry()
    r.register(_RaisingTool(asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await r.invoke("raises", {}, timeout=1.0)


async def test_invoke_propagates_keyboard_interrupt() -> None:
    r = Registry()
    r.register(_RaisingTool(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        await r.invoke("raises", {}, timeout=1.0)
