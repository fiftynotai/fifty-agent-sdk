"""Tests for :class:`agent_sdk.tools.mcp_provider.MCPProvider`.

These tests pair each scenario from the BR-008 plan §9 with the strict-mock
MCP server defined in :mod:`tests.mcp.conftest`. The provider tests live in
``tests/tools/`` rather than ``tests/mcp/`` so they sit alongside the rest
of the tool-layer suite — they exercise the bridge between protocol and
registry, not the protocol itself.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import httpx
import pytest
import structlog

from agent_sdk.errors import MCPError
from agent_sdk.mcp import MCPClient, MCPClientConfig
from agent_sdk.tools.mcp_provider import (
    MCPProvider,
    RefreshSummary,
    _to_tool_schema,
)
from agent_sdk.tools.protocol import ToolResult, ToolSchema
from agent_sdk.tools.registry import Registry
from tests.mcp.conftest import MCP_URL, MockMCPServer


def _config() -> MCPClientConfig:
    return MCPClientConfig(
        base_url=MCP_URL,
        connect_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
    )


def _make_client(transport_handler: Any) -> MCPClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    return MCPClient(_config(), client=http_client)


def _tool_def(name: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Tool {name}",
        "inputSchema": schema
        if schema is not None
        else {"type": "object", "properties": {}, "required": []},
    }


# ---------------------------------------------------------------------------
# P1 — attach() registers one adapter per tool
# ---------------------------------------------------------------------------


async def test_attach_registers_one_adapter_per_tool(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([_tool_def("alpha"), _tool_def("beta"), _tool_def("gamma")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)

    names = sorted(t.name for t in registry.list())
    assert names == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# P2 / P3 — schema translation
# ---------------------------------------------------------------------------


def test_schema_translation_identity() -> None:
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
    )
    assert isinstance(schema, ToolSchema)
    assert schema.type == "object"
    assert schema.properties == {"q": {"type": "string"}}
    assert schema.required == ["q"]
    assert schema.additionalProperties is False


def test_schema_translation_drops_unknown_top_level_keys() -> None:
    schema = _to_tool_schema(
        {
            "type": "object",
            "properties": {},
            "required": [],
            "$defs": {"X": {"type": "object"}},
            "examples": [{}],
        }
    )
    # No KeyError; ToolSchema is extra="forbid" so we silently drop the
    # unknown top-level keys at translation time.
    assert schema.type == "object"


def test_schema_translation_falls_back_for_non_object_top_level() -> None:
    schema = _to_tool_schema({"type": "string"})
    assert schema.type == "object"
    assert schema.properties == {}
    assert schema.required == []


# ---------------------------------------------------------------------------
# P4 / P5 / P6 — adapter invoke + error propagation
# ---------------------------------------------------------------------------


async def test_adapter_invoke_returns_tool_result_on_success(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([_tool_def("answer")])
    mcp_server.register_tool("answer", lambda _args: {"answer": 42})
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    adapter = registry.get("answer")
    result = await adapter.invoke({})
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.output == {"answer": 42}


async def test_adapter_propagates_mcp_error_unwrapped(
    mcp_server: MockMCPServer,
) -> None:
    # No handler registered -> dispatcher returns -32602.
    mcp_server.set_tool_catalog([_tool_def("missing")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    adapter = registry.get("missing")
    with pytest.raises(MCPError) as exc:
        await adapter.invoke({})
    assert exc.value.context["tool_name"] == "missing"


async def test_registry_invoke_propagates_mcp_error(
    mcp_server: MockMCPServer,
) -> None:
    """Registry.invoke() re-raises AgentSdkError subclasses untouched."""
    mcp_server.set_tool_catalog([_tool_def("missing")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    with pytest.raises(MCPError):
        await registry.invoke("missing", {}, timeout=1.0)


# ---------------------------------------------------------------------------
# P7 / P8 — refresh semantics
# ---------------------------------------------------------------------------


async def test_refresh_adds_new_tools_without_removing_existing(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([_tool_def("A"), _tool_def("B")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    assert sorted(t.name for t in registry.list()) == ["A", "B"]

    # Catalog mutates: A disappears, C arrives.
    mcp_server.set_tool_catalog([_tool_def("B"), _tool_def("C")])
    summary = await provider.refresh()

    names = sorted(t.name for t in registry.list())
    assert names == ["A", "B", "C"]
    assert isinstance(summary, RefreshSummary)
    assert summary.added == 1  # C is new
    assert summary.refreshed == 1  # B already present


async def test_refresh_updates_existing_tool_definition(
    mcp_server: MockMCPServer,
) -> None:
    """Last-write-wins: schema changes propagate to the registry."""
    mcp_server.set_tool_catalog(
        [
            _tool_def(
                "A",
                schema={
                    "type": "object",
                    "properties": {"v1": {"type": "string"}},
                    "required": ["v1"],
                },
            )
        ]
    )
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    assert "v1" in registry.get("A").schema.properties

    mcp_server.set_tool_catalog(
        [
            _tool_def(
                "A",
                schema={
                    "type": "object",
                    "properties": {"v2": {"type": "integer"}},
                    "required": ["v2"],
                },
            )
        ]
    )
    await provider.refresh()
    refreshed = registry.get("A")
    assert "v2" in refreshed.schema.properties
    assert "v1" not in refreshed.schema.properties


async def test_refresh_without_attach_raises_runtime_error() -> None:
    client = _make_client(lambda req: httpx.Response(200, json={}))
    provider = MCPProvider(client)
    with pytest.raises(RuntimeError, match="prior attach"):
        await provider.refresh()


# ---------------------------------------------------------------------------
# P9 / P10 — periodic refresh
# ---------------------------------------------------------------------------


async def test_periodic_refresh_runs_and_can_be_cancelled(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([_tool_def("A")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    initial_calls = len([e for e in mcp_server.observed_envelopes if e["method"] == "tools/list"])
    assert initial_calls == 1

    await provider.start_periodic_refresh(interval_seconds=0.02)
    await asyncio.sleep(0.07)
    await provider.aclose()

    discovers = [e for e in mcp_server.observed_envelopes if e["method"] == "tools/list"]
    # At least one periodic refresh fired (in addition to the attach one).
    assert len(discovers) >= 2


async def test_periodic_refresh_double_start_raises(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.start_periodic_refresh(interval_seconds=10.0)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await provider.start_periodic_refresh(interval_seconds=10.0)
    finally:
        await provider.aclose()


async def test_periodic_refresh_non_positive_interval_raises(
    mcp_server: MockMCPServer,
) -> None:
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    mcp_server.set_tool_catalog([])
    await provider.attach(registry)
    with pytest.raises(ValueError, match="must be > 0"):
        await provider.start_periodic_refresh(interval_seconds=0)


async def test_periodic_refresh_requires_prior_attach() -> None:
    client = _make_client(lambda req: httpx.Response(200, json={}))
    provider = MCPProvider(client)
    with pytest.raises(RuntimeError, match="prior attach"):
        await provider.start_periodic_refresh(interval_seconds=1.0)


async def test_periodic_refresh_survives_transient_failure(
    mcp_server: MockMCPServer,
) -> None:
    """A discover() failure inside the periodic loop must NOT terminate it."""
    mcp_server.set_tool_catalog([_tool_def("A")])
    state = {"fail_next": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail_next"]:
            state["fail_next"] = False
            return httpx.Response(503)
        return mcp_server.handle(request)

    client = _make_client(handler)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.start_periodic_refresh(interval_seconds=0.02)
    state["fail_next"] = True
    await asyncio.sleep(0.07)  # spans the failure + at least one success
    await provider.aclose()
    # Provider must still be functional after the failure.
    await provider.refresh()
    assert "A" in [t.name for t in registry.list()]


# ---------------------------------------------------------------------------
# P11 / P12 — async contract + lifecycle
# ---------------------------------------------------------------------------


def test_attach_is_async() -> None:
    assert inspect.iscoroutinefunction(MCPProvider.attach)


def test_refresh_is_async() -> None:
    assert inspect.iscoroutinefunction(MCPProvider.refresh)


async def test_aclose_does_not_close_underlying_client(
    mcp_server: MockMCPServer,
) -> None:
    mcp_server.set_tool_catalog([])
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(mcp_server.handle))
    client = MCPClient(_config(), client=http_client)
    provider = MCPProvider(client)
    registry = Registry()
    await provider.attach(registry)
    await provider.aclose()
    # The MCPClient (and the underlying http_client) remain usable.
    await client.discover()


async def test_aclose_is_idempotent_when_no_task_running(
    mcp_server: MockMCPServer,
) -> None:
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    await provider.aclose()  # no-op
    await provider.aclose()  # still safe


# ---------------------------------------------------------------------------
# P13 — name-collision warning
# ---------------------------------------------------------------------------


async def test_attach_warns_on_name_collision(
    mcp_server: MockMCPServer,
) -> None:
    """Pre-register a tool with the same name; provider must log a warning."""

    class _LocalTool:
        name = "search"
        description = "local"
        schema = ToolSchema()

        async def invoke(self, args: dict[str, Any]) -> ToolResult:
            return ToolResult(output="local")

    registry = Registry()
    registry.register(_LocalTool())  # type: ignore[arg-type]

    mcp_server.set_tool_catalog([_tool_def("search")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    with structlog.testing.capture_logs() as logs:
        await provider.attach(registry)

    overwrites = [
        entry
        for entry in logs
        if entry.get("event") == "mcp.tool_overwrite" and entry.get("name") == "search"
    ]
    assert len(overwrites) == 1, f"expected exactly one MCPProvider overwrite warning, got {logs}"
    assert overwrites[0]["log_level"] == "warning"


async def test_refresh_does_not_warn_for_first_time_tools(
    mcp_server: MockMCPServer,
) -> None:
    """The collision warning fires ONLY for names already in the registry."""
    mcp_server.set_tool_catalog([_tool_def("new")])
    client = _make_client(mcp_server.handle)
    provider = MCPProvider(client)
    registry = Registry()
    with structlog.testing.capture_logs() as logs:
        await provider.attach(registry)
    overwrites = [e for e in logs if e.get("event") == "mcp.tool_overwrite"]
    assert overwrites == []
