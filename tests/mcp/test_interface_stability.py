"""Public-surface stability guard for the BR-042 ``mcp``-SDK swap.

BR-042 swapped the hand-rolled JSON-RPC internals for the official ``mcp``
SDK while holding the public surface byte-for-byte stable (the load-bearing
contract: vendored consumers — notably ``mbrgea-ai`` — need only a re-vendor +
dependency bump, no code change). This test pins that surface with
:func:`inspect.signature` / :func:`getattr` so a rename or signature drift
fails here loudly rather than at a downstream consumer.
"""

from __future__ import annotations

import inspect

import agent_sdk
from agent_sdk.mcp import MCPClient, MCPClientConfig, MCPToolDef
from agent_sdk.tools.mcp_provider import MCPProvider, RefreshSummary


def test_top_level_exports_present() -> None:
    """``agent_sdk.__all__`` still exports the MCP public surface."""
    for name in ("MCPClient", "MCPClientConfig", "MCPToolDef", "MCPProvider", "RefreshSummary"):
        assert name in agent_sdk.__all__, f"{name} missing from agent_sdk.__all__"
        assert getattr(agent_sdk, name) is not None


def test_mcpclient_signatures_stable() -> None:
    """``MCPClient`` keeps its method signatures."""
    init_sig = inspect.signature(MCPClient.__init__)
    params = init_sig.parameters
    assert list(params) == ["self", "config", "auth", "client"]
    assert params["auth"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["client"].kind is inspect.Parameter.KEYWORD_ONLY

    assert inspect.iscoroutinefunction(MCPClient.discover)
    assert inspect.iscoroutinefunction(MCPClient.invoke)
    assert inspect.iscoroutinefunction(MCPClient.aclose)

    invoke_sig = inspect.signature(MCPClient.invoke)
    assert list(invoke_sig.parameters) == ["self", "name", "args"]

    discover_sig = inspect.signature(MCPClient.discover)
    assert list(discover_sig.parameters) == ["self"]


def test_mcpprovider_signatures_stable() -> None:
    """``MCPProvider`` keeps its method signatures."""
    init_sig = inspect.signature(MCPProvider.__init__)
    assert list(init_sig.parameters) == ["self", "client"]

    for method in ("attach", "refresh", "start_periodic_refresh", "aclose"):
        assert hasattr(MCPProvider, method), f"MCPProvider lost {method}"
        assert inspect.iscoroutinefunction(getattr(MCPProvider, method))

    attach_sig = inspect.signature(MCPProvider.attach)
    assert list(attach_sig.parameters) == ["self", "registry"]
    periodic_sig = inspect.signature(MCPProvider.start_periodic_refresh)
    assert list(periodic_sig.parameters) == ["self", "interval_seconds"]


def test_model_field_names_stable() -> None:
    """``MCPToolDef`` / ``MCPClientConfig`` field names are unchanged."""
    assert set(MCPToolDef.model_fields) == {"name", "description", "input_schema"}
    assert set(MCPClientConfig.model_fields) == {
        "base_url",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "connect_retries",
        "user_agent",
    }
    assert set(RefreshSummary.model_fields) == {"added", "refreshed"}
