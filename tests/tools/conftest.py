"""Conftest for tool-layer tests.

Re-exports the controllable in-memory MCP server fixture
(:func:`controllable_server`) from :mod:`tests.mcp.conftest` so the
MCPProvider regression tests in this directory share the same harness that
drives the real :class:`agent_sdk.mcp.client.MCPClient` mapping/unwrap code.
"""

from __future__ import annotations

from tests.mcp.conftest import controllable_server  # noqa: F401 — fixture re-export.
