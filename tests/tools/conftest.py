"""Conftest for tool-layer tests.

Imports the strict-mock MCP fixtures (:func:`mcp_server`,
:func:`mcp_transport`, :func:`mcp_http_client`) from
:mod:`tests.mcp.conftest` so the MCPProvider tests in this directory can
share the same mock without duplicating the dispatcher logic.
"""

from __future__ import annotations

from tests.mcp.conftest import (  # noqa: F401 — fixture re-exports.
    mcp_http_client,
    mcp_server,
    mcp_transport,
)
