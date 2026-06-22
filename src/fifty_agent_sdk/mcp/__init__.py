"""MCP (Model Context Protocol) client surface.

This subpackage wraps the official ``mcp`` Python SDK
(``mcp>=1.27.0,<2.0.0``): :class:`fifty_agent_sdk.mcp.client.MCPClient` drives a
:class:`mcp.ClientSession` over Streamable HTTP. The protocol wire, envelope
correlation, session-id handling, and handshake are owned by ``mcp``; this
package contributes the fifty-agent-sdk-facing contract (typed
:class:`MCPToolDef`s, the uniform :class:`fifty_agent_sdk.errors.MCPError`
translation, auth-header redaction, and the no-secrets-in-logs discipline).

Module boundaries
    :mod:`fifty_agent_sdk.mcp` MUST NOT import from :mod:`fifty_agent_sdk.tools`. The
    bridge from the protocol layer into the tool layer lives in
    :mod:`fifty_agent_sdk.tools.mcp_provider`, which imports from BOTH this package
    and the tools package.

Transport seam
    The Streamable HTTP transport lives in :mod:`fifty_agent_sdk.mcp.transport`
    behind a small :class:`fifty_agent_sdk.mcp.transport.Transport` protocol so a
    future stdio transport slots in without touching :class:`MCPClient` or
    its consumers. Transport selection is internal — the public surface is
    intentionally not widened to expose it.

Supported MCP surface
    The ``tools/list`` and ``tools/call`` methods. Push refresh
    (``notifications/tools/list_changed``) is deferred; the client is
    poll-only (see :class:`fifty_agent_sdk.tools.mcp_provider.MCPProvider` for the
    periodic-refresh loop).
"""

from __future__ import annotations

from fifty_agent_sdk.mcp.client import MCPClient, MCPClientConfig, MCPToolDef

__all__ = [
    "MCPClient",
    "MCPClientConfig",
    "MCPToolDef",
]
