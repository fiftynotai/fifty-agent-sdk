"""Protocol-only MCP (Model Context Protocol) client surface.

This subpackage speaks JSON-RPC 2.0 over HTTP against any MCP server using
the already-vendored :mod:`httpx` dependency. There is NO dependency on the
official ``mcp`` SDK or ``fastmcp`` packages — the wire envelopes are
constructed by hand. Consumers wanting protocol-only opt-out can simply not
import :mod:`agent_sdk.mcp`; the surface is unconditional because ``httpx``
is already a hard dependency of :mod:`agent_sdk`.

Module boundaries
    :mod:`agent_sdk.mcp` MUST NOT import from :mod:`agent_sdk.tools`. The
    bridge from the protocol layer into the tool layer lives in
    :mod:`agent_sdk.tools.mcp_provider`, which imports from BOTH this package
    and the tools package.

Targeted MCP spec
    JSON-RPC 2.0 envelope per the MCP specification. The client implements
    the ``tools/list`` and ``tools/call`` methods. Push refresh
    (``notifications/tools/list_changed``) is deferred to a future revision;
    the SDK is poll-only in v1.
"""

from __future__ import annotations

from agent_sdk.mcp.client import MCPClient, MCPClientConfig, MCPToolDef

__all__ = [
    "MCPClient",
    "MCPClientConfig",
    "MCPToolDef",
]
