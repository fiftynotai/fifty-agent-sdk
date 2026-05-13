"""Tool layer subpackage.

Public surface:

- :class:`Tool` — the pluggable tool Protocol.
- :class:`ToolSchema` — JSON-schema-shaped input description.
- :class:`ToolResult` — invocation outcome (success or recoverable failure).
- :class:`ToolCall` — model-issued (name, args) envelope, re-exported from
  :mod:`agent_sdk.llm.types` for ergonomic local imports.
- :class:`Registry` — name-keyed dispatch with timeout enforcement.
- :func:`tool` — decorator that lifts an ``async def`` into a :class:`Tool`.
- :class:`InProcProvider` — bulk-register helper for decorated callables.
- :class:`MCPProvider` — bridges an :class:`agent_sdk.mcp.client.MCPClient`
  into the registry, registering one adapter per MCP-advertised tool.
"""

from agent_sdk.tools.inproc_provider import InProcProvider, tool
from agent_sdk.tools.mcp_provider import MCPProvider, RefreshSummary
from agent_sdk.tools.protocol import Tool, ToolCall, ToolResult, ToolSchema
from agent_sdk.tools.registry import Registry

__all__ = [
    "InProcProvider",
    "MCPProvider",
    "RefreshSummary",
    "Registry",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "tool",
]
