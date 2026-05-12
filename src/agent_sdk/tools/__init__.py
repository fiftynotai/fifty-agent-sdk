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
"""

from agent_sdk.tools.inproc_provider import InProcProvider, tool
from agent_sdk.tools.protocol import Tool, ToolCall, ToolResult, ToolSchema
from agent_sdk.tools.registry import Registry

__all__ = [
    "InProcProvider",
    "Registry",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolSchema",
    "tool",
]
