"""Typed exception hierarchy for the agent SDK.

All errors raised by SDK internals inherit from :class:`AgentSdkError`, which
allows callers to write a single ``except AgentSdkError`` to catch every
expected failure mode. Subclasses act as discriminators that carry no new
behavior — they only narrow what went wrong so callers can decide whether to
retry, surface the error to the user, or fail loudly.

Every error instance carries a structured ``context: dict[str, Any]`` payload
intended for debugging and logging. The context is always a real dict (never
``None``) so call sites can safely do ``error.context["foo"] = bar`` without
guarding for absence.
"""

from __future__ import annotations

from typing import Any


class AgentSdkError(Exception):
    """Base class for every exception raised by the agent SDK.

    Catching :class:`AgentSdkError` catches every SDK-originated failure.

    Attributes:
        message: The human-readable error message also stored in ``args[0]``.
        context: A dictionary of structured debugging information. Always a
            real ``dict`` (never ``None``); defaults to ``{}`` when not given.

    Caution:
        Callers MUST NOT place credentials (API keys, tokens, passwords) into
        ``context``: the dict is included verbatim in ``repr(error)`` and in
        any log line that captures the exception.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context if context is not None else {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class LLMError(AgentSdkError):
    """LLM call failed.

    Raised when an LLM provider call cannot be completed for any reason:
    network errors, HTTP non-2xx responses, malformed provider envelopes,
    timeouts, or missing required response fields. Implementations of
    :class:`agent_sdk.llm.protocol.LLMClient` MUST wrap any provider-specific
    exception into this type so callers never see provider SDK exceptions
    leak out.
    """


class MCPError(AgentSdkError):
    """MCP transport, protocol, or tool-call failure.

    Raised by :class:`agent_sdk.mcp.client.MCPClient` for:

    - Transport failures (connection refused, timeout, non-2xx HTTP)
    - Malformed JSON-RPC envelopes (missing ``jsonrpc``/``id``, bad ``result``)
    - Server-returned JSON-RPC ``error`` payloads (``error.code``,
      ``error.message``)
    - Argument validation errors surfaced by the remote tool

    ``context`` typically carries ``server_url``, ``method``, ``tool_name``,
    ``error_code``, ``status_code``, or ``wrapped`` (the underlying httpx
    exception class name). It NEVER includes auth headers — they are stripped
    before recording (see :class:`agent_sdk.mcp.client.MCPClient`).

    Because :class:`MCPError` is an :class:`AgentSdkError`, the tool
    :class:`agent_sdk.tools.registry.Registry` re-raises it untouched rather
    than wrapping it in a :class:`agent_sdk.tools.protocol.ToolResult` with
    ``is_error=True``. This treats MCP failures as system errors the
    surrounding runner can catch and surface, NOT as recoverable per-tool
    failures the LLM should reason about.
    """


class ToolNotFound(AgentSdkError):
    """A requested tool name was not present in the registry."""


class ToolTimeout(AgentSdkError):
    """A tool invocation exceeded its allotted timeout budget."""


class MaxIterationsExceeded(AgentSdkError):
    """The ReACT loop reached its iteration cap without producing a final answer."""


class ParserError(AgentSdkError):
    """The parser could not extract a structured action from model output."""


class StateStoreError(AgentSdkError):
    """A state-store backend operation (load, save, delete) failed."""


__all__ = [
    "AgentSdkError",
    "LLMError",
    "MCPError",
    "MaxIterationsExceeded",
    "ParserError",
    "StateStoreError",
    "ToolNotFound",
    "ToolTimeout",
]
