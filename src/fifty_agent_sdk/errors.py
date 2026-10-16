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
    :class:`fifty_agent_sdk.llm.protocol.LLMClient` MUST wrap any provider-specific
    exception into this type so callers never see provider SDK exceptions
    leak out.
    """


class MCPError(AgentSdkError):
    """MCP transport, protocol, or session failure (a genuine system error).

    Raised by :class:`fifty_agent_sdk.mcp.client.MCPClient`, which wraps the
    official ``mcp`` SDK's :class:`mcp.ClientSession`, for:

    - Transport failures (connection refused, timeout, non-2xx HTTP) — the
      underlying ``httpx`` exception is unwrapped from the ``mcp`` transport's
      anyio ``ExceptionGroup`` and translated here.
    - Protocol/handshake failures and server-returned JSON-RPC errors
      surfaced by the SDK as :class:`mcp.shared.exceptions.McpError`
      (``error.code``/``error.data``).
    - Session/handshake :class:`RuntimeError`s (e.g. an unsupported protocol
      version) and use after :meth:`fifty_agent_sdk.mcp.client.MCPClient.aclose`.

    The wrapping guarantees the uniform-MCPError contract (TD-007) for these
    fatal paths: neither ``mcp``'s ``McpError`` nor any ``httpx`` exception
    leaks from :meth:`fifty_agent_sdk.mcp.client.MCPClient.discover`/``invoke``.

    A per-call ``tools/call`` result carrying ``isError=True`` does NOT raise
    this error (BR-005). It is a *recoverable* per-tool failure — the server
    ran and reported failure, while the transport/protocol/session are healthy —
    so :meth:`MCPClient.invoke` returns a per-call signal that the
    :class:`fifty_agent_sdk.tools.mcp_provider._MCPToolAdapter` surfaces as a
    :class:`fifty_agent_sdk.tools.protocol.ToolResult` with ``is_error=True``,
    which the agent loop feeds back to the model as a tool observation rather
    than terminating the run.

    ``context`` typically carries ``server_url``, ``method``, ``tool_name``,
    ``error_code``, ``error_data``, ``status_code``, ``operation``, or
    ``wrapped`` (the underlying httpx exception class name). It NEVER includes
    auth headers — auth lives on the httpx client and is stripped before
    recording (see :class:`fifty_agent_sdk.mcp.client.MCPClient`).

    Because :class:`MCPError` is an :class:`AgentSdkError`, when one DOES occur
    (transport/protocol/session) the tool
    :class:`fifty_agent_sdk.tools.registry.Registry` re-raises it untouched rather
    than wrapping it in a :class:`fifty_agent_sdk.tools.protocol.ToolResult` with
    ``is_error=True``. This treats a genuine connection/protocol failure as a
    system error the surrounding runner can catch and surface, NOT as a
    recoverable per-tool failure the LLM should reason about.
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


class SkillError(AgentSdkError):
    """A skill directory or its ``SKILL.md`` file could not be loaded (FR-004).

    Raised by :meth:`fifty_agent_sdk.skills.Skill.from_directory` and
    :func:`fifty_agent_sdk.skills.discover_skills` for problems with skill
    FILES. The message names the file, for example ``Invalid skill file
    /srv/skills/pdf/SKILL.md: frontmatter line 3: ...``. Configuration
    mistakes made in code are builtins instead: a duplicate skill name or a
    tool-name collision is a :class:`ValueError`, a wrong argument type a
    :class:`TypeError`, and a :class:`~fifty_agent_sdk.skills.Skill` built
    directly with a bad field raises pydantic's ``ValidationError``.

    ``context`` always holds ``path`` (the file or directory, as a ``str``)
    and ``reason``, a fixed code a host can branch on without parsing the
    message: ``not_found``, ``not_a_directory``, ``missing_skill_md``,
    ``not_a_file``, ``too_large``, ``not_utf8``, ``unreadable`` (an
    ``OSError`` while resolving, checking, listing or reading; also a
    ``RuntimeError`` from resolving a symlink loop on CPython before 3.13,
    and a ``ValueError`` for a path holding NUL), ``no_frontmatter``,
    ``unclosed_frontmatter``, ``malformed_frontmatter``, ``duplicate_key``,
    ``missing_field``, ``invalid_field``, ``name_mismatch`` or
    ``empty_body``. It may also hold ``line`` (the 1-based line of
    ``SKILL.md``), ``field`` (``"name"``, ``"description"`` or ``"body"``),
    ``offset`` (the byte offset of the first byte that is not UTF-8, counted
    after a stripped BOM, for ``not_utf8``), ``error_type`` (the exception
    class name, for ``unreadable``) and, for ``name_mismatch``,
    ``directory_name``. The path in the message has each surrogate code
    point and C0/C1 control character written as a six-character escape
    (``escape_for_log``), so printing it cannot raise or write a terminal
    control sequence; ``context["path"]`` keeps the path as it is.
    """


__all__ = [
    "AgentSdkError",
    "LLMError",
    "MCPError",
    "MaxIterationsExceeded",
    "ParserError",
    "SkillError",
    "StateStoreError",
    "ToolNotFound",
    "ToolTimeout",
]
