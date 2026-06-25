"""FastMCP test-server factory for the in-memory compatibility oracle.

The official-client compatibility tests (``test_client_compat.py``) drive a
real :class:`mcp.server.fastmcp.FastMCP` server through
:func:`mcp.shared.memory.create_connected_server_and_client_session`. This
module builds that server with deliberately-sanitised tools.

Sanitiser discipline (learning #760 pt3)
    The intentionally-failing tool raises
    :class:`mcp.server.fastmcp.exceptions.ToolError` with a *safe* message —
    never a raw exception carrying sensitive text. FastMCP serialises a
    ``ToolError`` into ``isError=True`` content as ``"Error executing tool
    <name>: <safe message>"``, so the test server never leaks secrets into
    the error-content path our
    :class:`fifty_agent_sdk.mcp.client._MCPCallError` captures (BR-005: a
    per-call ``isError`` is surfaced as a recoverable ``ToolResult(is_error=
    True)``, not raised as an ``MCPError``).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel


class LookupResult(BaseModel):
    """Typed return so FastMCP populates ``structuredContent``."""

    key: str
    found: bool


def build_test_server() -> FastMCP:
    """Build a FastMCP server with sanitised tools for the compat oracle.

    Tools:
        - ``search``: a search-like tool with NO return annotation, so FastMCP
          generates no output schema and its ``structuredContent`` stays
          ``None`` — exercising the content-blocks success branch of
          :meth:`fifty_agent_sdk.mcp.client.MCPClient.invoke`. (A typed return such
          as ``-> dict[str, object]`` WOULD populate ``structuredContent``.)
        - ``lookup``: returns a Pydantic-typed :class:`LookupResult` so FastMCP
          populates ``structuredContent`` (exercising the structured success
          branch — a plain ``dict`` return does NOT populate it).
        - ``boom``: raises :class:`ToolError` with a safe message
          (exercising the ``isError`` → ``_MCPCallError`` path with sanitised
          ``content``, surfaced to the loop as a recoverable
          ``ToolResult(is_error=True)``).
    """
    server: FastMCP = FastMCP("fifty-agent-sdk-compat-test")

    @server.tool()
    def search(q: str):  # no return annotation -> no output schema
        """Search the corpus and return raw hits (no schema → content blocks)."""
        return {"hits": [q, q.upper()]}

    @server.tool()
    def lookup(key: str) -> LookupResult:
        """Look a key up; the typed return populates ``structuredContent``."""
        return LookupResult(key=key, found=True)

    @server.tool()
    def boom(x: str) -> str:
        """Always fail with a SAFE message (no secrets in the error content)."""
        raise ToolError("safe message")

    return server
