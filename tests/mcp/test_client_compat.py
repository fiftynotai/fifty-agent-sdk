"""Official-client in-memory compatibility oracle for :class:`MCPClient`.

These tests drive our ``MCPClient`` mapping/unwrap code through a REAL
official :class:`mcp.ClientSession` connected to a :class:`FastMCP` server via
:func:`mcp.shared.memory.create_connected_server_and_client_session` (learning
#760 pt6). The official client/server pair is the compatibility oracle — our
wrapper is NOT its own oracle. The full ``initialize → list_tools →
call_tool`` path is exercised: the harness runs ``initialize``, and these
tests assert ``list_tools`` round-trips the FastMCP-declared catalog and
``call_tool`` produces the chosen success-shape (and the sanitised per-call
``isError`` → :class:`fifty_agent_sdk.mcp.client._MCPCallError`, BR-005).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from fifty_agent_sdk.mcp import MCPToolDef
from fifty_agent_sdk.mcp.client import _MCPCallError

from .conftest import make_compat_client


async def test_discover_round_trips_fastmcp_catalog(fastmcp_server: FastMCP) -> None:
    """``list_tools`` maps the FastMCP-declared tools into typed defs."""
    async with make_compat_client(fastmcp_server) as client:
        defs = await client.discover()

    assert all(isinstance(d, MCPToolDef) for d in defs)
    by_name = {d.name: d for d in defs}
    assert {"search", "lookup", "boom"} <= set(by_name)
    # inputSchema is the camelCase JSON-Schema FastMCP generates from the
    # tool signature: search(q: str) -> a required string param.
    search_schema = by_name["search"].input_schema
    assert search_schema["type"] == "object"
    assert "q" in search_schema["properties"]
    assert search_schema["required"] == ["q"]


async def test_invoke_returns_content_blocks_for_plain_dict(fastmcp_server: FastMCP) -> None:
    """A plain-dict tool yields ``structuredContent=None`` → content blocks list.

    This pins the chosen success-shape's content-blocks branch: the return is
    a list of serialised content-block dicts (no single-block flattening).
    """
    async with make_compat_client(fastmcp_server) as client:
        result = await client.invoke("search", {"q": "kittens"})

    assert isinstance(result, list)
    assert len(result) == 1
    block = result[0]
    assert block["type"] == "text"
    # FastMCP serialises the dict return into the TextContent text body.
    assert "kittens" in block["text"]
    assert "KITTENS" in block["text"]


async def test_invoke_returns_structured_content_when_present(fastmcp_server: FastMCP) -> None:
    """When the server populates ``structuredContent`` it is returned verbatim.

    ``lookup`` returns a Pydantic-typed object, so FastMCP populates
    ``structuredContent`` (a plain ``dict`` return would NOT). This pins the
    structured branch of the success-shape.
    """
    async with make_compat_client(fastmcp_server) as client:
        result = await client.invoke("lookup", {"key": "alpha"})

    assert isinstance(result, dict)
    assert result["key"] == "alpha"
    assert result["found"] is True


async def test_invoke_maps_is_error_to_call_error_with_sanitised_content(
    fastmcp_server: FastMCP,
) -> None:
    """``isError=True`` RETURNS a :class:`_MCPCallError` with sanitised content.

    BR-005: a per-call ``isError`` no longer raises — ``invoke`` returns a
    recoverable :class:`_MCPCallError`. The ``boom`` tool raises
    ``ToolError("safe message")`` (learning #760 pt3), so the FastMCP-serialised
    error content carries only the safe message — asserting it appears in
    ``.content`` proves both the ``isError`` path and that no raw exception text
    leaked. The bounded ``.message`` names the tool (the only field the adapter
    surfaces to the model).
    """
    async with make_compat_client(fastmcp_server) as client:
        result = await client.invoke("boom", {"x": "y"})

    assert isinstance(result, _MCPCallError)
    assert "boom" in result.message
    content = result.content
    assert isinstance(content, list)
    joined = " ".join(block.get("text", "") for block in content)
    assert "safe message" in joined
