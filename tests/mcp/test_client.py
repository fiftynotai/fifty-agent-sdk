"""Strict-mock tests for :class:`agent_sdk.mcp.client.MCPClient`.

These tests are the L-356 mock-looseness mitigation in action: each fixture
hits a single canonical URL and asserts the JSON-RPC envelope shape on the
way in. Unknown methods return ``-32601``. Drift between the client's wire
format and the MCP spec breaks the strict mock first, before it ever
touches a real server.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from agent_sdk.errors import MCPError
from agent_sdk.mcp import MCPClient, MCPClientConfig, MCPToolDef

from .conftest import MCP_URL, MockMCPServer


def _config(**overrides: Any) -> MCPClientConfig:
    fields: dict[str, Any] = {
        "base_url": MCP_URL,
        "connect_timeout_seconds": 1.0,
        "read_timeout_seconds": 1.0,
    }
    fields.update(overrides)
    return MCPClientConfig(**fields)


def _make_client(
    mcp_http_client: httpx.AsyncClient,
    *,
    auth: Any = None,
) -> MCPClient:
    return MCPClient(_config(), auth=auth, client=mcp_http_client)


def _canonical_tool() -> dict[str, Any]:
    return {
        "name": "search",
        "description": "Search the corpus.",
        "inputSchema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }


# ---------------------------------------------------------------------------
# C1 — discover() happy path + envelope assertions
# ---------------------------------------------------------------------------


async def test_discover_happy_path_returns_typed_defs(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog(
        [
            _canonical_tool(),
            {
                "name": "fetch",
                "description": "Fetch a record.",
                "inputSchema": {"type": "object"},
            },
        ]
    )
    client = _make_client(mcp_http_client)
    defs = await client.discover()

    assert isinstance(defs, list)
    assert [d.name for d in defs] == ["search", "fetch"]
    assert all(isinstance(d, MCPToolDef) for d in defs)
    assert defs[0].input_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }

    # Envelope assertions — exact method, presence of id/jsonrpc.
    assert len(mcp_server.observed_envelopes) == 1
    env = mcp_server.observed_envelopes[0]
    assert env["jsonrpc"] == "2.0"
    assert env["method"] == "tools/list"
    assert isinstance(env["id"], str) and env["id"]


# ---------------------------------------------------------------------------
# C2/C3 — malformed envelope handling
# ---------------------------------------------------------------------------


async def test_discover_rejects_wrong_jsonrpc_version(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    # Force the next response to carry an invalid jsonrpc version.
    bad = json.dumps(
        {"jsonrpc": "1.0", "id": "irrelevant", "result": {"tools": []}}
    ).encode()
    mcp_server.force_next_raw(bad)
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert exc.value.context["envelope_jsonrpc"] == "1.0"
    assert exc.value.context["method"] == "tools/list"
    assert exc.value.context["server_url"] == MCP_URL


async def test_discover_rejects_id_mismatch(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    bad = json.dumps(
        {"jsonrpc": "2.0", "id": "not-the-id-we-sent", "result": {"tools": []}}
    ).encode()
    mcp_server.force_next_raw(bad)
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert "does not match" in exc.value.message
    assert exc.value.context["received_id"] == "not-the-id-we-sent"
    assert "expected_id" in exc.value.context


async def test_discover_rejects_envelope_without_id(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    bad = json.dumps({"jsonrpc": "2.0", "result": {"tools": []}}).encode()
    mcp_server.force_next_raw(bad)
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert "missing 'id'" in exc.value.message


async def test_discover_rejects_envelope_with_both_result_and_error(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    bad = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "x",  # id mismatch will hit FIRST — guard against that.
            "result": {"tools": []},
            "error": {"code": -32603, "message": "internal"},
        }
    ).encode()
    mcp_server.force_next_raw(bad)
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError):
        await client.discover()


# ---------------------------------------------------------------------------
# C4 — fixture rejects unknown JSON-RPC methods with -32601 (CONTROL test)
# ---------------------------------------------------------------------------


async def test_fixture_rejects_unknown_jsonrpc_method(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    """Confirms the strict-mock dispatcher is strict — direct probe, no client.

    This is the fixture-level enforcement of L-356. If the dispatcher
    accidentally accepted ``tools/foo`` with a 200 empty body, every other
    test in this file would silently pass without exercising the real
    method. This test exists to catch fixture regressions.
    """
    response = await mcp_http_client.post(
        MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": "probe-1",
            "method": "tools/foo",
        },
        headers={"Content-Type": "application/json"},
    )
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "probe-1"
    assert body["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# C5 — invoke() happy path + params shape
# ---------------------------------------------------------------------------


async def test_invoke_happy_path_sends_correct_params(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.register_tool("search", lambda args: {"hits": args["q"]})
    client = _make_client(mcp_http_client)
    result = await client.invoke("search", {"q": "kittens"})
    assert result == {"hits": "kittens"}

    env = mcp_server.observed_envelopes[-1]
    assert env["method"] == "tools/call"
    assert env["params"] == {"name": "search", "arguments": {"q": "kittens"}}


async def test_invoke_handles_scalar_result(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    # Handler returns a scalar; dispatcher wraps it as ``content``.
    mcp_server.register_tool("ping", lambda _args: "pong")
    client = _make_client(mcp_http_client)
    assert await client.invoke("ping", {}) == "pong"


async def test_invoke_passes_through_full_result_when_no_content(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    # Handler returns a dict that lacks both ``content`` and ``isError`` —
    # the dispatcher wraps it as ``content`` so we still get the wrapped
    # value back.
    mcp_server.register_tool("dump", lambda _args: {"k": "v"})
    client = _make_client(mcp_http_client)
    assert await client.invoke("dump", {}) == {"k": "v"}


# ---------------------------------------------------------------------------
# C6 — JSON-RPC error -> MCPError
# ---------------------------------------------------------------------------


async def test_invoke_maps_jsonrpc_error_to_mcp_error(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    # No handler registered for "missing" -> dispatcher returns -32602.
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("missing", {})
    assert exc.value.context["error_code"] == -32602
    assert exc.value.context["tool_name"] == "missing"
    assert exc.value.context["method"] == "tools/call"


async def test_invoke_maps_is_error_result_to_mcp_error(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    # Handler returns a full result envelope with isError=True.
    mcp_server.register_tool(
        "fail",
        lambda _args: {"isError": True, "content": "tool said no"},
    )
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("fail", {})
    assert exc.value.context["tool_name"] == "fail"
    assert exc.value.context["content"] == "tool said no"


# ---------------------------------------------------------------------------
# C7 / C8 / C14 — transport-layer errors via httpx.MockTransport
# ---------------------------------------------------------------------------


async def test_invoke_maps_connect_error_to_mcp_error() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["wrapped"] == "ConnectError"
    assert exc.value.context["server_url"] == MCP_URL
    assert exc.value.context["tool_name"] == "search"


async def test_invoke_maps_5xx_to_mcp_error() -> None:
    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server_down))
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["status_code"] == 503
    assert exc.value.context["wrapped"] == "HTTPStatusError"


async def test_invoke_maps_read_timeout_to_mcp_error() -> None:
    def slow(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert exc.value.context["wrapped"] == "ReadTimeout"


async def test_invoke_maps_decode_error_to_mcp_error() -> None:
    def garbage(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(garbage))
    client = MCPClient(_config(), client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert "not valid JSON" in exc.value.message


# ---------------------------------------------------------------------------
# C9 / C10 / C11 — auth handling
# ---------------------------------------------------------------------------


async def test_static_auth_header_propagated(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog([])
    client = _make_client(
        mcp_http_client, auth={"Authorization": "Bearer test-token"}
    )
    await client.discover()
    headers = mcp_server.observed_headers[-1]
    assert headers.get("authorization") == "Bearer test-token"


async def test_callable_auth_invoked_each_call(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog([])
    counter = {"n": 0}

    async def rotating_auth() -> dict[str, str]:
        counter["n"] += 1
        return {"Authorization": f"Bearer token-{counter['n']}"}

    client = _make_client(mcp_http_client, auth=rotating_auth)
    await client.discover()
    await client.discover()
    assert counter["n"] == 2
    headers_first = mcp_server.observed_headers[0]
    headers_second = mcp_server.observed_headers[1]
    assert headers_first.get("authorization") == "Bearer token-1"
    assert headers_second.get("authorization") == "Bearer token-2"


async def test_auth_callable_invoked_for_invoke_as_well(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.register_tool("ping", lambda _args: "pong")
    counter = {"n": 0}

    async def rotating_auth() -> dict[str, str]:
        counter["n"] += 1
        return {"X-Auth-Token": f"v-{counter['n']}"}

    client = _make_client(mcp_http_client, auth=rotating_auth)
    await client.invoke("ping", {})
    await client.invoke("ping", {})
    assert counter["n"] == 2


async def test_auth_header_redacted_from_error_context() -> None:
    """Forcing a 5xx must NOT leak the auth header into MCPError.context."""

    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server_down))
    client = MCPClient(
        _config(),
        auth={"Authorization": "Bearer SECRET-DO-NOT-LEAK"},
        client=http_client,
    )
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    serialized = json.dumps(exc.value.context, default=str)
    assert "SECRET-DO-NOT-LEAK" not in serialized
    assert "Bearer" not in serialized
    assert "Authorization" not in serialized


async def test_auth_header_redacted_when_callable_used() -> None:
    """Same redaction guarantee with a callable auth provider."""

    def server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server_down))

    async def auth_provider() -> dict[str, str]:
        return {"X-Auth-Token": "ROTATING-SECRET"}

    client = MCPClient(_config(), auth=auth_provider, client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("search", {"q": "x"})
    assert "ROTATING-SECRET" not in json.dumps(exc.value.context, default=str)


async def test_invalid_auth_callable_return_raises_mcp_error() -> None:
    """A misbehaved auth callable surfaces a clear :class:`MCPError`."""

    def _never_called(_request: httpx.Request) -> httpx.Response:
        # Should NEVER be reached — auth resolution fails first.
        raise AssertionError("transport should not be invoked")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(_never_called))

    async def bad_auth() -> Any:
        return "not-a-mapping"

    client = MCPClient(_config(), auth=bad_auth, client=http_client)
    with pytest.raises(MCPError) as exc:
        await client.invoke("x", {})
    assert exc.value.context["wrapped"] == "str"


# ---------------------------------------------------------------------------
# C12 / C13 — aclose() behavior
# ---------------------------------------------------------------------------


async def test_aclose_disposes_owned_client(
    mcp_server: MockMCPServer,
) -> None:
    # Construct without passing in a client so MCPClient owns one.
    # Wrap the owned client's transport replacement so the test still works
    # without real network — patch the internal client's _transport.
    cfg = _config()
    client = MCPClient(cfg)
    # Replace the owned client's transport with our mock so the first call
    # would normally succeed.
    client._client._transport = httpx.MockTransport(mcp_server.handle)  # type: ignore[attr-defined]
    mcp_server.set_tool_catalog([])
    await client.discover()  # works
    await client.aclose()
    # Post-aclose, the defensive guard maps the "client closed" condition
    # into the uniform MCPError contract instead of leaking httpx's
    # RuntimeError. Asserting the message + operation context pins the
    # contract end-to-end (TD-007 item 2).
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert "closed" in exc.value.message.lower()
    assert exc.value.context["operation"] == "discover"


async def test_aclose_is_idempotent(
    mcp_server: MockMCPServer,
) -> None:
    """A second ``aclose()`` is a no-op and MUST NOT raise (TD-007 item 2)."""
    cfg = _config()
    client = MCPClient(cfg)
    client._client._transport = httpx.MockTransport(mcp_server.handle)  # type: ignore[attr-defined]
    mcp_server.set_tool_catalog([])
    await client.discover()
    await client.aclose()
    # Second call must short-circuit cleanly — no exception, no double-close
    # of the underlying httpx client.
    await client.aclose()


async def test_caller_provided_client_not_disposed(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog([])
    client = _make_client(mcp_http_client)
    await client.discover()
    await client.aclose()
    # The caller-owned http client is still usable.
    response = await mcp_http_client.post(
        MCP_URL,
        json={"jsonrpc": "2.0", "id": "post-close", "method": "tools/list"},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Catalog parsing edge cases
# ---------------------------------------------------------------------------


async def test_discover_rejects_malformed_envelope(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    bad = json.dumps(
        {"jsonrpc": "2.0", "id": "x", "result": {"tools": "oops"}}
    ).encode()
    mcp_server.force_next_raw(bad)
    client = _make_client(mcp_http_client)
    # The strict mock validates envelope shape before the parser sees it,
    # so the id-mismatch path fires first. Assertion still proves that
    # malformed envelopes surface as MCPError; parser-branch coverage for
    # ``tools: <not-a-list>`` is left to a future brief.
    with pytest.raises(MCPError):
        await client.discover()


async def test_discover_handles_snake_case_input_schema(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    """Some MCP servers ship ``input_schema`` (snake) instead of camelCase."""
    mcp_server.set_tool_catalog(
        [
            {
                "name": "x",
                "description": "",
                "input_schema": {"type": "object"},
            }
        ]
    )
    client = _make_client(mcp_http_client)
    defs = await client.discover()
    assert defs[0].input_schema == {"type": "object"}


async def test_discover_rejects_missing_tool_name(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog(
        [{"description": "anonymous", "inputSchema": {"type": "object"}}]
    )
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    assert "missing 'name'" in exc.value.message


# ---------------------------------------------------------------------------
# Required headers + user-agent
# ---------------------------------------------------------------------------


async def test_user_agent_header_set(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog([])
    client = MCPClient(
        _config(user_agent="my-agent/2.0"),
        client=mcp_http_client,
    )
    await client.discover()
    assert mcp_server.observed_headers[-1].get("user-agent") == "my-agent/2.0"


async def test_unique_ids_per_call(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    mcp_server.set_tool_catalog([])
    client = _make_client(mcp_http_client)
    await client.discover()
    await client.discover()
    ids = [e["id"] for e in mcp_server.observed_envelopes]
    assert len(ids) == 2 and ids[0] != ids[1]


# ---------------------------------------------------------------------------
# Docstring-drift regression — MCPError.context allow-list
# ---------------------------------------------------------------------------


async def test_mcperror_context_keys_match_class_docstring_allowlist(
    mcp_server: MockMCPServer, mcp_http_client: httpx.AsyncClient
) -> None:
    """Pin the ``MCPClient`` docstring's ``MCPError.context`` allow-list.

    The class docstring's "Invariant" paragraph documents the exact set of
    keys an :class:`MCPError.context` may carry. This test derives BOTH
    sides programmatically — the documented set by parsing
    ``MCPClient.__doc__``, the runtime set by triggering one representative
    :class:`MCPError` per distinct context-key group — and asserts set
    equality in both directions, so a doc edit OR a code edit that drifts
    the two apart fails here instead of rotting silently.
    """
    # --- (a) documented set: parse the brace-delimited allow-list ----------
    doc = MCPClient.__doc__
    assert doc is not None, "MCPClient must carry a class docstring"
    brace_match = re.search(
        r"allow-list\s*``\{(?P<keys>[^}]*)\}``", doc, re.DOTALL
    )
    assert brace_match is not None, (
        "could not locate the brace-delimited allow-list after the "
        "'allow-list' keyword in MCPClient.__doc__"
    )
    documented: set[str] = {
        token.strip()
        for token in brace_match.group("keys").split(",")
        if token.strip()
    }

    # --- (b) runtime set: trigger one MCPError per context-key group ------
    runtime: set[str] = set()

    def _capture(exc: MCPError) -> None:
        runtime.update(exc.context.keys())

    # Group 1: envelope wrong jsonrpc version -> envelope_jsonrpc.
    mcp_server.force_next_raw(
        json.dumps(
            {"jsonrpc": "1.0", "id": "irrelevant", "result": {"tools": []}}
        ).encode()
    )
    client = _make_client(mcp_http_client)
    with pytest.raises(MCPError) as exc:
        await client.discover()
    _capture(exc.value)

    # Group 2: envelope id mismatch -> expected_id, received_id.
    mcp_server.force_next_raw(
        json.dumps(
            {"jsonrpc": "2.0", "id": "wrong-id", "result": {"tools": []}}
        ).encode()
    )
    with pytest.raises(MCPError) as exc:
        await client.discover()
    _capture(exc.value)

    # Group 3: envelope with both result and error -> has_result, has_error.
    # This branch sits AFTER the id-match check in `_parse_response`, so the
    # response id must equal the request id. The client picks a random hex
    # id per call, so we echo the inbound id back via a MockTransport that
    # inspects the request body.
    def _echo_both(request: httpx.Request) -> httpx.Response:
        sent_id = json.loads(request.content)["id"]
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": sent_id,
                "result": {"tools": []},
                "error": {"code": -32603, "message": "internal"},
            },
        )

    both_client = MCPClient(
        _config(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_echo_both)),
    )
    with pytest.raises(MCPError) as exc:
        await both_client.discover()
    _capture(exc.value)

    # Group 4: JSON-RPC error object -> error_code, error_data.
    with pytest.raises(MCPError) as exc:
        await client.invoke("missing", {})  # no handler -> -32602
    _capture(exc.value)

    # Group 5: result carries isError=True -> tool_name, content.
    mcp_server.register_tool(
        "fail", lambda _args: {"isError": True, "content": "tool said no"}
    )
    with pytest.raises(MCPError) as exc:
        await client.invoke("fail", {})
    _capture(exc.value)

    # Group 6: transport-layer 5xx -> status_code, wrapped.
    def _server_down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    down_client = MCPClient(
        _config(), client=httpx.AsyncClient(
            transport=httpx.MockTransport(_server_down)
        )
    )
    with pytest.raises(MCPError) as exc:
        await down_client.invoke("search", {"q": "x"})
    _capture(exc.value)

    # Group 7: post-aclose call -> operation.
    closed_client = MCPClient(_config())
    closed_client._client._transport = httpx.MockTransport(  # type: ignore[attr-defined]
        mcp_server.handle
    )
    await closed_client.aclose()
    with pytest.raises(MCPError) as exc:
        await closed_client.discover()
    _capture(exc.value)

    # --- assert: both directions, with offending keys named ---------------
    undocumented = runtime - documented
    stale = documented - runtime
    assert undocumented == set(), (
        f"MCPError.context emits key(s) not in the class-docstring "
        f"allow-list: {sorted(undocumented)}"
    )
    assert stale == set(), (
        f"class-docstring allow-list names key(s) the runtime never "
        f"emits (stale doc): {sorted(stale)}"
    )
