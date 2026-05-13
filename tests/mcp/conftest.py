"""Strict-mock MCP server fixture for the protocol-only client tests.

The fixture implements the L-356 mock-looseness mitigation: every observed
HTTP request is validated as a JSON-RPC 2.0 envelope (``jsonrpc == "2.0"``,
str ``id``, str ``method``, optional ``params`` object), POSTed to the
configured server URL with ``Content-Type: application/json``. Unknown
JSON-RPC methods are rejected with ``error.code == -32601`` (Method not
found) — they MUST NOT silently return 200/empty so tests cannot
accidentally certify a client that drifted from the spec.

Test-side wiring
    Each test gets a fresh :class:`MockMCPServer`. ``register_tool(name,
    handler_fn)`` registers a per-tool ``tools/call`` handler;
    ``set_tool_catalog(defs)`` configures the ``tools/list`` payload.

Captured envelopes
    Every successfully-routed request envelope is appended to
    :attr:`MockMCPServer.observed_envelopes` so individual tests can
    assert on exact method names, params shapes, and id values.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

MCP_URL = "https://mcp.test.local/mcp"
"""Single canonical URL the strict mock serves. Any other path is rejected."""

ToolHandler = Callable[[dict[str, Any]], Any]
"""``params.arguments`` -> the ``content`` of a ``tools/call`` result.

Handlers MAY also return a full dict including ``isError`` to exercise the
error-path branch of :meth:`agent_sdk.mcp.client.MCPClient.invoke`.
"""


class MockMCPServer:
    """In-process MCP server registered with :class:`httpx.MockTransport`.

    Wires a single endpoint at :data:`MCP_URL` that dispatches JSON-RPC
    requests by ``method`` field. Asserts envelope shape on every call.
    """

    def __init__(self) -> None:
        self._catalog: list[dict[str, Any]] = []
        self._tool_handlers: dict[str, ToolHandler] = {}
        self.observed_envelopes: list[dict[str, Any]] = []
        self.observed_headers: list[httpx.Headers] = []
        # Optional override: when set, the next request returns this raw
        # body verbatim regardless of method (used by malformed-envelope
        # tests like envelope-version mismatch or id mismatch).
        self.raw_body_override: bytes | None = None
        self.raw_status_override: int = 200

    # ------------------------------------------------------------------
    # Test-facing helpers
    # ------------------------------------------------------------------

    def set_tool_catalog(self, defs: list[dict[str, Any]]) -> None:
        """Configure the response payload for ``tools/list``."""
        self._catalog = list(defs)

    def register_tool(self, name: str, handler: ToolHandler) -> None:
        """Register a per-tool ``tools/call`` handler."""
        self._tool_handlers[name] = handler

    def force_next_raw(self, body: bytes, *, status: int = 200) -> None:
        """Force the next request to return the given raw body.

        Used to exercise malformed-envelope handling — the body bypasses
        the dispatcher entirely.
        """
        self.raw_body_override = body
        self.raw_status_override = status

    # ------------------------------------------------------------------
    # Transport handler
    # ------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Dispatch one request, asserting envelope shape on the way in."""
        # Path strictness — reject anything that didn't hit the canonical URL.
        if str(request.url) != MCP_URL:
            raise AssertionError(
                f"MockMCPServer received unexpected URL: {request.url!r} "
                f"(expected {MCP_URL!r})"
            )
        if request.method != "POST":
            raise AssertionError(
                f"MockMCPServer received non-POST method: {request.method!r}"
            )
        content_type = request.headers.get("content-type")
        if content_type is None or "application/json" not in content_type:
            raise AssertionError(
                f"MockMCPServer expected Content-Type: application/json; "
                f"got {content_type!r}"
            )

        body = request.read()
        # Raw body override path (malformed-envelope tests).
        if self.raw_body_override is not None:
            override = self.raw_body_override
            status = self.raw_status_override
            self.raw_body_override = None
            self.raw_status_override = 200
            return httpx.Response(status, content=override)

        try:
            envelope = json.loads(body)
        except ValueError as exc:
            raise AssertionError(
                f"MockMCPServer expected valid JSON body; got {body!r}"
            ) from exc

        # JSON-RPC 2.0 envelope shape — the heart of the L-356 mitigation.
        if not isinstance(envelope, dict):
            raise AssertionError(
                f"JSON-RPC envelope must be a JSON object; got {type(envelope).__name__}"
            )
        if envelope.get("jsonrpc") != "2.0":
            raise AssertionError(
                f"JSON-RPC envelope missing jsonrpc=2.0; got "
                f"{envelope.get('jsonrpc')!r}"
            )
        req_id = envelope.get("id")
        if not isinstance(req_id, str) or not req_id:
            raise AssertionError(
                f"JSON-RPC envelope missing string id; got {req_id!r}"
            )
        method = envelope.get("method")
        if not isinstance(method, str) or not method:
            raise AssertionError(
                f"JSON-RPC envelope missing string method; got {method!r}"
            )
        if "params" in envelope and not isinstance(envelope["params"], dict):
            raise AssertionError(
                f"JSON-RPC envelope params must be an object; got "
                f"{type(envelope['params']).__name__}"
            )

        self.observed_envelopes.append(envelope)
        self.observed_headers.append(httpx.Headers(request.headers))

        # Dispatch by method.
        params = envelope.get("params") or {}
        if method == "tools/list":
            return self._respond_result(req_id, {"tools": self._catalog})
        if method == "tools/call":
            return self._handle_tools_call(req_id, params)
        # Unknown method — REJECT with -32601 per JSON-RPC 2.0 spec.
        return self._respond_error(
            req_id,
            code=-32601,
            message=f"Method not found: {method}",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _handle_tools_call(
        self, req_id: str, params: dict[str, Any]
    ) -> httpx.Response:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return self._respond_error(
                req_id, code=-32602, message="tools/call: name required"
            )
        if "arguments" not in params or not isinstance(params["arguments"], dict):
            return self._respond_error(
                req_id,
                code=-32602,
                message="tools/call: arguments object required",
            )
        handler = self._tool_handlers.get(name)
        if handler is None:
            return self._respond_error(
                req_id,
                code=-32602,
                message=f"tools/call: unknown tool {name!r}",
            )
        payload = handler(params["arguments"])
        # If the handler returns a dict that already looks like a
        # ``tools/call`` result envelope (carries ``isError`` or
        # ``content``), pass it through; otherwise wrap it as content.
        if isinstance(payload, dict) and ("isError" in payload or "content" in payload):
            return self._respond_result(req_id, payload)
        return self._respond_result(req_id, {"content": payload})

    @staticmethod
    def _respond_result(req_id: str, result: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": req_id, "result": result},
        )

    @staticmethod
    def _respond_error(
        req_id: str,
        *,
        code: int,
        message: str,
        data: Any = None,
    ) -> httpx.Response:
        error_obj: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": req_id, "error": error_obj},
        )


@pytest.fixture
def mcp_server() -> MockMCPServer:
    """One :class:`MockMCPServer` per test, freshly constructed."""
    return MockMCPServer()


@pytest.fixture
def mcp_transport(mcp_server: MockMCPServer) -> httpx.MockTransport:
    """An :class:`httpx.MockTransport` wired to :attr:`mcp_server.handle`."""
    return httpx.MockTransport(mcp_server.handle)


@pytest.fixture
def mcp_http_client(
    mcp_transport: httpx.MockTransport,
) -> httpx.AsyncClient:
    """An :class:`httpx.AsyncClient` backed by the strict mock transport.

    Suitable for direct injection into :class:`agent_sdk.mcp.MCPClient`'s
    ``client`` constructor argument.
    """
    return httpx.AsyncClient(transport=mcp_transport)
