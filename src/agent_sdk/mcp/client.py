"""Protocol-only MCP client over JSON-RPC 2.0 / HTTP.

:class:`MCPClient` is a hand-rolled JSON-RPC 2.0 client implementing the
``tools/list`` and ``tools/call`` methods from the Model Context Protocol
specification. It depends only on :mod:`httpx` (already a hard dep of
:mod:`agent_sdk`) — no ``mcp`` SDK, no ``fastmcp``.

Authentication
    The constructor accepts an ``auth`` argument that is either a static
    ``Mapping[str, str]`` (e.g. ``{"Authorization": "Bearer ..."}``) or an
    async callable returning the same shape on each invocation. Callables
    enable token rotation without re-creating the client. Auth header values
    are NEVER logged and NEVER appear in :class:`agent_sdk.errors.MCPError`
    context — header names registered through ``auth`` are tracked and
    redacted via :func:`_redact_headers` before any header dict is captured
    into an error.

Retry policy
    The client uses ``httpx``'s transport-level retry (connection retries
    only) via ``httpx.AsyncHTTPTransport(retries=cfg.connect_retries)``.
    Application-level retry on 5xx is out of scope; those responses become
    :class:`MCPError` and the consumer decides how to react.

Timeouts
    ``connect_timeout_seconds`` and ``read_timeout_seconds`` map to
    ``httpx.Timeout``'s ``connect``/``read``/``write``/``pool`` slots. A
    timeout surfaces as :class:`MCPError` with
    ``context["wrapped"] == "ReadTimeout"`` (or similar).

Error contract
    Every public method either returns successfully or raises
    :class:`MCPError`. No ``httpx`` exceptions leak; no envelope parsing
    error leaks. The :class:`agent_sdk.tools.mcp_provider._MCPToolAdapter`
    deliberately does NOT catch :class:`MCPError` so the
    :class:`agent_sdk.tools.registry.Registry`'s ``AgentSdkError`` branch
    re-raises it untouched.

Concurrency / id correlation
    v1 issues one JSON-RPC request per ``httpx`` call and reads the response
    synchronously; ids are scoped to that single round-trip. No pipelining,
    no batching. A future concurrent-invoke layer would need an id→Future
    map; the parse helper carries a TODO marker for that work.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from agent_sdk.errors import MCPError

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

DEBUG-level log lines include ``method``, request ``id``, and a duration in
milliseconds. Headers are NEVER logged — the redaction contract is enforced
by funnelling all header dicts through :func:`_redact_headers` before any
log or :class:`MCPError` context capture.
"""

_REDACTION_SENTINEL: Final[str] = "<redacted>"
"""Replacement value substituted for any redacted header value."""

# Header names that the SDK treats as sensitive regardless of whether the
# caller declared them via ``auth``. Kept lowercase for case-insensitive
# comparison.
_BUILTIN_SENSITIVE_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-auth-token",
        "cookie",
        "set-cookie",
    }
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class MCPToolDef(BaseModel):
    """A single MCP-advertised tool definition.

    Returned in batches by :meth:`MCPClient.discover`. ``input_schema`` is
    the raw JSON-Schema dict shipped by the MCP server; the
    :class:`agent_sdk.tools.mcp_provider.MCPProvider` translates it into a
    :class:`agent_sdk.tools.protocol.ToolSchema` at adapter-construction
    time.

    Attributes:
        name: Tool name. Matches the string the LLM emits and that the
            tool :class:`agent_sdk.tools.registry.Registry` uses as its
            dict key.
        description: Human-readable explanation rendered into the system
            prompt by the BR-003 prompt builder.
        input_schema: Raw JSON-Schema object (top-level ``"type": "object"``
            per the MCP spec). May be empty for parameter-less tools.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPClientConfig(BaseModel):
    """Connection configuration for :class:`MCPClient`.

    Attributes:
        base_url: Full URL of the MCP server's JSON-RPC HTTP endpoint
            (e.g. ``"https://mcp.example.com/mcp"``). Used verbatim; the
            client does NOT append paths.
        connect_timeout_seconds: TCP connect timeout passed to
            :class:`httpx.Timeout`.
        read_timeout_seconds: Per-call read/write/pool timeout passed to
            :class:`httpx.Timeout`. The ``Registry``'s ``tool_timeout``
            wraps the whole adapter call (loop safety budget); this
            httpx-level timeout is the inner budget and SHOULD be smaller
            than the outer one.
        connect_retries: Number of TCP connect retries (passed to
            :class:`httpx.AsyncHTTPTransport`). Application-level 5xx
            retry is out of scope — those responses surface as
            :class:`MCPError`.
        user_agent: Value of the ``User-Agent`` request header.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 30.0
    connect_retries: int = 0
    user_agent: str = "agent-sdk-mcp/0.0.1"


# ---------------------------------------------------------------------------
# Internal envelope helpers
# ---------------------------------------------------------------------------


def _make_request(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Construct a JSON-RPC 2.0 request envelope with a fresh hex id.

    ``params`` is omitted from the envelope when ``None`` (per spec).
    """
    envelope: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": method,
    }
    if params is not None:
        envelope["params"] = params
    return envelope


def _parse_response(
    envelope: dict[str, Any],
    *,
    expected_id: str,
    method: str,
    server_url: str,
    tool_name: str | None = None,
) -> Any:
    """Validate a JSON-RPC 2.0 response envelope and return ``result``.

    Args:
        envelope: The decoded JSON body returned by the server.
        expected_id: The id of the request we issued. The response id must
            match.
        method: The JSON-RPC method that was sent (echoed into MCPError
            context on failure).
        server_url: For MCPError context.
        tool_name: Set on ``tools/call`` so MCPError context carries the
            offending tool name.

    Returns:
        The unwrapped ``result`` value when the envelope is well-formed.

    Raises:
        MCPError: When the envelope is malformed (missing ``jsonrpc``,
            wrong version, missing/mismatched ``id``, both ``result`` and
            ``error`` present, neither present), or when the server
            returned a JSON-RPC ``error`` object.

    TODO(TD-MCP-NOTIFY): when push refresh is wired, an envelope without
    ``id`` is a notification rather than an error — branch here on
    ``"method" in envelope and "id" not in envelope``.
    """
    base_context: dict[str, Any] = {
        "server_url": server_url,
        "method": method,
    }
    if tool_name is not None:
        base_context["tool_name"] = tool_name

    if envelope.get("jsonrpc") != "2.0":
        raise MCPError(
            "MCP server returned non-2.0 jsonrpc envelope",
            context={**base_context, "envelope_jsonrpc": envelope.get("jsonrpc")},
        )
    if "id" not in envelope:
        raise MCPError(
            "MCP response envelope missing 'id'",
            context=base_context,
        )
    if envelope["id"] != expected_id:
        raise MCPError(
            "MCP response id does not match request id",
            context={
                **base_context,
                "expected_id": expected_id,
                "received_id": envelope["id"],
            },
        )
    has_result = "result" in envelope
    has_error = "error" in envelope
    if has_result == has_error:
        # Both present or neither — both invalid per spec.
        raise MCPError(
            "MCP response envelope must contain exactly one of 'result' or 'error'",
            context={
                **base_context,
                "has_result": has_result,
                "has_error": has_error,
            },
        )
    if has_error:
        err = envelope["error"]
        if not isinstance(err, dict):
            raise MCPError(
                "MCP error payload is not an object",
                context=base_context,
            )
        code = err.get("code")
        message = err.get("message") or "MCP server returned an error"
        raise MCPError(
            str(message),
            context={
                **base_context,
                "error_code": code,
                "error_data": err.get("data"),
            },
        )
    return envelope["result"]


def _redact_headers(
    headers: Mapping[str, str],
    *,
    extra_sensitive_keys: frozenset[str],
) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced.

    Sensitivity is determined case-insensitively against
    :data:`_BUILTIN_SENSITIVE_HEADERS` and ``extra_sensitive_keys`` (the
    header names declared by the caller's ``auth`` mapping).

    The returned dict is suitable for inclusion in :class:`MCPError`
    context or log lines. The sentinel value :data:`_REDACTION_SENTINEL`
    replaces every redacted value verbatim.
    """
    sensitive = _BUILTIN_SENSITIVE_HEADERS | extra_sensitive_keys
    return {
        k: (_REDACTION_SENTINEL if k.lower() in sensitive else v)
        for k, v in headers.items()
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


AuthHeaders = Mapping[str, str]
AuthCallable = Callable[[], Awaitable[AuthHeaders]]
AuthSpec = AuthHeaders | AuthCallable | None


class MCPClient:
    """JSON-RPC 2.0 client for an MCP server over HTTP.

    See module docstring for design notes. Construction never performs
    network I/O; the first call to :meth:`discover` or :meth:`invoke`
    establishes the connection.

    Invariant — auth headers MUST NEVER appear in MCPError.context.
        Every :class:`MCPError` raised by this client builds its
        ``context`` dict from the small allow-list ``{server_url, method,
        tool_name, wrapped, status_code, error_code, error_data,
        expected_id, received_id, has_result, has_error,
        envelope_jsonrpc, content, operation}`` — no header dict is ever
        captured. ``_resolve_auth`` still tracks declared header names
        for any future code path that *does* need to surface headers
        (e.g. a debug-mode dump): that path MUST funnel through
        :func:`_redact_headers` first. The next person tempted to widen
        the context shape: do not put headers in. Use
        :func:`_redact_headers` and prove a test exercises the
        redaction.

    Args:
        config: Connection configuration.
        auth: Either a static header mapping (applied to every request)
            or an async callable invoked on every request (enabling
            token rotation). Header names appearing in this mapping are
            tracked as sensitive and stripped before any header dict
            reaches a log line or :class:`MCPError` context.
        client: An externally-constructed :class:`httpx.AsyncClient`.
            Provided for tests and dependency-injection scenarios; the
            client OWNS the lifecycle of internally-created clients and
            disposes them on :meth:`aclose`, but does NOT close
            externally-provided ones.
    """

    def __init__(
        self,
        config: MCPClientConfig,
        *,
        auth: AuthSpec = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._auth = auth
        # Pre-compute the set of sensitive header names declared by static
        # auth. Callables are resolved per-request, so their declared keys
        # are added to the sensitive set on the fly.
        self._declared_auth_keys: frozenset[str] = (
            frozenset(k.lower() for k in auth)
            if isinstance(auth, Mapping)
            else frozenset()
        )
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=config.connect_timeout_seconds,
                    read=config.read_timeout_seconds,
                    write=config.read_timeout_seconds,
                    pool=config.read_timeout_seconds,
                ),
                transport=httpx.AsyncHTTPTransport(retries=config.connect_retries),
            )
            self._owns_client = True
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self) -> list[MCPToolDef]:
        """Fetch the remote tool catalog via the ``tools/list`` JSON-RPC method.

        Returns:
            A list of :class:`MCPToolDef` parsed from the server's
            ``result.tools`` payload.

        Raises:
            MCPError: For any transport failure, malformed JSON-RPC
                envelope, server-returned error object, or unexpected
                ``result`` shape (e.g. ``tools`` not a list). Also
                raised with ``message == "MCP client is closed"`` when
                invoked after :meth:`aclose`.
        """
        if self._closed:
            raise MCPError(
                "MCP client is closed",
                context={
                    "operation": "discover",
                    "server_url": self._config.base_url,
                },
            ) from None
        result = await self._call("tools/list", params=None)
        return self._parse_tool_catalog(result)

    async def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke a remote tool via the ``tools/call`` JSON-RPC method.

        Args:
            name: Name of the remote tool (matches a
                :attr:`MCPToolDef.name`).
            args: Argument object passed verbatim as
                ``params.arguments``.

        Returns:
            The unwrapped ``result.content`` payload — or, when the
            server omits the ``content`` wrapper, the raw ``result``.
            Falls through to whatever the remote tool produced.

        Raises:
            MCPError: For transport failure, malformed envelope,
                JSON-RPC ``error`` response, or a ``result`` carrying
                ``isError=True``.
        """
        result = await self._call(
            "tools/call",
            params={"name": name, "arguments": args},
            tool_name=name,
        )
        return self._unwrap_invoke_result(result, tool_name=name)

    async def aclose(self) -> None:
        """Close the owned :class:`httpx.AsyncClient`, if any.

        Externally-provided clients are NOT closed — their lifecycle
        belongs to the caller. Idempotent: a second ``aclose()`` is a
        no-op and does NOT raise. After ``aclose()`` returns, calls to
        :meth:`discover` and :meth:`invoke` raise :class:`MCPError`
        with ``message == "MCP client is closed"``.
        """
        if self._closed:
            return
        if self._owns_client:
            await self._client.aclose()
        self._closed = True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _call(
        self,
        method: str,
        *,
        params: dict[str, Any] | None,
        tool_name: str | None = None,
    ) -> Any:
        """Execute one JSON-RPC round-trip and return the validated result.

        Wraps :class:`httpx.HTTPError` and parse failures into
        :class:`MCPError`. Auth headers are resolved fresh per call (so
        callable auth supports token rotation) and never appear in the
        error context.
        """
        if self._closed:
            raise MCPError(
                "MCP client is closed",
                context={
                    "operation": method,
                    "server_url": self._config.base_url,
                },
            ) from None
        envelope = _make_request(method, params)
        request_id: str = envelope["id"]

        resolved_auth, _ = await self._resolve_auth()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._config.user_agent,
        }
        for k, v in resolved_auth.items():
            headers[k] = v

        log = _log.bind(
            method=method,
            request_id=request_id,
            server_url=self._config.base_url,
        )

        base_context: dict[str, Any] = {
            "server_url": self._config.base_url,
            "method": method,
        }
        if tool_name is not None:
            base_context["tool_name"] = tool_name

        try:
            response = await self._client.post(
                self._config.base_url,
                json=envelope,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            # ConnectError, ReadTimeout, etc. — never include headers.
            log.debug("mcp.transport_error", wrapped=type(exc).__name__)
            raise MCPError(
                f"MCP transport error: {type(exc).__name__}",
                context={**base_context, "wrapped": type(exc).__name__},
            ) from exc

        if response.status_code >= 400:
            log.debug(
                "mcp.http_error",
                status_code=response.status_code,
            )
            raise MCPError(
                f"MCP server returned HTTP {response.status_code}",
                context={
                    **base_context,
                    "status_code": response.status_code,
                    "wrapped": "HTTPStatusError",
                },
            )

        try:
            decoded = response.json()
        except ValueError as exc:
            log.debug("mcp.decode_error", wrapped=type(exc).__name__)
            raise MCPError(
                "MCP response body is not valid JSON",
                context={
                    **base_context,
                    "wrapped": type(exc).__name__,
                },
            ) from exc

        if not isinstance(decoded, dict):
            raise MCPError(
                "MCP response body is not a JSON object",
                context={
                    **base_context,
                    "wrapped": type(decoded).__name__,
                },
            )

        # _parse_response raises MCPError on any envelope/protocol issue.
        result = _parse_response(
            decoded,
            expected_id=request_id,
            method=method,
            server_url=self._config.base_url,
            tool_name=tool_name,
        )
        log.debug("mcp.call_ok")
        return result

    async def _resolve_auth(self) -> tuple[Mapping[str, str], frozenset[str]]:
        """Resolve the auth header mapping and the set of declared keys.

        Returns:
            A tuple of ``(headers, declared_keys_lowercase)``. ``headers``
            is the mapping to merge into the outbound request;
            ``declared_keys_lowercase`` lists the header names treated as
            sensitive (used to extend redaction).
        """
        auth = self._auth
        if auth is None:
            return {}, frozenset()
        if isinstance(auth, Mapping):
            return auth, frozenset(k.lower() for k in auth)
        # auth is a callable
        resolved = await auth()
        if not isinstance(resolved, Mapping):
            raise MCPError(
                "auth callable must return a Mapping[str, str]",
                context={
                    "server_url": self._config.base_url,
                    "wrapped": type(resolved).__name__,
                },
            )
        return resolved, frozenset(k.lower() for k in resolved)

    def _parse_tool_catalog(self, result: Any) -> list[MCPToolDef]:
        """Translate the ``tools/list`` ``result`` payload into typed defs.

        Per MCP spec the result is ``{"tools": [<def>, ...]}``. Each def is
        ``{"name": str, "description": str, "inputSchema": {...}}``.
        """
        if not isinstance(result, dict):
            raise MCPError(
                "MCP tools/list result is not an object",
                context={
                    "server_url": self._config.base_url,
                    "method": "tools/list",
                    "wrapped": type(result).__name__,
                },
            )
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise MCPError(
                "MCP tools/list result missing 'tools' array",
                context={
                    "server_url": self._config.base_url,
                    "method": "tools/list",
                },
            )
        out: list[MCPToolDef] = []
        for raw in tools:
            if not isinstance(raw, dict):
                raise MCPError(
                    "MCP tool definition is not an object",
                    context={
                        "server_url": self._config.base_url,
                        "method": "tools/list",
                        "wrapped": type(raw).__name__,
                    },
                )
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                raise MCPError(
                    "MCP tool definition missing 'name'",
                    context={
                        "server_url": self._config.base_url,
                        "method": "tools/list",
                    },
                )
            description = raw.get("description") or ""
            input_schema = raw.get("inputSchema") or raw.get("input_schema") or {}
            if not isinstance(input_schema, dict):
                raise MCPError(
                    "MCP tool 'inputSchema' is not an object",
                    context={
                        "server_url": self._config.base_url,
                        "method": "tools/list",
                        "tool_name": name,
                    },
                )
            out.append(
                MCPToolDef(
                    name=name,
                    description=str(description),
                    input_schema=dict(input_schema),
                )
            )
        return out

    def _unwrap_invoke_result(self, result: Any, *, tool_name: str) -> Any:
        """Unwrap the ``tools/call`` result payload.

        Per MCP spec the success shape is::

            {"content": [...], "isError": false}

        Security note — error-content capture:
            When ``isError`` is True, ``result["content"]`` is captured
            verbatim into ``MCPError.context["content"]``. The SDK has no
            way to know what an MCP server places in that field — a
            non-conformant or poorly-implemented server MAY echo request
            arguments, credentials, or other sensitive material into the
            error content, in which case that material surfaces in our
            error logs. We intentionally preserve the value as-is rather
            than redact it: the SDK does not own the schema, redaction
            would mask real debug info, and the threat model here is a
            misbehaving downstream (which we cannot fix from the client
            side). Consumers running against untrusted MCP servers SHOULD
            scrub ``MCPError.context["content"]`` before logging.

        If ``isError`` is True, raise :class:`MCPError`. If ``content`` is
        present, return it verbatim; otherwise return the raw ``result`` —
        servers vary, and the SDK is intentionally permissive here.
        """
        if not isinstance(result, dict):
            # Some servers may return scalar results; pass them through.
            return result
        if result.get("isError") is True:
            raise MCPError(
                f"MCP tool '{tool_name}' returned isError=True",
                context={
                    "server_url": self._config.base_url,
                    "method": "tools/call",
                    "tool_name": tool_name,
                    "content": result.get("content"),
                },
            )
        if "content" in result:
            return result["content"]
        return result


__all__ = [
    "MCPClient",
    "MCPClientConfig",
    "MCPToolDef",
]
