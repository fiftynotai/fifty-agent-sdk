"""Transport abstraction for the MCP client wrapper.

The MCP protocol/wire is owned by the official :mod:`mcp` Python SDK
(``ClientSession`` over a stream pair). This module isolates *how the stream
pair is established* behind a small :class:`Transport` protocol so the
:class:`agent_sdk.mcp.client.MCPClient` is transport-agnostic.

Streamable HTTP today, stdio later
    :class:`StreamableHttpTransport` is the only concrete transport shipped
    today. It builds the auth/timeout-configured :class:`httpx.AsyncClient`,
    opens :func:`mcp.client.streamable_http.streamable_http_client`, and
    wraps the resulting stream pair in an *already-initialized*
    :class:`mcp.ClientSession`. A future ``StdioTransport`` would implement
    the SAME :meth:`Transport.connect` contract over
    :func:`mcp.client.stdio.stdio_client` — and nothing above this module
    (notably :class:`agent_sdk.tools.mcp_provider.MCPProvider`) changes,
    because it only ever sees an :class:`agent_sdk.mcp.client.MCPClient`.

The auth/header/timeout seam
    In ``mcp 1.27.0`` the transport-level ``headers=/timeout=/auth=`` params
    are deprecated and **silently ignored** at runtime. The supported seam is
    the :class:`httpx.AsyncClient` passed as ``http_client=`` to
    :func:`streamable_http_client`. Auth headers, timeouts, and the
    user-agent therefore live on the httpx client built here — never on the
    transport/session.

Lifecycle
    :meth:`Transport.connect` is an async context manager yielding a session
    that has already completed the ``initialize`` handshake. Exiting the
    context unwinds the session and the underlying ``streamable_http_client``
    streams. Per the :class:`agent_sdk.mcp.client.MCPClient` ``client=``
    contract, an *externally-provided* httpx client is NEVER closed here; an
    internally-created one is owned by the :class:`MCPClient` and closed in
    its :meth:`aclose`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from typing import Protocol, runtime_checkable

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation


@runtime_checkable
class Transport(Protocol):
    """A source of already-``initialize()``d :class:`mcp.ClientSession` objects.

    A transport encapsulates the stream-pair establishment (HTTP, stdio, …)
    and the ``initialize`` handshake. :class:`agent_sdk.mcp.client.MCPClient`
    opens one session per ``discover``/``invoke`` call (session-per-call), so
    :meth:`connect` is entered and exited around each operation.
    """

    def connect(self) -> AbstractAsyncContextManager[ClientSession]:  # pragma: no cover
        """Return an async context manager yielding an initialized session.

        Implementations decorate the method body with
        :func:`contextlib.asynccontextmanager`; that decoration produces an
        object satisfying :class:`contextlib.AbstractAsyncContextManager`,
        which is what callers ``async with``.
        """
        ...


class StreamableHttpTransport:
    """Streamable HTTP transport over the official :mod:`mcp` SDK.

    Builds the auth/timeout/user-agent-configured :class:`httpx.AsyncClient`
    (the supported seam — see module docstring), opens
    :func:`mcp.client.streamable_http.streamable_http_client`, and yields an
    initialized :class:`mcp.ClientSession`.

    Args:
        base_url: Full URL of the MCP server's Streamable HTTP endpoint.
        read_timeout_seconds: Per-call read timeout for the session
            (``ClientSession(read_timeout_seconds=...)``) and the read slot of
            the httpx timeout.
        connect_timeout_seconds: TCP connect timeout for the httpx client.
        user_agent: ``User-Agent`` header set on the owned httpx client. When
            an external ``http_client`` is provided, its own headers win and
            this value is ignored (config-shape stability; see
            :class:`agent_sdk.mcp.client.MCPClientConfig`).
        connect_retries: Number of TCP connect retries for the owned httpx
            transport. When an external ``http_client`` is provided this is a
            documented no-op (the external client's transport governs).
        headers: Resolved auth/header mapping merged onto the owned httpx
            client. Resolved fresh per-call by the client so callable auth
            keeps rotating; never logged, never captured into error context.
        http_client: An externally-provided client. When set, it is used
            verbatim (its headers/transport/timeout win) and is NOT closed by
            this transport.
    """

    def __init__(
        self,
        *,
        base_url: str,
        read_timeout_seconds: float,
        connect_timeout_seconds: float,
        user_agent: str,
        connect_retries: int,
        headers: Mapping[str, str],
        http_client: httpx.AsyncClient | None,
    ) -> None:
        self._base_url = base_url
        self._read_timeout_seconds = read_timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._user_agent = user_agent
        self._connect_retries = connect_retries
        self._headers = dict(headers)
        self._http_client = http_client

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[ClientSession]:
        """Open an initialized session for a single operation.

        Builds (or reuses the injected) httpx client, opens the Streamable
        HTTP streams, wraps them in a :class:`mcp.ClientSession`, runs
        ``initialize()``, and yields the session. The session, streams, and
        any internally-built httpx client are unwound on exit; an injected
        client is left open.
        """
        client = self._http_client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self._connect_timeout_seconds,
                    read=self._read_timeout_seconds,
                    write=self._read_timeout_seconds,
                    pool=self._read_timeout_seconds,
                ),
                transport=httpx.AsyncHTTPTransport(retries=self._connect_retries),
                headers={"User-Agent": self._user_agent, **self._headers},
            )
        else:
            # Externally-provided client: merge the per-call auth headers IN
            # PLACE on the injected client's default-header set. This DOES
            # mutate the caller's client — intentionally: per-call resolution
            # (callable auth) rotates tokens, so each call must overwrite the
            # client's auth header with the freshly-resolved value. Callers
            # injecting a client therefore share its header state with this
            # transport; an MCPClient that owns its client (the common path)
            # builds a fresh client per call and is unaffected.
            for key, value in self._headers.items():
                client.headers[key] = value
        try:
            async with (
                streamable_http_client(self._base_url, http_client=client) as (
                    read_stream,
                    write_stream,
                    _get_session_id,
                ),
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout_seconds),
                    client_info=Implementation(name=self._user_agent, version="1.0.0"),
                ) as session,
            ):
                await session.initialize()
                yield session
        finally:
            if owns_client:
                await client.aclose()


__all__ = [
    "StreamableHttpTransport",
    "Transport",
]
