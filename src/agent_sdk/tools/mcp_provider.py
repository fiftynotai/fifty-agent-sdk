"""Bridge from the :mod:`agent_sdk.mcp` protocol layer into the tool layer.

:class:`MCPProvider` is the only module in the SDK that imports from BOTH
:mod:`agent_sdk.mcp` and :mod:`agent_sdk.tools`. The protocol layer
intentionally has no awareness of :class:`agent_sdk.tools.protocol.Tool` or
:class:`agent_sdk.tools.registry.Registry`; the adapter built here turns
each :class:`agent_sdk.mcp.client.MCPToolDef` into a structurally-compliant
:class:`Tool` and registers it.

Refresh semantics
    :meth:`MCPProvider.refresh` is the always-available manual path; it
    re-discovers the remote catalog and re-registers every tool, leveraging
    :meth:`agent_sdk.tools.registry.Registry.register`'s last-write-wins
    overwrite (registry.py:62-65). Tools that disappeared upstream are NOT
    unregistered — the :class:`Registry` has no ``unregister`` (BR-008) and
    the consistency requirement around in-flight
    :class:`agent_sdk.loop.AgentLoop` snapshots makes silent removal a
    foot-gun.

    Periodic refresh is opt-in via
    :meth:`MCPProvider.start_periodic_refresh` and runs as a background
    :class:`asyncio.Task`. Per TD-003 the loop snapshots
    :meth:`agent_sdk.tools.registry.Registry.list` at run start, so
    mid-run refresh is benign for an already-running loop but is NOT a
    contractual ordering — callers should treat refresh as happening
    between :meth:`agent_sdk.loop.AgentLoop.run` calls. Refresh DURING a
    run has undefined visibility into the running loop's tool universe.

Tool-name collisions
    :meth:`MCPProvider.attach` and :meth:`MCPProvider.refresh` detect when
    a tool name they are about to register already exists in the target
    :class:`Registry` and log a ``structlog`` warning per collision. The
    :class:`Registry`'s own overwrite warning still fires; the provider's
    warning carries the MCP context (server URL) the registry does not
    know.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Final

import structlog
from pydantic import BaseModel, ConfigDict

from agent_sdk.mcp import MCPClient, MCPToolDef
from agent_sdk.tools.protocol import ToolResult, ToolSchema
from agent_sdk.tools.registry import Registry

_log: Final = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RefreshSummary(BaseModel):
    """Outcome counts returned by :meth:`MCPProvider.refresh`.

    Attributes:
        added: Number of MCP tools registered for the first time during
            this refresh (their name was not in the registry at refresh
            entry).
        refreshed: Number of MCP tools re-registered with an existing
            name (last-write-wins per
            :meth:`agent_sdk.tools.registry.Registry.register`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    added: int = 0
    refreshed: int = 0


# ---------------------------------------------------------------------------
# Internal adapter
# ---------------------------------------------------------------------------


def _to_tool_schema(input_schema: dict[str, Any]) -> ToolSchema:
    """Translate an MCP ``inputSchema`` JSON-Schema dict into a :class:`ToolSchema`.

    MCP requires ``"type": "object"`` at the top level. If the server
    violates that, the adapter falls back to an empty object schema and
    logs a WARNING — the resulting tool will reject any args via Pydantic
    validation downstream, which is the desired defensive behavior.

    Unknown top-level JSON-Schema keys (``$defs``, ``examples``, …) are
    dropped: :class:`ToolSchema` is ``extra="forbid"`` and exists to
    describe the parameter surface the LLM sees, not to round-trip the
    full JSON-Schema vocabulary.
    """
    raw_type = input_schema.get("type", "object")
    if raw_type != "object":
        _log.warning(
            "mcp.input_schema.non_object",
            type=raw_type,
        )
        return ToolSchema(
            type="object",
            properties={},
            required=[],
            additionalProperties=False,
        )
    properties_raw = input_schema.get("properties", {})
    required_raw = input_schema.get("required", [])
    properties: dict[str, Any] = dict(properties_raw) if isinstance(properties_raw, dict) else {}
    required: list[str] = [str(r) for r in required_raw] if isinstance(required_raw, list) else []
    return ToolSchema(
        type="object",
        properties=properties,
        required=required,
        additionalProperties=False,
    )


class _MCPToolAdapter:
    """Per-tool adapter satisfying :class:`agent_sdk.tools.protocol.Tool`.

    Constructed by :class:`MCPProvider` — one instance per MCP-advertised
    tool. ``invoke`` delegates to :meth:`agent_sdk.mcp.client.MCPClient.invoke`
    and translates the result into a
    :class:`agent_sdk.tools.protocol.ToolResult`. The adapter does NOT catch
    :class:`agent_sdk.errors.MCPError` — by design, the
    :class:`agent_sdk.tools.registry.Registry` re-raises
    :class:`agent_sdk.errors.AgentSdkError` subclasses untouched
    (registry.py:154), giving the surrounding runner the chance to surface
    the MCP failure as a system error rather than as a per-tool recoverable
    failure.
    """

    def __init__(self, defn: MCPToolDef, client: MCPClient) -> None:
        self._defn = defn
        self._client = client
        self.name: str = defn.name
        self.description: str = defn.description
        self.schema: ToolSchema = _to_tool_schema(defn.input_schema)

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Delegate to the underlying :class:`MCPClient`.

        On success returns ``ToolResult(output=..., is_error=False)``. On
        :class:`agent_sdk.errors.MCPError` the exception propagates
        untouched per the Tool protocol's "system failure" escape route
        (protocol.py:107-115).
        """
        output = await self._client.invoke(self._defn.name, args)
        return ToolResult(output=output, is_error=False, error=None)


# ---------------------------------------------------------------------------
# Public provider
# ---------------------------------------------------------------------------


class MCPProvider:
    """Adapt an :class:`MCPClient` into a tool provider for the SDK :class:`Registry`.

    A provider is bound to ONE :class:`MCPClient`. Wiring multiple MCP
    servers means constructing multiple providers, one per client, and
    attaching each to the same registry. Tool-name collisions between
    servers are detected and logged per registration; the
    :class:`Registry`'s last-write-wins semantics decide which adapter
    actually services invocations.

    Args:
        client: The :class:`MCPClient` to discover tools from. The
            provider does NOT own the client's lifecycle —
            :meth:`MCPProvider.aclose` cancels the background refresh
            task only.

    Background-refresh lifecycle:
        Calling :meth:`start_periodic_refresh` spawns an
        :class:`asyncio.Task` that calls :meth:`refresh` on a fixed
        interval. The task survives until :meth:`aclose` is called or
        the event loop is torn down; a second
        :meth:`start_periodic_refresh` without an intervening
        :meth:`aclose` raises :class:`RuntimeError`.

    Mid-run refresh caveat:
        Per TD-003, :class:`agent_sdk.loop.AgentLoop` snapshots
        :meth:`Registry.list` at the START of every
        :meth:`agent_sdk.loop.AgentLoop.run`. Refreshes that complete
        during a run will be visible to subsequent ``run`` calls but
        NOT to the in-flight one. Refresh DURING a run is therefore
        benign-but-undefined: do not rely on it for correctness.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        self._refresh_task: asyncio.Task[None] | None = None
        self._registry: Registry | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def attach(self, registry: Registry) -> None:
        """Discover MCP tools and register one adapter per tool into ``registry``.

        Idempotent: calling ``attach`` twice re-registers every tool
        (last-write-wins per
        :meth:`agent_sdk.tools.registry.Registry.register`). The provider
        retains a reference to ``registry`` so that subsequent
        :meth:`refresh` calls write into the same registry; calling
        ``attach`` with a different registry rebinds.

        Async because it issues a :meth:`MCPClient.discover` call.

        Args:
            registry: The tool registry to populate.

        Raises:
            agent_sdk.errors.MCPError: On any discovery failure
                (transport, protocol, envelope).
        """
        self._registry = registry
        await self._discover_and_register(registry)

    async def refresh(self) -> RefreshSummary:
        """Re-discover the remote catalog and reconcile against the registry.

        Newly-advertised tools are registered fresh; previously-advertised
        tools are re-registered (their adapter is rebuilt — useful when
        the remote schema changed). Tools removed upstream are NOT
        unregistered; the :class:`Registry` has no ``unregister``, and
        silent removal would race against an in-flight
        :class:`agent_sdk.loop.AgentLoop.run` snapshot.

        Returns:
            A :class:`RefreshSummary` reporting how many adapters were
            newly added vs. re-registered with an existing name.

        Raises:
            RuntimeError: If :meth:`attach` has not been called.
            agent_sdk.errors.MCPError: On discovery failure.
        """
        if self._registry is None:
            raise RuntimeError("MCPProvider.refresh() requires a prior attach(registry) call")
        return await self._discover_and_register(self._registry)

    async def start_periodic_refresh(self, interval_seconds: float) -> None:
        """Spawn a background :class:`asyncio.Task` that refreshes on a schedule.

        Opt-in. The first refresh fires after ``interval_seconds``, not
        immediately — :meth:`attach` is the always-immediate path.

        Args:
            interval_seconds: Seconds between refreshes. Must be > 0.

        Raises:
            RuntimeError: If a refresh task is already running (call
                :meth:`aclose` first) or :meth:`attach` has not yet been
                called.
            ValueError: If ``interval_seconds`` is not positive.
        """
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0; got {interval_seconds!r}")
        if self._refresh_task is not None and not self._refresh_task.done():
            raise RuntimeError(
                "MCPProvider periodic refresh is already running; call aclose() first"
            )
        if self._registry is None:
            raise RuntimeError(
                "MCPProvider.start_periodic_refresh() requires a prior attach(registry) call"
            )
        self._refresh_task = asyncio.create_task(
            self._periodic_loop(interval_seconds),
            name="mcp-provider-refresh",
        )

    async def aclose(self) -> None:
        """Cancel the background refresh task (if any).

        Does NOT close the underlying :class:`MCPClient` — that lifecycle
        is the caller's responsibility. Safe to call when no refresh
        task is running.
        """
        task = self._refresh_task
        if task is None:
            return
        self._refresh_task = None
        if task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _discover_and_register(self, registry: Registry) -> RefreshSummary:
        """Shared body of :meth:`attach` and :meth:`refresh`."""
        defs = await self._client.discover()
        existing = {t.name for t in registry.list()}
        added = 0
        refreshed = 0
        for defn in defs:
            adapter = _MCPToolAdapter(defn, self._client)
            if defn.name in existing:
                _log.warning(
                    "mcp.tool_overwrite",
                    name=defn.name,
                    reason="name already present in registry",
                )
                refreshed += 1
            else:
                added += 1
                existing.add(defn.name)
            # _MCPToolAdapter structurally satisfies Tool; mypy resolves
            # the protocol match via runtime_checkable.
            registry.register(adapter)
        return RefreshSummary(added=added, refreshed=refreshed)

    async def _periodic_loop(self, interval_seconds: float) -> None:
        """Background task body: sleep, refresh, repeat until cancelled.

        Discovery failures are logged at WARNING (with the MCPError
        ``context``) but do NOT terminate the task — a transient server
        outage shouldn't permanently stop refresh.
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                return
            try:
                await self.refresh()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 — broad catch is intentional.
                _log.warning(
                    "mcp.refresh_failed",
                    wrapped=type(exc).__name__,
                    message=str(exc),
                )


__all__ = [
    "MCPProvider",
    "RefreshSummary",
]
