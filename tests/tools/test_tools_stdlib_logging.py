"""``Registry`` and ``MCPProvider`` under structlog routed through stdlib ``logging`` (BR-026, AC-2).

Before BR-026 three WARNING lines passed a key stdlib's ``LogRecord``
reserves: ``tool overwritten`` (``Registry.register``, ``name=``),
``mcp.tool_overwrite`` (``MCPProvider.attach``/``refresh()``, ``name=``)
and ``mcp.refresh_failed`` (the periodic refresh, ``message=``). With
structlog routed through stdlib by ``render_to_log_kwargs`` or
``render_to_log_args_and_kwargs``, at DEBUG, at INFO and with no level set
(stdlib's default passes WARNING), measured on structlog 26.1.0 with CPython
3.11.15, 3.13.2 and 3.14.3 (BR-026 evidence, P2):

* ``register`` of a name already registered raised ``KeyError`` for
  ``'name'``, and the registry kept the earlier tool;
* ``attach`` raised it at a server tool whose name was already registered
  (with a local ``search`` and the catalog ``[alpha, search, zeta]``,
  ``alpha`` was registered, ``search`` stayed local and ``zeta`` was not
  registered), and ``refresh()`` of an unchanged one-tool catalog raised it;
* the periodic refresh task ended at its first tick with ``KeyError`` for
  ``'message'``, whether that tick's refresh failed or refreshed an
  attached tool, and no later refresh ran. A failed refresh raised at
  ``mcp.refresh_failed`` in the task's ``except`` arm; a refresh of an
  attached tool raised at ``mcp.tool_overwrite``, and the ``except`` arm
  then raised at ``mcp.refresh_failed`` (by reading).

Each line now passes ``tool_name=`` or ``error_message=``. Every test runs
inside ``tests.stdlib_routing.stdlib_routed_structlog``, at WARNING (the
level stdlib's default lets through), INFO and DEBUG. The
``wrap_for_formatter`` rows are the control recipe, which never passed the
SDK's keys to ``makeRecord``.

The periodic rows wait (at most 5 s) for a later tick's refresh to run; a
dead task can never satisfy them, and none of them calls ``refresh()`` by
hand afterwards. ``failing_tick`` replaces ``refresh`` on the instance (the
background loop calls ``self.refresh()``), so it isolates
``mcp.refresh_failed``; the other two rows wrap the real ``refresh`` and
only count its outcomes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from fifty_agent_sdk.errors import MCPError
from fifty_agent_sdk.tools.mcp_provider import MCPProvider, RefreshSummary
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema
from fifty_agent_sdk.tools.registry import Registry
from tests.mcp.conftest import ControllableServer, make_controllable_client
from tests.stdlib_routing import (
    ALL_RECIPES,
    record_field,
    record_keys,
    records_for,
    stdlib_routed_structlog,
)

_LEVELS = [
    pytest.param(logging.WARNING, id="WARNING"),
    pytest.param(logging.INFO, id="INFO"),
    pytest.param(logging.DEBUG, id="DEBUG"),
]
_PROVIDER_LOGGER = "fifty_agent_sdk.tools.mcp_provider"
_REGISTRY_LOGGER = "fifty_agent_sdk.tools.registry"


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = "local"
        self.schema = ToolSchema()

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output="local")


def _tool_def(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Tool {name}",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    }


def _assert_overwrite_record(record: logging.LogRecord, logger: str, tool_name: str) -> None:
    assert record.name == logger
    assert record.levelno == logging.WARNING
    assert record_field(record, "tool_name") == tool_name


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("recipe", ALL_RECIPES)
def test_registry_overwrite_logs_under_stdlib_routing(recipe: str, level: int) -> None:
    """S5: ``register`` of a registered name returns, keeps the new tool and logs one WARNING with ``tool_name`` (BR-026, AC-2)."""
    registry = Registry()
    first, second = _Tool("same"), _Tool("same")
    registry.register(first)  # type: ignore[arg-type]

    with stdlib_routed_structlog(recipe, level) as records:
        registry.register(second)  # type: ignore[arg-type]

    assert registry.get("same") is second
    (record,) = records_for(records, "tool overwritten")
    _assert_overwrite_record(record, _REGISTRY_LOGGER, "same")
    if recipe == "wrap_for_formatter":
        assert record_keys(record) == {"tool_name", "level"}


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize("row", ["attach_over_local_tool", "refresh_unchanged_catalog"])
@pytest.mark.parametrize("recipe", ALL_RECIPES)
async def test_mcp_name_collision_logs_under_stdlib_routing(
    controllable_server: ControllableServer, recipe: str, row: str, level: int
) -> None:
    """S6: ``attach`` over a local tool and ``refresh()`` of an attached catalog return and log ``tool_name`` (BR-026, AC-2).

    ``attach_over_local_tool``: catalog ``[alpha, search, zeta]`` with a
    local ``search`` registered first; every catalog name ends up
    registered, ``search`` as the MCP adapter. ``refresh_unchanged_catalog``:
    a one-tool catalog attached, then refreshed, which re-registers it.
    """
    registry = Registry()
    provider = MCPProvider(make_controllable_client(controllable_server))
    if row == "attach_over_local_tool":
        local = _Tool("search")
        registry.register(local)  # type: ignore[arg-type]
        controllable_server.set_tool_catalog(
            [_tool_def("alpha"), _tool_def("search"), _tool_def("zeta")]
        )
        with stdlib_routed_structlog(recipe, level) as records:
            await provider.attach(registry)
        assert sorted(tool.name for tool in registry.list()) == ["alpha", "search", "zeta"]
        assert registry.get("search") is not local
        collided = "search"
    else:
        controllable_server.set_tool_catalog([_tool_def("A")])
        with stdlib_routed_structlog(recipe, level) as records:
            await provider.attach(registry)
            attached = registry.get("A")
            summary = await provider.refresh()
        assert summary == RefreshSummary(added=0, refreshed=1)
        assert registry.get("A") is not attached
        collided = "A"

    (mcp_record,) = records_for(records, "mcp.tool_overwrite")
    _assert_overwrite_record(mcp_record, _PROVIDER_LOGGER, collided)
    assert record_field(mcp_record, "reason") == "name already present in registry"
    (registry_record,) = records_for(records, "tool overwritten")
    _assert_overwrite_record(registry_record, _REGISTRY_LOGGER, collided)
    if recipe == "wrap_for_formatter":
        assert record_keys(mcp_record) == {"tool_name", "reason", "level"}
        assert record_keys(registry_record) == {"tool_name", "level"}


@pytest.mark.parametrize("level", _LEVELS)
@pytest.mark.parametrize(
    "row", ["failing_tick", "server_failure_then_refresh", "unchanged_catalog"]
)
@pytest.mark.parametrize("recipe", ALL_RECIPES)
async def test_mcp_periodic_refresh_keeps_running_under_stdlib_routing(
    controllable_server: ControllableServer, recipe: str, row: str, level: int
) -> None:
    """S7: the periodic refresh task outlives a failing tick and a tick that refreshes an attached tool (BR-026, AC-2).

    * ``failing_tick``: ``refresh`` is replaced; call 1 raises
      ``MCPError("transient")`` and the test waits for call 2.
    * ``server_failure_then_refresh``: the real ``refresh``; the server's
      first ``tools/list`` after ``attach`` raises ``MCPError``, and the
      test waits for a later tick's refresh to succeed.
    * ``unchanged_catalog``: the real ``refresh`` with no failure; the test
      waits for two ticks' refreshes to succeed.

    Before BR-026, under the ``extra`` recipes, the task ended at its first
    tick in every row (P2), so each of those rows timed out.
    """
    controllable_server.set_tool_catalog([_tool_def("A")])
    registry = Registry()
    provider = MCPProvider(make_controllable_client(controllable_server))
    await provider.attach(registry)
    outcomes: list[str] = []
    reached = asyncio.Event()

    if row == "failing_tick":

        async def refresh() -> RefreshSummary:
            outcomes.append("failed" if not outcomes else "ok")
            if outcomes == ["failed"]:
                raise MCPError("transient")
            reached.set()
            return RefreshSummary(added=0, refreshed=0)

    else:
        real_refresh = provider.refresh
        if row == "server_failure_then_refresh":
            real_list = controllable_server.list_tools_result
            fail_next = [True]

            def flaky_list() -> Any:
                if fail_next:
                    fail_next.clear()
                    raise MCPError("transient", context={"server_url": "x"})
                return real_list()

            controllable_server.list_tools_result = flaky_list  # type: ignore[method-assign]
        successes_needed = 2 if row == "unchanged_catalog" else 1

        async def refresh() -> RefreshSummary:
            try:
                summary = await real_refresh()
            except Exception:
                outcomes.append("failed")
                raise
            outcomes.append("ok")
            if outcomes.count("ok") >= successes_needed:
                reached.set()
            return summary

    provider.refresh = refresh  # type: ignore[method-assign]

    with stdlib_routed_structlog(recipe, level) as records:
        await provider.start_periodic_refresh(interval_seconds=0.001)
        try:
            async with asyncio.timeout(5):
                await reached.wait()
        finally:
            await provider.aclose()

    failed = records_for(records, "mcp.refresh_failed")
    overwrites = records_for(records, "mcp.tool_overwrite")
    if row == "unchanged_catalog":
        assert outcomes[:2] == ["ok", "ok"]
        assert failed == []
        assert len(overwrites) >= 2
        for record in overwrites:
            _assert_overwrite_record(record, _PROVIDER_LOGGER, "A")
        return

    assert outcomes[:2] == ["failed", "ok"]
    (record,) = failed
    assert record.name == _PROVIDER_LOGGER
    assert record.levelno == logging.WARNING
    assert record_field(record, "error_message") == "transient"
    assert record_field(record, "wrapped") == "MCPError"
    if recipe == "wrap_for_formatter":
        assert record_keys(record) == {"wrapped", "error_message", "level"}
    if row == "server_failure_then_refresh":
        assert overwrites
        for overwrite in overwrites:
            _assert_overwrite_record(overwrite, _PROVIDER_LOGGER, "A")
