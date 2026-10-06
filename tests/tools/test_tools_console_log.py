"""MCP and registry log lines under structlog's default configuration, with control characters (BR-028).

Six SDK log calls (five events) besides the loop's ``tool_invoked`` lines write a string
an MCP server controls as a top-level value: ``mcp.tool_overwrite``
(``MCPProvider.attach`` and ``refresh()``), ``tool overwritten``
(``Registry.register``, which gets the server's name when ``MCPProvider``
registers an attached tool again), ``mcp.refresh_failed`` (the periodic
refresh; ``error_message`` is the exception's text, for a JSON-RPC error the
server's own message) and the three ``mcp.tool_error_hook_*`` calls
(``MCPClient``, the server's tool name). Before BR-028, structlog's default
``ConsoleRenderer`` wrote such a value as it was (structlog 26.1.0 unless it
held one of the seven characters listed below; 24.1.0 always), so a control
character in it reached the log output: measured with structlog 26.1.0 and
24.1.0 on CPython 3.11.15, 3.13.2 and 3.14.3, through the controllable transport below
and through real in-memory ``mcp`` sessions (a FastMCP server with a tool
named ``find`` + ESC + ``[2J``, and a low-level server whose ``tools/list``
raised that kind of message) (BR-028 evidence, P1c). Each value is now
written through ``_model_json.escape_for_log``: each control character
U+0000-U+001F and U+007F-U+009F as a backslash, ``u`` and four lowercase hex
digits.

* T1: ``mcp.refresh_failed`` with a JSON-RPC error message holding ESC,
  through the periodic refresh, which a later tick's successful refresh
  shows still running (no manual ``refresh()``).
* T2: ``mcp.tool_overwrite`` and ``tool overwritten`` for a server tool name
  holding ESC; the registry key keeps the server's name.
* T3: the three hook lines, for a hook that raises, returns a non-string
  and returns a blank string.
* T4 (control): the same paths with values without a control character are
  logged as before, including a message holding spaces, which structlog
  26.1.0 writes with ``repr`` (24.1.0 as it is).
* T5 (control): a protocol-violating tool whose ``name`` is not a ``str``
  is still logged as it is on the registry line, and ``register`` still
  returns.
* T6: the same lines now write a surrogate code point (U+D800-U+DFFF) as
  BR-024's six-character escape, which ``escape_for_log`` applies first.
  Before BR-028 these calls passed it as it was, and with structlog's
  default configuration printing to a strict UTF-8 stream (a file or a
  terminal under a UTF-8 locale) each raised ``UnicodeEncodeError`` for a
  value the renderer wrote as it was:
  ``register`` kept the earlier tool, ``refresh()`` raised, the periodic
  refresh task ended, and a hook line's raise became the tool call's error
  text (measured through the controllable transport, with structlog 26.1.0
  and 24.1.0 on CPython 3.11.15, 3.13.2 and 3.14.3, and on 1.7.0, 1.8.0,
  1.9.0 and 1.10.1; BR-028 evidence, P3). ``tool_invoked`` has escaped it
  since BR-024.

Each test runs inside ``tests.structlog_chains.configured_structlog()``,
structlog's default configuration on a ``StringIO``. The lines are read with
this module's own parser, which splits on ``"\\n"`` only and removes only
SGR colour codes: ``splitlines()`` would also split on VT, FF, NEL and
U+2028, and a wider ANSI strip would remove a raw ``ESC [2J`` too. The
values hold none of the seven characters for which structlog 26.1.0's
renderer switches to ``repr`` (space, tab, ``=``, ``"``, ``'``, CR, LF),
apart from T4's spaced row, so a missing escape would show.

The transport is the controllable server of ``tests/mcp/conftest.py``,
which runs the SDK's production ``MCPClient`` mapping, unwrap and error
translation over ``mcp.types`` results. What these do NOT pin: a real HTTP
MCP server, renderers other than the default ``ConsoleRenderer`` (the
evidence's P1 covers them), and Windows.
"""

from __future__ import annotations

import asyncio
import io
import re
from collections.abc import Callable
from typing import Any

import pytest
import structlog
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from fifty_agent_sdk.tools.mcp_provider import MCPProvider, RefreshSummary
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema
from fifty_agent_sdk.tools.registry import Registry
from tests.mcp.conftest import ControllableServer, make_controllable_client
from tests.structlog_chains import configured_structlog

_CSI_NAME = "find\x1b[2J"
_CSI_NAME_LOGGED = "find\\u001b[2J"
_CSI_MESSAGE = "denied\x1b[2J"
_CSI_MESSAGE_LOGGED = "denied\\u001b[2J"

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _is_control(ch: str) -> bool:
    return ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F


def _lines(buffer: io.StringIO, event: str) -> list[str]:
    """The lines holding ``event``, split on ``"\\n"`` only, with only SGR colour codes removed."""
    text = _SGR.sub("", buffer.getvalue())
    return [line for line in text.split("\n") if f" {event} " in line]


def _value_after(line: str, key: str) -> str:
    """The rest of the line after `` key=``: the renderer sorts keys, so pass the last one."""
    _, separator, value = line.partition(f" {key}=")
    assert separator, line
    return value


def _assert_clean(line: str) -> None:
    assert not any(_is_control(ch) for ch in line), line


def _tool_def(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Tool {name}",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    }


class _Tool:
    def __init__(self, name: Any) -> None:
        self.name = name
        self.description = "local"
        self.schema = ToolSchema()

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output="local")


def _renderer_writes_with_repr(value: str) -> bool:
    """Whether the installed structlog's default ``ConsoleRenderer`` writes ``value`` with ``repr``."""
    rendered = structlog.dev.ConsoleRenderer(colors=False)(None, "info", {"event": "e", "k": value})
    return f"k={value!r}" in rendered


async def _refresh_failure(server: ControllableServer, message: str) -> list[str]:
    """Attach, make the server's next ``tools/list`` fail with ``message``, run the periodic refresh.

    Returns the ``mcp.refresh_failed`` lines once a later tick's refresh has
    succeeded (at most 5 s); a dead task cannot get there.
    """
    server.set_tool_catalog([_tool_def("A")])
    registry = Registry()
    provider = MCPProvider(make_controllable_client(server))
    await provider.attach(registry)
    real_list = server.list_tools_result
    fail_next = [True]

    def flaky_list() -> Any:
        if fail_next:
            fail_next.clear()
            raise McpError(ErrorData(code=-32000, message=message))
        return real_list()

    server.list_tools_result = flaky_list  # type: ignore[method-assign]
    real_refresh = provider.refresh
    outcomes: list[str] = []
    reached = asyncio.Event()

    async def refresh() -> RefreshSummary:
        try:
            summary = await real_refresh()
        except Exception:
            outcomes.append("failed")
            raise
        outcomes.append("ok")
        reached.set()
        return summary

    provider.refresh = refresh  # type: ignore[method-assign]

    with configured_structlog() as buffer:
        await provider.start_periodic_refresh(interval_seconds=0.001)
        try:
            async with asyncio.timeout(5):
                await reached.wait()
        finally:
            await provider.aclose()

    assert outcomes[:2] == ["failed", "ok"]
    return _lines(buffer, "mcp.refresh_failed")


async def _overwrite(server: ControllableServer, name: str) -> tuple[Registry, io.StringIO]:
    """Attach the catalog ``[name]``, then ``refresh()``, which registers the tool again."""
    server.set_tool_catalog([_tool_def(name)])
    registry = Registry()
    provider = MCPProvider(make_controllable_client(server))
    with configured_structlog() as buffer:
        await provider.attach(registry)
        summary = await provider.refresh()
    assert summary == RefreshSummary(added=0, refreshed=1)
    return registry, buffer


def _hook(kind: str) -> Callable[[str, Any], Any]:
    def hook(message: str, content: Any) -> Any:
        if kind == "failed":
            raise ValueError("hook failed")
        if kind == "not_a_string":
            return 42
        return "  "

    return hook


async def _hook_line(server: ControllableServer, name: str, kind: str) -> str:
    """Attach the catalog ``[name]`` with no handler (an ``isError`` result) and invoke it under the hook."""
    server.set_tool_catalog([_tool_def(name)])
    registry = Registry()
    provider = MCPProvider(make_controllable_client(server, on_tool_error=_hook(kind)))
    await provider.attach(registry)
    with configured_structlog() as buffer:
        result = await registry.invoke(name, {}, timeout=None)

    assert result.is_error
    assert result.error == f"MCP tool '{name}' returned isError=True"
    event = "mcp.tool_error_hook_failed" if kind == "failed" else "mcp.tool_error_hook_invalid"
    (line,) = _lines(buffer, event)
    if kind != "failed":
        assert f" reason={kind} " in line
    return line


async def test_mcp_refresh_failed_writes_the_server_message_escaped(
    controllable_server: ControllableServer,
) -> None:
    """T1: a JSON-RPC error message holding ESC is written escaped under ``error_message``, and the periodic refresh keeps running (BR-028)."""
    (line,) = await _refresh_failure(controllable_server, _CSI_MESSAGE)

    match = re.search(r" error_message=(\S+)", line)
    assert match is not None, line
    assert match.group(1) == _CSI_MESSAGE_LOGGED
    assert line.endswith(" wrapped=MCPError")
    _assert_clean(line)


async def test_mcp_and_registry_overwrite_lines_write_the_server_name_escaped(
    controllable_server: ControllableServer,
) -> None:
    """T2: ``mcp.tool_overwrite`` and ``tool overwritten`` write a server tool name holding ESC escaped; the registry key keeps the server's name (BR-028)."""
    registry, buffer = await _overwrite(controllable_server, _CSI_NAME)

    (mcp_line,) = _lines(buffer, "mcp.tool_overwrite")
    (registry_line,) = _lines(buffer, "tool overwritten")
    for line in (mcp_line, registry_line):
        assert _value_after(line, "tool_name") == _CSI_NAME_LOGGED
        _assert_clean(line)
    assert [tool.name for tool in registry.list()] == [_CSI_NAME]
    assert registry.get(_CSI_NAME).name == _CSI_NAME


@pytest.mark.parametrize("kind", ["failed", "not_a_string", "blank_string"])
async def test_mcp_tool_error_hook_lines_write_the_tool_name_escaped(
    controllable_server: ControllableServer, kind: str
) -> None:
    """T3: each ``mcp.tool_error_hook_*`` line writes a server tool name holding ESC escaped, and the call still ends as the server's ``isError`` result (BR-028)."""
    line = await _hook_line(controllable_server, _CSI_NAME, kind)

    assert _value_after(line, "tool_name") == _CSI_NAME_LOGGED
    _assert_clean(line)


@pytest.mark.parametrize(
    "row", ["refresh_denied", "refresh_spaced", "overwrite_arabic", "hook_boom"]
)
async def test_mcp_lines_keep_values_without_control_characters(
    controllable_server: ControllableServer, row: str
) -> None:
    """T4, control: values without a control character are logged exactly as before (BR-028).

    ``refresh_spaced`` takes structlog 26.1.0's ``repr`` branch: the message
    ``server said no`` is written ``'server said no'`` there, and as it is
    with 24.1.0, before and after BR-028.
    """
    if row == "refresh_denied":
        (line,) = await _refresh_failure(controllable_server, "denied")
        assert line.endswith(" error_message=denied wrapped=MCPError")
    elif row == "refresh_spaced":
        (line,) = await _refresh_failure(controllable_server, "server said no")
        spaced = "server said no"
        written = repr(spaced) if _renderer_writes_with_repr(spaced) else spaced
        assert line.endswith(f" error_message={written} wrapped=MCPError")
    elif row == "overwrite_arabic":
        _, buffer = await _overwrite(controllable_server, "بحث")
        for event in ("mcp.tool_overwrite", "tool overwritten"):
            (line,) = _lines(buffer, event)
            assert _value_after(line, "tool_name") == "بحث"
    else:
        line = await _hook_line(controllable_server, "boom", "failed")
        assert _value_after(line, "tool_name") == "boom"


def test_registry_overwrite_logs_a_non_string_name_as_before() -> None:
    """T5, control: a tool whose ``name`` is the int 5, registered twice, still returns and is logged as ``tool_name=5`` (BR-028, decision D4).

    ``Tool`` is a runtime-checkable protocol: its ``isinstance`` check tests
    that ``name`` is present, not its type, so such a tool registers.
    ``Registry.register`` escapes a ``str`` name and passes any other value
    as it is; the escape would raise ``AttributeError`` on an int.
    """
    registry = Registry()
    first, second = _Tool(5), _Tool(5)
    registry.register(first)  # type: ignore[arg-type]

    with configured_structlog() as buffer:
        registry.register(second)  # type: ignore[arg-type]

    assert registry.list() == [second]
    (line,) = _lines(buffer, "tool overwritten")
    assert line.endswith(" tool_name=5")


@pytest.mark.parametrize("row", ["registry", "mcp_overwrite", "refresh_failed", "hook_failed"])
async def test_mcp_and_registry_lines_write_a_surrogate_code_point_escaped(
    controllable_server: ControllableServer, row: str
) -> None:
    """T6: a surrogate code point in a registry or MCP log value is written as BR-024's escape (BR-028).

    The value goes through ``escape_for_log``, which runs
    ``escape_surrogates`` first. Before BR-028 these lines passed the code
    point as it was, which a strict UTF-8 stream cannot encode (see the
    module docstring). ``configured_structlog`` writes to a ``StringIO``, so
    the value itself is asserted.
    """
    name, name_logged = "find" + chr(0xD800), "find\\ud800"
    if row == "registry":
        registry = Registry()
        registry.register(_Tool(name))  # type: ignore[arg-type]
        with configured_structlog() as buffer:
            registry.register(_Tool(name))  # type: ignore[arg-type]
        (line,) = _lines(buffer, "tool overwritten")
        assert _value_after(line, "tool_name") == name_logged
    elif row == "mcp_overwrite":
        _, buffer = await _overwrite(controllable_server, name)
        for event in ("mcp.tool_overwrite", "tool overwritten"):
            (line,) = _lines(buffer, event)
            assert _value_after(line, "tool_name") == name_logged
    elif row == "refresh_failed":
        (line,) = await _refresh_failure(controllable_server, "denied" + chr(0xD800))
        assert line.endswith(" error_message=denied\\ud800 wrapped=MCPError")
    else:
        line = await _hook_line(controllable_server, name, "failed")
        assert _value_after(line, "tool_name") == name_logged
    line.encode("utf-8")  # the escaped line encodes
