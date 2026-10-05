"""A real ``AgentLoop`` and ``AgentRunner`` under structlog routed through stdlib ``logging`` (BR-026, AC-2).

Before BR-026 the loop's ``tool_invoked`` debug line passed ``name=``.
With structlog routed through stdlib by ``render_to_log_kwargs`` or
``render_to_log_args_and_kwargs`` and DEBUG enabled for the SDK's loggers,
stdlib's ``Logger.makeRecord`` raised ``KeyError: "Attempt to overwrite
'name' in LogRecord"`` at that line, and the ``KeyError`` left
``AgentLoop.run()`` at the run's first dispatched call after one request,
with no ``ErrorEvent`` or ``FinalEvent``: for a native call to a registered
or an unregistered tool, a native batch, a ``ToolMode.JSON`` and a
``ToolMode.PROSE`` call, and a call parsed by ``JsonModeParser`` without a
tool mode (BR-026 evidence, P2 and P2b, structlog 26.1.0 on
CPython 3.11.15, 3.13.2 and 3.14.3). The line now passes ``tool_name=``.

Each test runs inside ``tests.stdlib_routing.stdlib_routed_structlog``,
which restores structlog's and stdlib's configuration afterwards (S4). The
``wrap_for_formatter`` rows are the control recipe: it never passed the
SDK's keys to ``makeRecord``, and runs under it completed before BR-026
too (P2); on the pre-BR-026 tree those rows still fail, at the read of
``tool_name``, which that tree logged as ``name``. S2 is the INFO control:
``tool_invoked`` is filtered there, so those runs completed before BR-026,
and S2 passes on that tree.

What these do NOT pin: keys a host's own processors add; a structlog
release other than the one installed (the measured scope is in the
evidence file); every SDK log line (the sweep in ``tests/test_log_keys.py``
checks every logging call it can read (its docstring lists what it does not
see), and S3 adds the Runner's and audit sink's lines).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentEvent,
    AgentLoop,
    AgentRunner,
    AuditEvent,
    ChatMessage,
    ConsoleAuditSink,
    ErrorEvent,
    FinalEvent,
    JsonModeParser,
    MemoryStateStore,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    ToolStartedEvent,
)
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response
from tests.stdlib_routing import (
    ALL_RECIPES,
    EXTRA_RECIPES,
    SDK_LOGGER,
    record_field,
    record_keys,
    records_for,
    stdlib_routed_structlog,
)

_USER = [ChatMessage(role="user", content="q")]

# Each row: how the loop is built, and the tool names the model calls, in order.
_ROWS: dict[str, tuple[str, list[str]]] = {
    "native_single_registered": ("native", ["lookup"]),
    "native_single_unregistered": ("native", ["find"]),
    "native_batch": ("native", ["lookup", "find"]),
    "json_mode_single": ("json", ["lookup"]),
    "prose_mode_single": ("prose", ["lookup"]),
    "legacy_json_single": ("legacy", ["lookup"]),
}


def _envelope(action: str, *, tool_name: str | None = None, answer: str | None = None) -> str:
    return json.dumps(
        {
            "thought": "t",
            "action": action,
            "tool_name": tool_name,
            "tool_args": {"q": "x"} if action == "tool" else None,
            "answer": answer,
        }
    )


def _loop(row: str) -> tuple[AgentLoop, list[str]]:
    """A loop whose model makes ``row``'s tool call(s) and then answers ``done``; and the names it calls."""
    registry = Registry()
    registry.register(FakeTool("lookup"))
    kind, names = _ROWS[row]
    kwargs: dict[str, Any]
    if kind == "native":
        replies = [
            make_multi_tool_response([(name, {"q": "x"}) for name in names]),
            make_response("done"),
        ]
        kwargs = {"tool_mode": ToolMode.NATIVE}
    elif kind == "prose":
        replies = [
            make_response('Thought: t\nAction: lookup\nAction Input: {"q": "x"}'),
            make_response("Thought: t\nFinal Answer: done"),
        ]
        kwargs = {"tool_mode": ToolMode.PROSE}
    else:
        replies = [
            make_response(_envelope("tool", tool_name="lookup")),
            make_response(_envelope("final", answer="done")),
        ]
        if kind == "json":
            kwargs = {"tool_mode": ToolMode.JSON}
        else:
            kwargs = {"parser": JsonModeParser(), "output_format": JSON_MODE_OUTPUT_FORMAT}
    loop = AgentLoop(
        llm=FakeLLMClient(list(replies)),
        registry=registry,
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        **kwargs,
    )
    return loop, names


def _ends_on_done(events: list[AgentEvent]) -> None:
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].text == "done"
    assert not any(isinstance(event, ErrorEvent) for event in events)


@pytest.mark.parametrize("row", list(_ROWS))
@pytest.mark.parametrize("recipe", ALL_RECIPES)
async def test_tool_calls_complete_under_stdlib_routing_at_debug(recipe: str, row: str) -> None:
    """S1: with DEBUG enabled, each dispatched call logs ``tool_invoked`` under ``tool_name`` and the run ends on ``done`` (BR-026, AC-2).

    Before BR-026 the ``render_to_log_kwargs`` and
    ``render_to_log_args_and_kwargs`` rows raised ``KeyError`` for
    ``'name'`` out of ``run()`` (P2). The record's ``name`` stays the
    logger's name; under the control recipe the event dict carries
    ``tool_name`` and no ``name``.
    """
    loop, names = _loop(row)

    with stdlib_routed_structlog(recipe, logging.DEBUG) as records:
        events = [event async for event in loop.run(list(_USER))]

    _ends_on_done(events)
    invoked = records_for(records, "tool_invoked")
    assert [record_field(record, "tool_name") for record in invoked] == names
    started = [event.call_id for event in events if isinstance(event, ToolStartedEvent)]
    assert [record_field(record, "call_id") for record in invoked] == started
    assert all(record.name == "fifty_agent_sdk.loop" for record in invoked)
    assert all(record.levelno == logging.DEBUG for record in invoked)
    if recipe == "wrap_for_formatter":
        assert all(
            record_keys(record) == {"tool_name", "call_id", "run_id", "level"} for record in invoked
        )


@pytest.mark.parametrize("row", ["native_single_registered", "native_batch"])
@pytest.mark.parametrize("recipe", EXTRA_RECIPES)
async def test_tool_calls_complete_under_stdlib_routing_at_info(recipe: str, row: str) -> None:
    """S2 (control): at INFO the ``tool_invoked`` line is filtered by stdlib, so these runs completed before BR-026 too.

    The INFO lifecycle lines still reach the handler, so the routing is live.
    """
    loop, _ = _loop(row)

    with stdlib_routed_structlog(recipe, logging.INFO) as records:
        events = [event async for event in loop.run(list(_USER))]

    _ends_on_done(events)
    assert records_for(records, "tool_invoked") == []
    assert len(records_for(records, "agent_loop_started")) == 1


@pytest.mark.parametrize("recipe", ALL_RECIPES)
async def test_runner_turn_completes_under_stdlib_routing_at_debug(recipe: str) -> None:
    """S3: an ``AgentRunner`` turn with a native tool call and ``ConsoleAuditSink`` completes at DEBUG (BR-026, AC-2).

    Runtime breadth beyond the loop: the Runner's and the audit sink's
    lines go through the same recipe. Before BR-026 the extra-recipe rows
    raised at the loop's ``tool_invoked`` line.
    """
    loop, _ = _loop("native_single_registered")
    runner = AgentRunner(loop=loop, state=MemoryStateStore(), audit=ConsoleAuditSink())

    with stdlib_routed_structlog(recipe, logging.DEBUG) as records:
        events = [event async for event in runner.run("s1", "q")]

    _ends_on_done(events)
    for event in ("runner.run_started", "audit.event", "tool_invoked", "runner.run_completed"):
        assert records_for(records, event), event
    assert {record.name for record in records_for(records, "audit.event")} == {
        "fifty_agent_sdk.audit"
    }


def _logging_state() -> tuple[Any, ...]:
    sdk_logger = logging.getLogger(SDK_LOGGER)
    root = logging.getLogger()
    return (
        sdk_logger.level,
        list(sdk_logger.handlers),
        sdk_logger.propagate,
        root.level,
        list(root.handlers),
        logging.raiseExceptions,
    )


async def test_stdlib_routing_helper_restores_structlog_and_logging_state() -> None:
    """S4: after the helper, stdlib's and structlog's configuration are as before, and an SDK line reaches ``capture_logs``.

    ``ConsoleAuditSink`` logs once inside the helper (so a cached logger
    would keep the stdlib chain) and once after it, under
    ``structlog.testing.capture_logs()``. Its ``audit.event`` line passes
    no reserved key before or after BR-026, so this restore check does not
    depend on the rename.
    """
    before = _logging_state()
    was_configured = structlog.is_configured()
    config_before = structlog.get_config()
    sink = ConsoleAuditSink()
    event = AuditEvent(session_id="s1", timestamp=datetime.now(UTC), event_type="probe")

    with stdlib_routed_structlog("render_to_log_kwargs", logging.DEBUG) as records:
        await sink.record(event)
    assert [record.msg for record in records] == ["audit.event"]

    assert _logging_state() == before
    assert structlog.is_configured() is was_configured
    config_after = structlog.get_config()
    assert type(config_after["logger_factory"]) is type(config_before["logger_factory"])
    assert config_after["wrapper_class"] is config_before["wrapper_class"]
    assert config_after["processors"] == config_before["processors"]
    assert config_after["cache_logger_on_first_use"] == config_before["cache_logger_on_first_use"]

    with structlog.testing.capture_logs() as logs:
        await sink.record(event)
    assert [entry["event"] for entry in logs] == ["audit.event"]
    assert len(records) == 1
