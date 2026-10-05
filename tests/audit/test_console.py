"""Unit tests for :class:`fifty_agent_sdk.audit.console.ConsoleAuditSink`.

Uses :func:`structlog.testing.capture_logs` to assert that
:meth:`ConsoleAuditSink.record` emits exactly one structured ``INFO``
record carrying every :class:`AuditEvent` field.

Since BR-027 the event's own time is logged as ``event_timestamp``, not
``timestamp``: ``timestamp`` is the key structlog's ``TimeStamper`` writes by
default, and structlog's default configuration, which the SDK does not
change, includes one, so the line used to carry the log time instead. A1,
A1b and A3 run the real sink under structlog's default configuration and
under explicit ``TimeStamper`` and ``MaybeTimeStamper`` chains
(``tests/structlog_chains.py``), and under the stdlib recipes with a
``TimeStamper(fmt="iso")`` ahead of them (``tests/stdlib_routing.py``).
Measured with structlog 26.1.0 on CPython 3.11.15, 3.13.2 and 3.14.3, and
with structlog 24.1.0 on the same interpreters (BR-027 evidence, P2 and P3).
On the pre-BR-027 tree A1, A1b and A4 fail; A3 and A5 pass there, as
controls.

Not pinned: structlog releases other than the one installed, and a key a
host chooses for its ``TimeStamper`` (``TimeStamper(key=...)``): one that
chooses ``event_timestamp`` loses the event's time, which it kept under
``timestamp`` before BR-027 (BR-027 evidence, P3 row k).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from structlog.processors import TimeStamper

from fifty_agent_sdk import AuditEvent, AuditSink, ConsoleAuditSink
from tests.stdlib_routing import (
    EXTRA_RECIPES,
    record_field,
    records_for,
    stdlib_routed_structlog,
)
from tests.structlog_chains import CHAINS, audit_json_lines, configured_structlog, console_lines

# The event time BR-026's sentinel and BR-027's probes used; its isoformat() is fixed.
EVENT_TIME = datetime(2000, 1, 2, 3, 4, 5, tzinfo=UTC)
EVENT_ISO = "2000-01-02T03:04:05+00:00"


def _event() -> AuditEvent:
    return AuditEvent(
        session_id="s1",
        user_id="u-7",
        timestamp=EVENT_TIME,
        event_type="tool_invocation",
        payload={"tool_name": "search", "outcome": "ok"},
    )


async def test_record_emits_single_info_event() -> None:
    """One ``record`` call produces exactly one ``INFO`` structlog entry."""
    sink = ConsoleAuditSink()
    event = AuditEvent(
        session_id="s1",
        timestamp=datetime.now(UTC),
        event_type="session_start",
        payload={"run_id": "abc"},
    )
    with structlog.testing.capture_logs() as logs:
        await sink.record(event)

    entries = [e for e in logs if e.get("event") == "audit.event"]
    assert len(entries) == 1
    assert entries[0]["log_level"] == "info"


async def test_record_spreads_all_event_fields() -> None:
    """A4: the structlog record carries every :class:`AuditEvent` field, the time as ``event_timestamp`` (BR-027).

    The exact key set also fails a line that still passes ``timestamp``
    beside ``event_timestamp``.
    """
    sink = ConsoleAuditSink()
    ts = datetime.now(UTC)
    event = AuditEvent(
        session_id="sess-xyz",
        user_id="u-7",
        timestamp=ts,
        event_type="tool_invocation",
        payload={"tool_name": "search", "outcome": "ok"},
    )
    with structlog.testing.capture_logs() as logs:
        await sink.record(event)

    entry = next(e for e in logs if e.get("event") == "audit.event")
    assert entry["session_id"] == "sess-xyz"
    assert entry["user_id"] == "u-7"
    assert entry["event_type"] == "tool_invocation"
    assert entry["event_timestamp"] == ts.isoformat()
    assert entry["payload"] == {"tool_name": "search", "outcome": "ok"}
    assert set(entry) == {
        "event",
        "log_level",
        "session_id",
        "user_id",
        "event_type",
        "event_timestamp",
        "payload",
    }


async def test_record_handles_none_user_id() -> None:
    """A ``None`` ``user_id`` is logged as ``None``, not omitted."""
    sink = ConsoleAuditSink()
    event = AuditEvent(session_id="s1", timestamp=datetime.now(UTC), event_type="error")
    with structlog.testing.capture_logs() as logs:
        await sink.record(event)

    entry = next(e for e in logs if e.get("event") == "audit.event")
    assert entry["user_id"] is None


def test_console_sink_satisfies_audit_sink_protocol() -> None:
    """:class:`ConsoleAuditSink` matches the :class:`AuditSink` protocol."""
    assert isinstance(ConsoleAuditSink(), AuditSink)


def test_console_sink_exported_from_top_level() -> None:
    """:class:`ConsoleAuditSink` is importable from the package root."""
    from fifty_agent_sdk import ConsoleAuditSink as _ConsoleAuditSink

    assert _ConsoleAuditSink is ConsoleAuditSink


# --- BR-027: the event's own time under a host's TimeStamper -----------------------------

_FIELDS = ("session_id", "user_id", "event_type", "event_timestamp", "timestamp", "payload")


async def _native_line(chain: str) -> dict[str, Any]:
    """The one ``audit.event`` line a JSON chain wrote, parsed."""
    with configured_structlog(CHAINS[chain]) as buffer:
        await ConsoleAuditSink().record(_event())
    (line,) = audit_json_lines(buffer)
    return line


async def _stdlib_record(recipe: str) -> dict[str, Any]:
    """The one ``audit.event`` record under ``recipe``, with ``TimeStamper(fmt="iso")`` ahead of it.

    A field the record lacks raises ``KeyError`` (``record_field``).
    """
    with stdlib_routed_structlog(recipe, logging.INFO) as records:
        structlog.configure(
            processors=[TimeStamper(fmt="iso"), *structlog.get_config()["processors"]]
        )
        await ConsoleAuditSink().record(_event())
    (record,) = records_for(records, "audit.event")
    return {key: record_field(record, key) for key in _FIELDS}


@pytest.mark.parametrize(
    "row",
    [
        *(pytest.param(chain, id=chain) for chain in CHAINS),
        *EXTRA_RECIPES,
        pytest.param("wrap_for_formatter", id="wrap_for_formatter"),
    ],
)
async def test_audit_line_keeps_the_event_time_under_a_host_timestamper(row: str) -> None:
    """A1: under a timestamping processor the line carries the event's time as ``event_timestamp`` (BR-027, AC-1).

    Rows: structlog's default processors with a JSON renderer (their
    ``TimeStamper`` is local time to the second), ``TimeStamper(fmt="iso")``,
    ``TimeStamper()`` (a UNIX time), ``MaybeTimeStamper(fmt="iso")``, and the
    three stdlib recipes after a ``TimeStamper(fmt="iso")``. Before BR-027
    the line logged the event's time as ``timestamp``: in every row but
    ``maybe_timestamper`` the processor then wrote the log time there and the
    event's time was lost; under ``MaybeTimeStamper``, which adds a time only
    when the event has none, ``timestamp`` kept the event's time, and it now
    holds the log time (BR-027 evidence, P2 and P3).
    """
    fields = await (_native_line(row) if row in CHAINS else _stdlib_record(row))

    assert fields["event_timestamp"] == EVENT_ISO
    assert fields["timestamp"] != EVENT_ISO
    assert fields["session_id"] == "s1"
    assert fields["user_id"] == "u-7"
    assert fields["event_type"] == "tool_invocation"
    assert fields["payload"] == {"tool_name": "search", "outcome": "ok"}


async def test_audit_line_shows_the_event_time_under_structlog_default_configuration() -> None:
    """A1b: structlog's default configuration, ``ConsoleRenderer`` included, shows the event's time (BR-027).

    The SDK never configures structlog. The default chain's ``TimeStamper``
    writes the log time, in local time to the second, as the line's first
    column; before BR-027 the event's time appeared nowhere on the line
    (BR-027 evidence, P2 row a). ``ConsoleRenderer`` writes the ISO string as
    it is: it holds no space, tab, ``=``, quote or line break (by reading
    structlog 26.1.0; measured in P3 row a).
    """
    with configured_structlog() as buffer:
        await ConsoleAuditSink().record(_event())

    (line,) = console_lines(buffer, "audit.event")
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", line), line
    assert f" event_timestamp={EVENT_ISO}" in line, line
    assert line.count(EVENT_ISO) == 1, line


async def test_audit_line_keeps_its_other_fields_under_a_timestamper() -> None:
    """A3 (control): the line's event, level and other fields are what they were; it passes before BR-027 too."""
    with configured_structlog(CHAINS["timestamper_iso"]) as buffer:
        await ConsoleAuditSink().record(_event())

    (line,) = audit_json_lines(buffer)
    assert line["event"] == "audit.event"
    assert line["level"] == "info"
    assert line["session_id"] == "s1"
    assert line["user_id"] == "u-7"
    assert line["event_type"] == "tool_invocation"
    assert line["payload"] == {"tool_name": "search", "outcome": "ok"}
    assert "timestamp" in line


async def test_configured_structlog_restores_the_previous_configuration() -> None:
    """A5: after ``configured_structlog`` structlog's configuration is as before, and an SDK line reaches ``capture_logs``.

    Modelled on BR-026's S4. The sink logs once inside the helper (so a cached
    logger would keep the helper's chain and buffer) and once after it, under
    ``structlog.testing.capture_logs()``.
    """
    was_configured = structlog.is_configured()
    config_before = structlog.get_config()
    sink = ConsoleAuditSink()

    with configured_structlog(CHAINS["timestamper_iso"]) as buffer:
        await sink.record(_event())
    assert len(buffer.getvalue().splitlines()) == 1

    assert structlog.is_configured() is was_configured
    config_after = structlog.get_config()
    assert type(config_after["logger_factory"]) is type(config_before["logger_factory"])
    assert config_after["wrapper_class"] is config_before["wrapper_class"]
    assert config_after["processors"] == config_before["processors"]
    assert config_after["cache_logger_on_first_use"] == config_before["cache_logger_on_first_use"]

    with structlog.testing.capture_logs() as logs:
        await sink.record(_event())
    assert [entry["event"] for entry in logs] == ["audit.event"]
    assert len(buffer.getvalue().splitlines()) == 1
