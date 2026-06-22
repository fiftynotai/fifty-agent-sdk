"""Unit tests for :class:`fifty_agent_sdk.audit.console.ConsoleAuditSink`.

Uses :func:`structlog.testing.capture_logs` to assert that
:meth:`ConsoleAuditSink.record` emits exactly one structured ``INFO``
record carrying every :class:`AuditEvent` field.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from fifty_agent_sdk import AuditEvent, AuditSink, ConsoleAuditSink


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
    """The structlog record carries every :class:`AuditEvent` field."""
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
    assert entry["timestamp"] == ts.isoformat()
    assert entry["payload"] == {"tool_name": "search", "outcome": "ok"}


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
