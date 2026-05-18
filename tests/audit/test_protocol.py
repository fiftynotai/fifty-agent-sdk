"""Unit tests for :mod:`agent_sdk.audit.protocol`.

Covers the :class:`AuditEvent` Pydantic model (required/optional fields,
``extra="forbid"``) and the :class:`AuditSink` runtime-checkable protocol.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_sdk import AuditEvent, AuditSink

# ---------------------------------------------------------------------------
# AuditEvent model
# ---------------------------------------------------------------------------


def test_audit_event_constructs_with_required_fields() -> None:
    """The three required fields are enough to build an :class:`AuditEvent`."""
    ts = datetime.now(UTC)
    event = AuditEvent(session_id="s1", timestamp=ts, event_type="session_start")
    assert event.session_id == "s1"
    assert event.timestamp == ts
    assert event.event_type == "session_start"


def test_audit_event_user_id_defaults_to_none() -> None:
    """``user_id`` is optional and defaults to ``None``."""
    event = AuditEvent(
        session_id="s1", timestamp=datetime.now(UTC), event_type="error"
    )
    assert event.user_id is None


def test_audit_event_payload_defaults_to_empty_dict() -> None:
    """``payload`` is optional and defaults to a fresh empty dict."""
    event = AuditEvent(
        session_id="s1", timestamp=datetime.now(UTC), event_type="error"
    )
    assert event.payload == {}


def test_audit_event_payload_default_is_not_shared() -> None:
    """Each event gets its own ``payload`` dict (``default_factory``)."""
    a = AuditEvent(
        session_id="s1", timestamp=datetime.now(UTC), event_type="error"
    )
    b = AuditEvent(
        session_id="s2", timestamp=datetime.now(UTC), event_type="error"
    )
    a.payload["k"] = "v"
    assert b.payload == {}


def test_audit_event_accepts_explicit_fields() -> None:
    """All fields round-trip when supplied explicitly."""
    ts = datetime.now(UTC)
    event = AuditEvent(
        session_id="s1",
        user_id="u-42",
        timestamp=ts,
        event_type="tool_invocation",
        payload={"tool_name": "search", "outcome": "ok"},
    )
    assert event.user_id == "u-42"
    assert event.payload == {"tool_name": "search", "outcome": "ok"}


def test_audit_event_rejects_unknown_field() -> None:
    """``extra="forbid"`` rejects an unknown field at construction."""
    with pytest.raises(ValidationError):
        AuditEvent(
            session_id="s1",
            timestamp=datetime.now(UTC),
            event_type="error",
            not_a_real_field="boom",  # type: ignore[call-arg]
        )


def test_audit_event_requires_session_id() -> None:
    """``session_id`` is required — omitting it is a validation error."""
    with pytest.raises(ValidationError):
        AuditEvent(  # type: ignore[call-arg]
            timestamp=datetime.now(UTC),
            event_type="error",
        )


# ---------------------------------------------------------------------------
# AuditSink protocol conformance
# ---------------------------------------------------------------------------


def test_minimal_class_satisfies_audit_sink_protocol() -> None:
    """A class with an ``async record`` matches :class:`AuditSink`."""

    class MinimalSink:
        async def record(self, event: AuditEvent) -> None:
            return None

    sink = MinimalSink()
    assert isinstance(sink, AuditSink)


def test_class_missing_record_fails_protocol_check() -> None:
    """A class without ``record`` does NOT match :class:`AuditSink`."""

    class NoRecord:
        async def something_else(self) -> None:
            return None

    assert not isinstance(NoRecord(), AuditSink)


def test_protocol_is_exported_from_top_level() -> None:
    """:class:`AuditSink` and :class:`AuditEvent` are importable from the root."""
    from agent_sdk import AuditEvent as _AuditEvent
    from agent_sdk import AuditSink as _AuditSink

    assert _AuditEvent is AuditEvent
    assert _AuditSink is AuditSink
