"""Unit tests for :class:`fifty_agent_sdk.audit.sql.SqlAuditSink`.

Runs against an in-memory aiosqlite engine. Covers the
:class:`fifty_agent_sdk.audit.protocol.AuditSink` contract plus the SQL-specific
commitments from BR-011:

* Round-trip preservation of every :class:`AuditEvent` field.
* Nested ``payload`` survives the JSON column.
* ``user_id=None`` round-trips as ``NULL``.
* Backend failures wrap into :class:`StateStoreError` with the documented
  context shape.
* The ``audit_metadata`` symbol exposes ``agent_audit_log`` with the
  documented columns.
* Constructor accepts both a URL string (engine owned) and an
  :class:`AsyncEngine` (engine NOT owned; not disposed on
  :meth:`SqlAuditSink.aclose`).

Engine fixture
    SQLite ``:memory:`` is connection-scoped. The fixture uses
    :class:`sqlalchemy.pool.StaticPool` so every session opened by the
    sink sees the same underlying connection (and therefore the same
    schema and rows). A fresh engine is built per test for isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import AuditEvent, AuditSink, StateStoreError
from fifty_agent_sdk.audit.sql import AgentAuditLog, SqlAuditSink, audit_metadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """Yield a fresh in-memory aiosqlite engine with the audit schema created.

    ``StaticPool`` + ``check_same_thread=False`` pin a single underlying
    SQLite connection so all sessions share the same in-memory database
    state. The schema is created from :data:`audit_metadata` (the audit
    schema), NOT the conversation-state metadata.
    """
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(audit_metadata.create_all)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def sink(engine: AsyncEngine) -> AsyncIterator[SqlAuditSink]:
    """A :class:`SqlAuditSink` over the in-memory engine."""
    s = SqlAuditSink(engine)
    try:
        yield s
    finally:
        await s.aclose()  # no-op on caller-owned engine; documents the contract


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_record_round_trips_all_fields(sink: SqlAuditSink, engine: AsyncEngine) -> None:
    """Every :class:`AuditEvent` field survives a record + read-back."""
    ts = datetime.now(UTC)
    event = AuditEvent(
        session_id="s1",
        user_id="u-9",
        timestamp=ts,
        event_type="tool_invocation",
        payload={"tool_name": "search", "outcome": "ok"},
    )
    await sink.record(event)

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(
                    AgentAuditLog.session_id,
                    AgentAuditLog.user_id,
                    AgentAuditLog.event_type,
                    AgentAuditLog.timestamp,
                    AgentAuditLog.payload,
                )
            )
        ).one()
    assert row.session_id == "s1"
    assert row.user_id == "u-9"
    assert row.event_type == "tool_invocation"
    assert row.payload == {"tool_name": "search", "outcome": "ok"}
    # Timestamps survive the round-trip; SQLite normalises tz-aware values.
    assert row.timestamp.replace(tzinfo=None) == ts.replace(tzinfo=None)


async def test_record_nested_payload_round_trips(sink: SqlAuditSink, engine: AsyncEngine) -> None:
    """A ``payload`` with nested dict/list survives the JSON column."""
    payload = {
        "args": {"query": "weather", "limits": [1, 2, 3]},
        "meta": {"nested": {"deep": True}},
    }
    await sink.record(
        AuditEvent(
            session_id="s1",
            timestamp=datetime.now(UTC),
            event_type="tool_invocation",
            payload=payload,
        )
    )
    async with engine.connect() as conn:
        stored = (await conn.execute(select(AgentAuditLog.payload))).scalar_one()
    assert stored == payload


async def test_record_user_id_none_round_trips_as_null(
    sink: SqlAuditSink, engine: AsyncEngine
) -> None:
    """A ``None`` ``user_id`` is persisted as SQL ``NULL``."""
    await sink.record(
        AuditEvent(
            session_id="s1",
            timestamp=datetime.now(UTC),
            event_type="error",
        )
    )
    async with engine.connect() as conn:
        stored = (await conn.execute(select(AgentAuditLog.user_id))).scalar_one()
    assert stored is None


async def test_record_appends_rows(sink: SqlAuditSink, engine: AsyncEngine) -> None:
    """Multiple ``record`` calls append distinct rows (append-only)."""
    for index in range(3):
        await sink.record(
            AuditEvent(
                session_id="s1",
                timestamp=datetime.now(UTC),
                event_type="session_start",
                payload={"n": index},
            )
        )
    async with engine.connect() as conn:
        payloads = (
            (await conn.execute(select(AgentAuditLog.payload).order_by(AgentAuditLog.id.asc())))
            .scalars()
            .all()
        )
    assert len(payloads) == 3
    assert [p["n"] for p in payloads] == [0, 1, 2]


async def test_recorded_at_is_populated(sink: SqlAuditSink, engine: AsyncEngine) -> None:
    """The server-default ``recorded_at`` column is populated on insert."""
    await sink.record(
        AuditEvent(
            session_id="s1",
            timestamp=datetime.now(UTC),
            event_type="session_start",
        )
    )
    async with engine.connect() as conn:
        recorded_at = (await conn.execute(select(AgentAuditLog.recorded_at))).scalar_one()
    assert recorded_at is not None


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


async def test_backend_failure_is_wrapped(engine: AsyncEngine) -> None:
    """Disposing the engine then recording surfaces a wrapped error."""
    sink = SqlAuditSink(engine)
    await engine.dispose()

    with pytest.raises(StateStoreError) as exc_info:
        await sink.record(
            AuditEvent(
                session_id="s-err",
                timestamp=datetime.now(UTC),
                event_type="error",
            )
        )
    err = exc_info.value
    assert err.context["session_id"] == "s-err"
    assert err.context["operation"] == "record"
    assert isinstance(err.context["wrapped"], str)
    assert err.context["wrapped"] != ""
    assert err.__cause__ is not None


# ---------------------------------------------------------------------------
# Protocol conformance / constructor surface
# ---------------------------------------------------------------------------


async def test_sql_sink_satisfies_audit_sink_protocol(
    sink: SqlAuditSink,
) -> None:
    """:class:`SqlAuditSink` matches the :class:`AuditSink` runtime protocol."""
    assert isinstance(sink, AuditSink)


async def test_constructor_accepts_url_string() -> None:
    """Constructing from a URL string creates an internal engine."""
    s = SqlAuditSink("sqlite+aiosqlite:///:memory:")
    try:
        assert s._owns_engine is True
        async with s._engine.begin() as conn:
            row = (await conn.execute(text("SELECT 1"))).scalar_one()
        assert row == 1
    finally:
        await s.aclose()


async def test_constructor_accepts_async_engine(engine: AsyncEngine) -> None:
    """Constructing from an explicit engine does NOT take ownership."""
    s = SqlAuditSink(engine)
    assert s._owns_engine is False
    await s.aclose()
    # aclose was a no-op — the engine is still usable.
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT 1"))).scalar_one()
    assert row == 1


async def test_consumer_owned_engine_is_not_disposed_on_aclose(
    engine: AsyncEngine,
) -> None:
    """:meth:`SqlAuditSink.aclose` on a consumer-passed engine is a no-op."""
    s = SqlAuditSink(engine)
    await s.aclose()
    async with engine.connect() as conn:
        row = (await conn.execute(text("SELECT 1"))).scalar_one()
    assert row == 1


# ---------------------------------------------------------------------------
# Metadata / schema introspection
# ---------------------------------------------------------------------------


def test_metadata_exposes_audit_table() -> None:
    """``audit_metadata`` lists the audit table for Alembic autogenerate."""
    assert "agent_audit_log" in audit_metadata.tables


def test_metadata_columns_match_schema() -> None:
    """Column names align with the brief's documented audit schema."""
    table = audit_metadata.tables["agent_audit_log"]
    assert {c.name for c in table.columns} == {
        "id",
        "session_id",
        "user_id",
        "event_type",
        "timestamp",
        "recorded_at",
        "payload",
    }


def test_audit_metadata_is_separate_from_state_metadata() -> None:
    """The audit schema does NOT share metadata with the state schema."""
    from fifty_agent_sdk import sql_metadata

    assert audit_metadata is not sql_metadata
    # The audit table is absent from the state metadata and vice versa.
    assert "agent_audit_log" not in sql_metadata.tables
    assert "agent_sessions" not in audit_metadata.tables


async def test_metadata_create_all_is_idempotent(engine: AsyncEngine) -> None:
    """Running ``create_all`` twice on the same engine is a clean no-op."""
    async with engine.begin() as conn:
        await conn.run_sync(audit_metadata.create_all)
    sink = SqlAuditSink(engine)
    await sink.record(
        AuditEvent(
            session_id="s1",
            timestamp=datetime.now(UTC),
            event_type="session_start",
        )
    )
    async with engine.connect() as conn:
        count = (await conn.execute(select(AgentAuditLog.id))).all()
    assert len(count) == 1
