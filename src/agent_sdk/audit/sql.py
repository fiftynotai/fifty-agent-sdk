"""SQLAlchemy 2.0 asyncio implementation of :class:`AuditSink`.

:class:`SqlAuditSink` is the SDK's durable audit-trail backend — a
queryable, append-only record of every agent action. It targets the
SQLAlchemy 2.0 asyncio API and is tested against SQLite (via ``aiosqlite``)
and Postgres (via ``asyncpg`` or ``psycopg``).

Extras requirement
    This module requires the optional ``sql`` extra::

        pip install 'agent-sdk[sql]'

    Importing :mod:`agent_sdk` itself does NOT pull SQLAlchemy. The audit
    SQL surface is re-exported lazily from :mod:`agent_sdk.audit` and the
    package root via module-level ``__getattr__``; first access triggers
    this module's import, and a missing dependency surfaces as a clear
    :class:`ImportError` referencing the extras line above.

Schema lifecycle
    The SDK ships the table definition as a declarative ORM model but does
    NOT own migrations. Consumers register :data:`audit_metadata` with
    their own Alembic environment::

        # alembic/env.py
        from agent_sdk import audit_metadata
        target_metadata = audit_metadata

    Then::

        alembic revision --autogenerate -m "agent_sdk audit schema"

    The audit table uses a :class:`sqlalchemy.MetaData` object DISTINCT
    from :data:`agent_sdk.sql_metadata` (the conversation-state schema), so
    a consumer who wants only one of the two does not autogenerate both.

Engine ownership
    The constructor accepts EITHER an :class:`AsyncEngine` or a connection
    URL string. When a URL is provided, the sink creates its own engine and
    tracks ``self._owns_engine = True``; :meth:`aclose` disposes it. When an
    externally-built engine is passed, the sink DOES NOT dispose it on
    close — the caller owns its lifecycle. :class:`SqlAuditSink` and
    :class:`agent_sdk.state.sql.SqlStateStore` are independent
    collaborators; a consumer wanting shared infrastructure passes the same
    :class:`AsyncEngine` instance to both constructors.

Error wrapping contract
    :meth:`record` wraps :class:`sqlalchemy.exc.SQLAlchemyError` (the
    SQLAlchemy base class) into :class:`agent_sdk.errors.StateStoreError`
    with:

    * ``message``: ``"SqlAuditSink.record failed for session_id=<id>"``
    * ``context["session_id"]``: the input session id (echoed for log
      correlation)
    * ``context["wrapped"]``: the underlying exception's class name
      (e.g., ``"OperationalError"``)
    * ``context["operation"]``: always ``"record"``
    * ``__cause__``: the original exception, via ``raise ... from exc``

    :class:`StateStoreError` is reused deliberately rather than introducing
    a dedicated audit error type: the audit sink is a state-store-shaped
    backend boundary, and :class:`AgentRunner` swallows audit errors anyway,
    so the wrapped type is only visible to a consumer calling the sink
    directly. :class:`asyncio.CancelledError` propagates untouched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

import structlog

try:
    from sqlalchemy import (
        BigInteger,
        DateTime,
        Index,
        Integer,
        String,
        func,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
    from sqlalchemy.types import JSON
except ImportError as exc:  # pragma: no cover - exercised via importlib in tests
    raise ImportError(
        "agent_sdk.audit.sql requires SQLAlchemy. "
        "Install with: pip install 'agent-sdk[sql]'"
    ) from exc

from agent_sdk.audit.protocol import AuditEvent
from agent_sdk.errors import StateStoreError

_log: Final = structlog.get_logger("agent_sdk.audit")
"""Module-level structured logger.

Bound to the fixed name ``agent_sdk.audit`` (shared with
:mod:`agent_sdk.audit.console`). Successful writes log at ``DEBUG``;
failures are NOT logged here — the wrapped :class:`StateStoreError` carries
the context, and the Runner's ``audit.emit_failed`` ``WARNING`` is the
operator-visible signal.
"""


class Base(DeclarativeBase):
    """Declarative base for the SDK's audit ORM model.

    Module-private on purpose, and deliberately SEPARATE from
    :class:`agent_sdk.state.sql.Base`: the audit table must be registerable
    with Alembic independently of the conversation-state schema, so a
    consumer using only one of the two does not autogenerate both. Consumers
    register :data:`audit_metadata` with their Alembic environment rather
    than subclassing this base.
    """


class AgentAuditLog(Base):
    """One persisted :class:`AuditEvent` row — the audit trail's grain.

    Append-only by contract: the SDK only ever inserts. The pairing of
    :attr:`timestamp` (the action time, written by the Runner) and
    :attr:`recorded_at` (the row-insertion time, server-generated) lets a
    consumer detect drift — an unexpected gap between the two is itself a
    tamper signal.
    """

    __tablename__ = "agent_audit_log"
    __table_args__ = (
        Index(
            "ix_agent_audit_log_session_timestamp",
            "session_id",
            "timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer().with_variant(BigInteger(), "postgresql"),
        primary_key=True,
        autoincrement=True,
    )
    """Surrogate row id; ``BigInteger`` on Postgres, ``Integer`` on SQLite."""

    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    """Opaque session id the action belongs to. Session-scoped audit queries
    are served by the composite ``ix_agent_audit_log_session_timestamp`` index
    (``session_id`` is its leading column), so no separate single-column index
    is declared."""

    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """Optional end-user identifier; ``NULL`` when not supplied."""

    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    """Short tag for the kind of action (e.g. ``"tool_invocation"``)."""

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    """When the action occurred (timezone-aware). Written by the Runner —
    NOT a server default, since the audit record's time must be the action
    time, not the persist time."""

    recorded_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """When the row was persisted (server-side default; timezone-aware).
    Drift between this and :attr:`timestamp` is itself a tamper signal."""

    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
    )
    """Structured event detail. ``JSONB`` on Postgres (indexable, queryable
    with ``->>``), JSON-encoded ``TEXT`` on SQLite — same dialect-parity
    treatment as :attr:`agent_sdk.state.sql.AgentSession.meta`."""


audit_metadata = Base.metadata
"""The :class:`sqlalchemy.MetaData` describing the SDK's audit schema.

Consumers register this with their Alembic environment::

    # alembic/env.py
    from agent_sdk import audit_metadata
    target_metadata = audit_metadata

It is DISTINCT from :data:`agent_sdk.sql_metadata` (the conversation-state
schema) so the two are independently registerable — a consumer using only
one does not autogenerate both. The SDK ships no migration scripts;
consumers run ``alembic revision --autogenerate`` against this object and
review the output before applying.
"""


def _wrap_audit_sink_error(
    exc: SQLAlchemyError,
    *,
    session_id: str,
    operation: str,
) -> StateStoreError:
    """Build the SDK's standard wrap of a SQLAlchemy audit-backend failure.

    Returns the :class:`StateStoreError`; the caller writes
    ``raise _wrap_audit_sink_error(...) from exc`` so the ``__cause__``
    chain is preserved. :class:`StateStoreError` is reused deliberately —
    see the module docstring's "Error wrapping contract" section.
    """
    return StateStoreError(
        f"SqlAuditSink.{operation} failed for session_id={session_id}",
        context={
            "session_id": session_id,
            "wrapped": type(exc).__name__,
            "operation": operation,
        },
    )


class SqlAuditSink:
    """SQLAlchemy-backed implementation of :class:`AuditSink`.

    Satisfies :class:`agent_sdk.audit.protocol.AuditSink` structurally (no
    explicit inheritance needed thanks to ``@runtime_checkable``).

    Concurrency model:
        :meth:`record` is a pure INSERT with a server-generated primary
        key — there is no ``MAX(sequence) + 1`` race, so the per-session
        :class:`asyncio.Lock` machinery of
        :class:`agent_sdk.state.sql.SqlStateStore` is intentionally OMITTED.
        Concurrent :meth:`record` calls from multiple runs are serialized
        only by the database's own write path, which is sufficient for
        independent inserts.

    Engine ownership:
        * Construct with a URL string → the sink creates and owns the
          engine. :meth:`aclose` disposes it.
        * Construct with an :class:`AsyncEngine` → the caller owns the
          engine. :meth:`aclose` is a no-op on the engine.

    Failure mode:
        :meth:`record` wraps :class:`sqlalchemy.exc.SQLAlchemyError` into
        :class:`agent_sdk.errors.StateStoreError` with ``context["wrapped"]``
        carrying the underlying class name. See the module docstring for
        the full contract.
    """

    def __init__(self, engine_or_url: AsyncEngine | str) -> None:
        """Construct a :class:`SqlAuditSink`.

        Args:
            engine_or_url: Either a fully-configured
                :class:`sqlalchemy.ext.asyncio.AsyncEngine` (caller-owned)
                or a connection URL string (e.g.,
                ``"sqlite+aiosqlite:///:memory:"``,
                ``"postgresql+asyncpg://..."``) — in which case the sink
                creates and owns the engine.
        """
        if isinstance(engine_or_url, str):
            self._engine: AsyncEngine = create_async_engine(engine_or_url)
            self._owns_engine: bool = True
        else:
            self._engine = engine_or_url
            self._owns_engine = False
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    async def aclose(self) -> None:
        """Dispose the underlying engine if (and only if) the sink owns it.

        Safe to call multiple times. When the engine was passed in by the
        caller, this is a no-op — the caller's engine lifecycle is not the
        sink's responsibility.
        """
        if self._owns_engine:
            await self._engine.dispose()

    async def record(self, event: AuditEvent) -> None:
        """Insert ``event`` as one append-only row in ``agent_audit_log``.

        Opens a session, begins a transaction, adds an
        :class:`AgentAuditLog` row built from the :class:`AuditEvent`
        fields, and commits.

        Args:
            event: The :class:`AuditEvent` to persist.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name and
                ``context["operation"]`` is ``"record"``.
        """
        try:
            async with self._session_factory() as session, session.begin():
                session.add(
                    AgentAuditLog(
                        session_id=event.session_id,
                        user_id=event.user_id,
                        event_type=event.event_type,
                        timestamp=event.timestamp,
                        payload=event.payload,
                    )
                )
                # commit on session.begin() context exit
            _log.debug(
                "sql_audit_sink.record",
                session_id=event.session_id,
                event_type=event.event_type,
            )
        except SQLAlchemyError as exc:
            raise _wrap_audit_sink_error(
                exc, session_id=event.session_id, operation="record"
            ) from exc


__all__ = ["AgentAuditLog", "SqlAuditSink", "audit_metadata"]
