"""SQLAlchemy 2.0 asyncio implementation of :class:`StateStore`.

:class:`SqlStateStore` is the SDK's durable conversation-state backend
when callers need persistence across process restarts. It targets the
SQLAlchemy 2.0 asyncio API and is tested against SQLite (via
``aiosqlite``) by default and Postgres (via ``asyncpg`` or ``psycopg``)
behind a marker.

Extras requirement
    This module requires the optional ``sql`` extra::

        pip install 'agent-sdk[sql]'

    Importing :mod:`agent_sdk` itself does NOT pull SQLAlchemy. The SQL
    surface is re-exported lazily from :mod:`agent_sdk.state` and the
    package root via module-level ``__getattr__``; first access triggers
    this module's import, and a missing dependency surfaces as a clear
    :class:`ImportError` referencing the extras line above.

Schema lifecycle
    The SDK ships the table definitions as declarative ORM models but
    does NOT own migrations. Consumers register :data:`sql_metadata`
    with their own Alembic environment::

        # alembic/env.py
        from agent_sdk import sql_metadata
        target_metadata = sql_metadata

    Then::

        alembic revision --autogenerate -m "agent_sdk schema"

    Review the generated revision before applying — the SDK takes no
    position on naming conventions, schemas, or tablespaces in your
    deployment.

Dialect parity
    The schema is engineered to work on SQLite and Postgres without
    conditional code paths in the store itself. Dialect drift is
    confined to:

    * :class:`AgentSession.meta` — ``JSON().with_variant(JSONB(),
      "postgresql")`` so the column becomes ``JSONB`` on Postgres
      (indexable, queryable with ``->>``) and ``TEXT``-with-JSON on
      SQLite.
    * Timestamp columns are ``DateTime(timezone=True)`` so Postgres
      stores ``timestamptz``; SQLAlchemy normalises SQLite's ISO-string
      representation on read.

    SQLite-only note: ``ON DELETE CASCADE`` at the schema level
    requires ``PRAGMA foreign_keys = ON`` per connection. The store's
    :meth:`delete` uses the ORM cascade path which fires regardless of
    that pragma, so the SDK's own delete is correct on either dialect.
    Consumers issuing raw-SQL ``DELETE FROM agent_sessions`` against a
    SQLite engine must enable the pragma themselves.

Sequence semantics
    :meth:`append` allocates a per-session monotonically-increasing
    ``sequence`` value (starting at ``1``) inside a transaction that
    acquires a ``SELECT ... FOR UPDATE`` lock on the parent
    ``agent_sessions`` row. On Postgres this serialises concurrent
    appenders for the same session; on SQLite ``with_for_update`` is a
    documented no-op and the database's global writer serialisation
    provides the same guarantee. A
    ``UniqueConstraint(session_id, sequence)`` is the schema-level
    safety net — a hypothetical race that reached the INSERT step
    would surface as :class:`sqlalchemy.exc.IntegrityError`, wrapped
    into :class:`agent_sdk.errors.StateStoreError` with
    ``context["wrapped"] == "IntegrityError"`` for deterministic
    caller-side handling.

Error wrapping contract
    Every public method wraps :class:`sqlalchemy.exc.SQLAlchemyError`
    (the SQLAlchemy base class) into
    :class:`agent_sdk.errors.StateStoreError` with:

    * ``message``: ``"SqlStateStore.<operation> failed for session_id=<id>"``
    * ``context["session_id"]``: the input session id (echoed for log
      correlation)
    * ``context["wrapped"]``: the underlying exception's class name
      (e.g., ``"IntegrityError"``, ``"OperationalError"``) — read by
      the Runner's ``runner.persist_failed`` ERROR log per TD-004
    * ``context["operation"]``: ``"get_messages"``, ``"append"``, or
      ``"delete"``
    * ``__cause__``: the original exception, via ``raise ... from exc``

    :class:`asyncio.CancelledError` propagates untouched. Pydantic
    ``ValidationError`` on read (which would indicate corrupt rows) is
    not wrapped either — it is a corruption signal, not a backend
    failure.

Engine ownership
    The constructor accepts EITHER an :class:`AsyncEngine` or a
    connection URL string. When a URL is provided, the store creates
    its own engine and tracks ``self._owns_engine = True``;
    :meth:`aclose` disposes it. When an externally-built engine is
    passed, the store DOES NOT dispose it on close — the caller owns
    its lifecycle.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

import structlog

try:
    from sqlalchemy import (
        BigInteger,
        DateTime,
        ForeignKey,
        Index,
        Integer,
        String,
        Text,
        UniqueConstraint,
        func,
        select,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import (
        DeclarativeBase,
        Mapped,
        mapped_column,
        relationship,
    )
    from sqlalchemy.types import JSON
except ImportError as exc:  # pragma: no cover - exercised via importlib in tests
    raise ImportError(
        "agent_sdk.state.sql requires SQLAlchemy. Install with: pip install 'agent-sdk[sql]'"
    ) from exc

from agent_sdk.errors import StateStoreError
from agent_sdk.llm.types import ChatMessage

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Successful operations log at ``DEBUG`` with the session id and a small
shape summary (rows affected, sequence assigned). Failures are NOT
logged here — the wrapped :class:`StateStoreError` carries everything
the Runner's ``runner.persist_failed`` ERROR log needs.
"""


class Base(DeclarativeBase):
    """Declarative base shared by the SDK's ORM models.

    Module-private on purpose: consumers register :data:`sql_metadata`
    (the underlying ``Base.metadata``) with their Alembic environment
    rather than subclassing this base. Keeping ``Base`` out of
    ``__all__`` prevents consumer tables from accidentally merging into
    the SDK's autogenerate run.
    """


class AgentSession(Base):
    """Per-session metadata row.

    A parent row exists for every active conversation; it is created
    lazily on the first :meth:`SqlStateStore.append` and removed by
    :meth:`SqlStateStore.delete` (which cascades to messages).

    The ``meta`` Python attribute maps to a SQL column literally named
    ``metadata``; we cannot name the attribute ``metadata`` directly
    because :class:`DeclarativeBase` reserves that name for the
    metadata registry.
    """

    __tablename__ = "agent_sessions"

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    """Opaque caller-provided id (the SDK does no validation)."""

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Insertion timestamp (server-side default; timezone-aware)."""

    last_active_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    """Bumped on every :meth:`SqlStateStore.append`."""

    meta: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=True,
        default=None,
    )
    """Forward-compatibility column for per-session tags / annotations.

    Not exposed at the store API in BR-009. ``JSONB`` on Postgres,
    JSON-encoded ``TEXT`` on SQLite. The Python attribute is ``meta`` to
    avoid shadowing :attr:`DeclarativeBase.metadata`.
    """

    messages: Mapped[list[AgentMessage]] = relationship(
        "AgentMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=False,
        order_by="AgentMessage.sequence",
    )
    """Per-session messages, ordered by sequence ascending.

    ``cascade="all, delete-orphan"`` makes
    :meth:`SqlStateStore.delete` cascade through the ORM regardless of
    schema-level FK enforcement (SQLite needs an explicit ``PRAGMA
    foreign_keys = ON`` for the FK-level cascade to fire). With
    ``passive_deletes=False``, SQLAlchemy issues an explicit DELETE for
    each child row from Python, which is correct on every dialect.
    """


class AgentMessage(Base):
    """One persisted :class:`ChatMessage` row.

    Round-trips all four :class:`ChatMessage` fields (``role``,
    ``content``, ``name``, ``tool_call_id``) plus a per-session
    ``sequence`` and the surrogate ``id`` primary key. The
    ``sequence`` is the durable ordering key; ``id`` is only used for
    row identity.
    """

    __tablename__ = "agent_messages"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "sequence",
            name="uq_agent_messages_session_sequence",
        ),
        Index(
            "ix_agent_messages_session_sequence",
            "session_id",
            "sequence",
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
        ForeignKey("agent_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    """Foreign key to :attr:`AgentSession.session_id`."""

    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    """Per-session monotonic order key (``>= 1``).

    Allocated inside :meth:`SqlStateStore.append` as ``MAX(sequence)+1``
    under a ``SELECT ... FOR UPDATE`` lock on the parent row.
    """

    role: Mapped[str] = mapped_column(String(32), nullable=False)
    """One of ``"system"`` / ``"user"`` / ``"assistant"`` / ``"tool"``."""

    content: Mapped[str] = mapped_column(Text, nullable=False)
    """Arbitrary text; an empty string is permitted (see
    :class:`ChatMessage`)."""

    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """Optional named speaker / tool name."""

    tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """Optional id echoed back on ``role="tool"`` replies."""

    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    """Insertion timestamp (server-side default; timezone-aware)."""

    session: Mapped[AgentSession] = relationship(
        "AgentSession",
        back_populates="messages",
    )
    """Parent session (the inverse side of :attr:`AgentSession.messages`)."""


sql_metadata = Base.metadata
"""The :class:`sqlalchemy.MetaData` object describing the SDK's schema.

Consumers register this with their Alembic environment::

    # alembic/env.py
    from agent_sdk import sql_metadata
    target_metadata = sql_metadata

The SDK does NOT ship migration scripts; consumers run
``alembic revision --autogenerate`` against this object and review the
output before applying.
"""


def _wrap_state_store_error(
    exc: SQLAlchemyError,
    *,
    session_id: str,
    operation: str,
) -> StateStoreError:
    """Build the SDK's standard wrap of a SQLAlchemy backend failure.

    Returns the :class:`StateStoreError`; the caller writes
    ``raise _wrap_state_store_error(...) from exc`` so the
    ``__cause__`` chain (the BR-009 error-wrapping contract) is
    preserved at every call site.
    """
    return StateStoreError(
        f"SqlStateStore.{operation} failed for session_id={session_id}",
        context={
            "session_id": session_id,
            "wrapped": type(exc).__name__,
            "operation": operation,
        },
    )


class SqlStateStore:
    """SQLAlchemy-backed implementation of :class:`StateStore`.

    Satisfies :class:`agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).

    Concurrency model:
        Two layers of serialisation cover same-session writes:

        * **In-process:** a per-``session_id`` :class:`asyncio.Lock`
          serialises :meth:`append` and :meth:`delete` calls on the
          same store instance. Different sessions never block each
          other. Mirrors the locking shape of
          :class:`MemoryStateStore`.
        * **Cross-process / DB-level:** every transaction acquires
          ``SELECT ... FOR UPDATE`` on the parent
          :class:`AgentSession` row. On Postgres this serialises
          concurrent writers from any process; on SQLite the DB
          serialises writers globally and ``with_for_update`` is a
          documented no-op.

        Parent-row creation is idempotent via ``INSERT ... ON
        CONFLICT DO NOTHING``, so the cold-start race (two appenders
        on a never-seen ``session_id`` both attempting to create the
        parent row) cannot produce a primary-key violation.

    Engine ownership:
        * Construct with a URL string → the store creates and owns the
          engine. :meth:`aclose` disposes it.
        * Construct with an :class:`AsyncEngine` → the caller owns the
          engine. :meth:`aclose` is a no-op on the engine; the caller
          must dispose it themselves.

    Example:
        Construct from a URL (the store owns and disposes the engine)
        and wire it into an :class:`AgentRunner`::

            from agent_sdk import (
                JSON_MODE_OUTPUT_FORMAT, AgentLoop, AgentRunner,
                JsonModeParser, PromptSections, Registry, SafetyConfig,
                SqlStateStore,
            )
            from agent_sdk.llm import OpenAICompatibleClient

            state = SqlStateStore("sqlite+aiosqlite:///./agent.db")
            try:
                runner = AgentRunner(
                    loop=AgentLoop(
                        llm=OpenAICompatibleClient(...),
                        registry=Registry(),
                        parser=JsonModeParser(),
                        prompts=PromptSections(persona="You are helpful."),
                        safety=SafetyConfig(),
                        model="gpt-4o",
                        output_format=JSON_MODE_OUTPUT_FORMAT,
                    ),
                    state=state,
                    system_prompt="You are a helpful customer-support agent.",
                )
                async for event in runner.run("session-abc", "Hello"):
                    print(event)
            finally:
                await state.aclose()

        Schema must be created out-of-band (the SDK does not own
        migrations); register :data:`sql_metadata` with your Alembic
        environment first. See the module docstring's
        "Schema lifecycle" section.

    Failure mode:
        Every public method wraps :class:`sqlalchemy.exc.SQLAlchemyError`
        into :class:`agent_sdk.errors.StateStoreError` with
        ``context["wrapped"]`` carrying the underlying class name. See
        the module docstring for the full contract.
    """

    def __init__(self, engine_or_url: AsyncEngine | str) -> None:
        """Construct a :class:`SqlStateStore`.

        Args:
            engine_or_url: Either a fully-configured
                :class:`sqlalchemy.ext.asyncio.AsyncEngine` (caller-owned)
                or a connection URL string (e.g.,
                ``"sqlite+aiosqlite:///:memory:"``,
                ``"postgresql+asyncpg://..."``) — in which case the
                store creates and owns the engine.
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
        # Per-session asyncio locks serialise concurrent appends/deletes
        # within this process. The DB-level row lock
        # (`SELECT ... FOR UPDATE`) is still the cross-process correctness
        # mechanism on Postgres; the in-process lock is an additional
        # guarantee that mirrors :class:`MemoryStateStore` and avoids
        # relying on `with_for_update` being honoured (which it is not on
        # SQLite). Different sessions never block each other.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._lock_registry_lock: asyncio.Lock = asyncio.Lock()

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the per-session :class:`asyncio.Lock`, creating it lazily.

        Same double-checked-locking pattern as
        :class:`MemoryStateStore` — the hot path is lock-free; only
        first-time creation takes the registry lock.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The :class:`asyncio.Lock` associated with this session.
        """
        existing = self._session_locks.get(session_id)
        if existing is not None:
            return existing
        async with self._lock_registry_lock:
            existing = self._session_locks.get(session_id)
            if existing is not None:
                return existing
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
            return lock

    async def aclose(self) -> None:
        """Dispose the underlying engine if (and only if) the store owns it.

        Safe to call multiple times. When the engine was passed in by
        the caller, this is a no-op — the caller's engine lifecycle is
        not the store's responsibility.
        """
        if self._owns_engine:
            await self._engine.dispose()

    async def _upsert_parent(self, session: AsyncSession, session_id: str) -> None:
        """Idempotently ensure an :class:`AgentSession` row exists.

        Two concurrent appends on a never-seen ``session_id`` would
        both see a NULL parent on the initial SELECT and race to insert
        — the slower one would fail the primary-key constraint. We use
        the dialect-aware ``INSERT ... ON CONFLICT DO NOTHING`` so both
        appenders converge on a single row without an exception. The
        subsequent ``SELECT ... FOR UPDATE`` in :meth:`append` then
        serialises the rest of the work.

        SQLite ``>=3.24`` and Postgres both support ``ON CONFLICT``;
        the SDK targets SQLAlchemy 2.0 which requires recent SQLite,
        so no fallback path is needed.

        Args:
            session: Active :class:`AsyncSession` (inside a transaction).
            session_id: Opaque session identifier.
        """
        dialect = session.bind.dialect.name if session.bind is not None else ""
        stmt: Any
        if dialect == "postgresql":
            stmt = pg_insert(AgentSession).values(session_id=session_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=["session_id"])
        else:
            # SQLite (and any other dialect we run against in tests).
            stmt = sqlite_insert(AgentSession).values(session_id=session_id)
            stmt = stmt.on_conflict_do_nothing(index_elements=["session_id"])
        await session.execute(stmt)

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Return the persisted messages for ``session_id``.

        Returns a freshly-constructed list of :class:`ChatMessage`
        instances ordered by ``sequence`` ascending. An unknown session
        yields ``[]`` (not an error).

        Args:
            session_id: Opaque session identifier.

        Returns:
            A list of :class:`ChatMessage` values in append order.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name.
        """
        try:
            async with self._session_factory() as session:
                rows = await session.scalars(
                    select(AgentMessage)
                    .where(AgentMessage.session_id == session_id)
                    .order_by(AgentMessage.sequence.asc())
                )
                messages = [
                    ChatMessage(
                        role=row.role,  # type: ignore[arg-type]
                        content=row.content,
                        name=row.name,
                        tool_call_id=row.tool_call_id,
                    )
                    for row in rows
                ]
                _log.debug(
                    "sql_state_store.get_messages",
                    session_id=session_id,
                    count=len(messages),
                )
                return messages
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="get_messages"
            ) from exc

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's ordered message log.

        Allocates a per-session ``sequence`` value transactionally:
        opens a transaction, takes a ``SELECT ... FOR UPDATE`` lock on
        the parent :class:`AgentSession` row (creating it if absent),
        computes ``MAX(sequence) + 1`` for this session under the held
        lock, and inserts the new :class:`AgentMessage` row. The
        ``UniqueConstraint(session_id, sequence)`` is the database-level
        safety net.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name (e.g.,
                ``"IntegrityError"`` if a hypothetical race reached the
                unique-constraint check).
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                # 1. Idempotently ensure the parent row exists. Two concurrent
                #    appenders on a never-seen session_id would both see a
                #    NULL parent and race to create it; ON CONFLICT DO NOTHING
                #    makes that race correct on both dialects (SQLite >= 3.24
                #    and Postgres). The asyncio lock acquired above already
                #    serialises in-process appenders for this session, but
                #    the upsert is still required for cross-process safety.
                await self._upsert_parent(session, session_id)

                # 2. Acquire a row lock on the parent. `with_for_update()` is
                #    a no-op on SQLite (writers are serialised globally) and
                #    SELECT ... FOR UPDATE on Postgres — serialising all
                #    appenders for this session_id.
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                # The upsert above guarantees this row exists.
                assert parent is not None
                # Touch the parent so `onupdate=func.now()` fires and
                # `last_active_at` is bumped on this append.
                parent.last_active_at = func.now()

                # 3. Compute the next sequence under the held lock.
                # COALESCE(MAX(sequence), 0) + 1 is always a non-null integer,
                # so we can rely on scalar_one() here. If this SELECT is ever
                # changed in a way that could yield no row (e.g., dropping
                # COALESCE), the call below will raise NoResultFound -- that
                # surfaces as a SQLAlchemyError and gets wrapped uniformly by
                # the surrounding except block.
                result = await session.execute(
                    select(func.coalesce(func.max(AgentMessage.sequence), 0) + 1).where(
                        AgentMessage.session_id == session_id
                    )
                )
                next_seq: int = int(result.scalar_one())

                # 4. Insert the message.
                session.add(
                    AgentMessage(
                        session_id=session_id,
                        sequence=next_seq,
                        role=message.role,
                        content=message.content,
                        name=message.name,
                        tool_call_id=message.tool_call_id,
                    )
                )
                # commit on session.begin() context exit
            _log.debug(
                "sql_state_store.append",
                session_id=session_id,
                sequence=next_seq,
            )
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="append") from exc

    async def delete(self, session_id: str) -> None:
        """Remove the session row and all its messages.

        Idempotent — deleting an unknown session is a silent no-op (no
        rows affected, no error). Cascade is handled via the ORM
        relationship (``cascade="all, delete-orphan"``) so it works on
        both SQLite (where schema-level CASCADE requires
        ``PRAGMA foreign_keys = ON``) and Postgres.

        Args:
            session_id: Opaque session identifier.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                SQLAlchemy exception class name.
        """
        lock = await self._get_session_lock(session_id)
        try:
            async with lock, self._session_factory() as session, session.begin():
                parent = await session.scalar(
                    select(AgentSession)
                    .where(AgentSession.session_id == session_id)
                    .with_for_update()
                )
                if parent is None:
                    _log.debug(
                        "sql_state_store.delete",
                        session_id=session_id,
                        existed=False,
                    )
                    return
                # ORM-level delete fires the cascade regardless of dialect.
                await session.delete(parent)
            _log.debug(
                "sql_state_store.delete",
                session_id=session_id,
                existed=True,
            )
        except SQLAlchemyError as exc:
            raise _wrap_state_store_error(exc, session_id=session_id, operation="delete") from exc


__all__ = [
    "AgentMessage",
    "AgentSession",
    "SqlStateStore",
    "sql_metadata",
]
