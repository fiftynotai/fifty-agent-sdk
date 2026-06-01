"""Redis implementation of :class:`StateStore` built on ``redis.asyncio``.

:class:`RedisStateStore` is the SDK's durable-but-ephemeral conversation-state
backend: it persists across process restarts (unlike
:class:`agent_sdk.state.memory.MemoryStateStore`) yet is a natural fit for a
TTL-bounded "hot session" cache rather than a system-of-record. For a true
system-of-record use the SQL backend (:class:`agent_sdk.state.sql.SqlStateStore`).

Extras requirement
    This module requires the optional ``redis`` extra::

        pip install 'agent-sdk[redis]'

    Importing :mod:`agent_sdk` itself does NOT pull redis-py. The Redis
    surface is re-exported lazily from :mod:`agent_sdk.state` and the
    package root via module-level ``__getattr__``; first access triggers
    this module's import, and a missing dependency surfaces as a clear
    :class:`ImportError` referencing the extras line above.

Key layout
    Each session maps to a single Redis key::

        <key_prefix><session_id>

    where ``key_prefix`` defaults to ``"agent_sdk:state:"``. The value is a
    Redis list whose members are JSON-encoded :class:`ChatMessage` payloads
    (one ``model_dump_json()`` string per element). ``RPUSH`` appends to the
    tail, ``LRANGE 0 -1`` reads the whole list in append order — Redis list
    ordering is exactly the contract's append ordering, with no separate
    sequence column needed.

TTL semantics
    When ``ttl_seconds`` is a positive integer, every :meth:`append` issues
    an ``EXPIRE`` alongside the ``RPUSH`` so the session's expiry window
    slides forward on each write — a "hot session stays alive" cache. A
    session that goes quiet for longer than ``ttl_seconds`` is evicted by
    Redis. When ``ttl_seconds`` is ``None`` no ``EXPIRE`` is ever issued and
    the list is durable until :meth:`delete`. :meth:`get_messages` NEVER
    sets or refreshes a TTL — reading a session does not keep it alive.

Atomicity
    :meth:`append` issues ``RPUSH`` and (when applicable) ``EXPIRE`` inside a
    single ``MULTI``/``EXEC`` transaction via a ``transaction=True`` pipeline.
    A concurrent reader therefore sees either the pre-append or post-append
    state of the list, never a half-written one — satisfying the
    :class:`StateStore` ``append``-vs-read atomicity invariant.

Error wrapping contract
    Every public method wraps :class:`redis.exceptions.RedisError` (the
    redis-py base exception class) into
    :class:`agent_sdk.errors.StateStoreError` with:

    * ``message``: ``"RedisStateStore.<operation> failed for session_id=<id>"``
    * ``context["session_id"]``: the input session id (echoed for log
      correlation)
    * ``context["wrapped"]``: the underlying exception's class name
      (e.g., ``"ConnectionError"``, ``"TimeoutError"``) — read by
      the Runner's ``runner.persist_failed`` ERROR log per TD-004
    * ``context["operation"]``: ``"get_messages"``, ``"append"``, or
      ``"delete"``
    * ``__cause__``: the original exception, via ``raise ... from exc``

    :class:`asyncio.CancelledError` propagates untouched (it is not a
    :class:`~redis.exceptions.RedisError`). Pydantic ``ValidationError`` on
    read — which would indicate a corrupt list member — is not wrapped
    either: it is a corruption signal, not a backend failure (same stance
    as the SQL backend).

Connection ownership
    The constructor accepts a connection URL string and the store creates
    and owns the underlying connection pool. :meth:`aclose` releases it;
    callers should invoke it in a ``finally`` block when done with the
    store.
"""

from __future__ import annotations

from typing import Any, Final, cast

import structlog

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError
except ImportError as exc:  # pragma: no cover - exercised via importlib in tests
    raise ImportError(
        "agent_sdk.state.redis requires redis-py. "
        "Install with: pip install 'agent-sdk[redis]'"
    ) from exc

from agent_sdk.errors import StateStoreError
from agent_sdk.llm.types import ChatMessage

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger.

Successful operations log at ``DEBUG`` with the session id and a small
shape summary (message count, TTL applied). Failures are NOT logged
here — the wrapped :class:`StateStoreError` carries everything the
Runner's ``runner.persist_failed`` ERROR log needs.
"""

_DEFAULT_KEY_PREFIX: Final = "agent_sdk:state:"
"""Default namespace prefix for session keys.

Prepended to each ``session_id`` to form the Redis key. A prefix keeps the
SDK's keys grouped under one namespace so they are easy to spot, scope, or
flush without disturbing co-tenant data in a shared Redis instance.
"""


def _wrap_state_store_error(
    exc: RedisError,
    *,
    session_id: str,
    operation: str,
) -> StateStoreError:
    """Build the SDK's standard wrap of a Redis backend failure.

    Returns the :class:`StateStoreError`; the caller writes
    ``raise _wrap_state_store_error(...) from exc`` so the ``__cause__``
    chain (the BR-009 error-wrapping contract, shared by BR-010) is
    preserved at every call site.
    """
    return StateStoreError(
        f"RedisStateStore.{operation} failed for session_id={session_id}",
        context={
            "session_id": session_id,
            "wrapped": type(exc).__name__,
            "operation": operation,
        },
    )


class RedisStateStore:
    """Redis-backed implementation of :class:`StateStore`.

    Satisfies :class:`agent_sdk.state.protocol.StateStore` structurally
    (no explicit inheritance needed thanks to ``@runtime_checkable``).

    Storage model:
        One Redis list per session, keyed ``<key_prefix><session_id>``,
        whose members are JSON-encoded :class:`ChatMessage` payloads.
        ``RPUSH`` appends; ``LRANGE 0 -1`` reads the whole list in append
        order. See the module docstring's "Key layout" section.

    TTL model:
        With a positive ``ttl_seconds`` every :meth:`append` slides the
        session's expiry window forward (``EXPIRE`` re-issued on each
        write) — a hot-session cache. With ``ttl_seconds=None`` the list
        never expires. :meth:`get_messages` never touches the TTL.

    Atomicity:
        :meth:`append` runs ``RPUSH`` + ``EXPIRE`` in a single
        ``MULTI``/``EXEC`` transaction, so a concurrent reader sees only
        pre- or post-append state — the :class:`StateStore` atomicity
        invariant.

    Connection ownership:
        The store owns the connection pool created from the URL.
        :meth:`aclose` releases it; call it in a ``finally`` block.

    Example:
        Construct from a URL and wire the store into an
        :class:`AgentRunner`, releasing the connection pool on exit::

            from agent_sdk import (
                JSON_MODE_OUTPUT_FORMAT, AgentLoop, AgentRunner,
                JsonModeParser, PromptSections, Registry, RedisStateStore,
                SafetyConfig,
            )
            from agent_sdk.llm import OpenAICompatibleClient

            state = RedisStateStore(
                "redis://localhost:6379/0", ttl_seconds=3600
            )
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

    Failure mode:
        Every public method wraps :class:`redis.exceptions.RedisError`
        into :class:`agent_sdk.errors.StateStoreError` with
        ``context["wrapped"]`` carrying the underlying class name. See
        the module docstring for the full contract.
    """

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        ttl_seconds: int | None = None,
    ) -> None:
        """Construct a :class:`RedisStateStore`.

        Args:
            url: A redis-py connection URL (e.g.,
                ``"redis://localhost:6379/0"`` or
                ``"rediss://user:pass@host:6379/1"``). The store creates
                and owns the connection pool built from this URL.
            key_prefix: Namespace prepended to each ``session_id`` to form
                the Redis key. Defaults to :data:`_DEFAULT_KEY_PREFIX`
                (``"agent_sdk:state:"``).
            ttl_seconds: Per-session time-to-live, in seconds. When a
                positive integer, every :meth:`append` re-issues an
                ``EXPIRE`` so the session's expiry window slides forward
                on each write. When ``None`` (the default), no ``EXPIRE``
                is ever issued and the session list is durable until
                :meth:`delete`.
        """
        # ``decode_responses=True`` makes list members come back as ``str``
        # (rather than ``bytes``), ready to hand straight to
        # ``ChatMessage.model_validate_json``.
        self._client: aioredis.Redis = aioredis.from_url(url, decode_responses=True)
        self._key_prefix: str = key_prefix
        self._ttl_seconds: int | None = ttl_seconds

    def _key(self, session_id: str) -> str:
        """Return the Redis key for ``session_id``.

        The ``session_id`` is opaque (the SDK does no validation, per the
        :class:`StateStore` contract); it is simply concatenated onto the
        configured prefix.

        Args:
            session_id: Opaque session identifier.

        Returns:
            The fully-qualified Redis key (``<key_prefix><session_id>``).
        """
        return f"{self._key_prefix}{session_id}"

    async def aclose(self) -> None:
        """Release the underlying connection pool.

        Safe to call multiple times — redis-py tolerates repeated
        ``aclose`` calls. Callers should invoke this in a ``finally``
        block to ensure clean connection-pool teardown.
        """
        await self._client.aclose()

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Return the persisted messages for ``session_id``.

        Returns a freshly-constructed list of :class:`ChatMessage`
        instances in append order. An unknown session yields ``[]``:
        ``LRANGE`` on a missing key returns an empty list, satisfying the
        empty-session invariant with no special case. Reading does NOT
        refresh the session's TTL.

        Args:
            session_id: Opaque session identifier.

        Returns:
            A list of :class:`ChatMessage` values in append order,
            possibly empty.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        try:
            # redis-py types command methods as ``Awaitable[Any] | Any``
            # (sync and async clients share method bodies); ``cast`` pins
            # the awaitable arm so ``await`` is unambiguous under mypy
            # --strict. ``decode_responses=True`` guarantees ``str``
            # members; LRANGE on a missing key returns an empty list.
            raw: list[str] = await cast(
                "Any", self._client.lrange(self._key(session_id), 0, -1)
            )
            # A fresh list per call satisfies the defensive-copy invariant
            # automatically — callers mutating it cannot affect the store.
            # ValidationError from a corrupt member is intentionally NOT
            # caught here: a malformed payload is a corruption signal, not
            # a backend failure.
            messages = [ChatMessage.model_validate_json(item) for item in raw]
            _log.debug(
                "redis_state_store.get_messages",
                session_id=session_id,
                count=len(messages),
            )
            return messages
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="get_messages"
            ) from exc

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's ordered message log.

        Issues ``RPUSH`` (and, when ``ttl_seconds`` is set, ``EXPIRE``)
        inside a single ``MULTI``/``EXEC`` transaction so the write is
        atomic with respect to concurrent reads on the same session — a
        reader sees either the pre- or post-append list, never a partial
        one. When ``ttl_seconds`` is set the ``EXPIRE`` slides the
        session's expiry window forward on every append.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        payload = message.model_dump_json()
        key = self._key(session_id)
        try:
            # transaction=True wraps RPUSH + EXPIRE in a MULTI/EXEC block:
            # the two commands apply as one atomic unit, giving the
            # append-vs-read atomicity the StateStore contract requires.
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, payload)
                if self._ttl_seconds is not None:
                    pipe.expire(key, self._ttl_seconds)
                await pipe.execute()
            _log.debug(
                "redis_state_store.append",
                session_id=session_id,
                ttl=self._ttl_seconds,
            )
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="append"
            ) from exc

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id``.

        Idempotent — ``DEL`` on a missing key returns ``0`` and is a
        silent no-op, satisfying the idempotent-delete invariant with no
        special case.

        Args:
            session_id: Opaque session identifier.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails. ``context["wrapped"]`` carries the underlying
                redis-py exception class name.
        """
        try:
            # redis-py types command methods as ``Awaitable[Any] | Any``
            # because the sync and async clients share the same method
            # bodies; ``cast`` pins the awaitable arm so ``await`` is
            # unambiguous under mypy --strict. The DEL reply is an int
            # count of keys removed.
            removed = int(
                await cast("Any", self._client.delete(self._key(session_id)))
            )
            _log.debug(
                "redis_state_store.delete",
                session_id=session_id,
                existed=removed >= 1,
            )
        except RedisError as exc:
            raise _wrap_state_store_error(
                exc, session_id=session_id, operation="delete"
            ) from exc


__all__ = ["RedisStateStore"]
