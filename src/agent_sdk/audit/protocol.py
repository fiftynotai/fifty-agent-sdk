"""AuditSink protocol — pluggable, tamper-evident action provenance.

The :class:`AuditSink` protocol is the SDK's abstraction for capturing a
durable record of every agent action. :class:`agent_sdk.runner.AgentRunner`
emits an :class:`AuditEvent` at four points of every ``run()``: session
start, each tool invocation, the final answer, and any error.

Implementations may write to structured logs (see
:class:`agent_sdk.audit.console.ConsoleAuditSink` — the dev default), a SQL
table (:class:`agent_sdk.audit.sql.SqlAuditSink`), or any other backend.

Contract invariants
    * **Ordering.** :meth:`AuditSink.record` is called in the order events
      occur within a run. Implementations MUST NOT reorder events; the
      append-only nature of the record is a load-bearing property for any
      consumer treating it as a compliance trail.
    * **Concurrency.** :meth:`AuditSink.record` MUST be safe to call from
      multiple concurrent runs sharing one sink instance. Implementations
      that serialize do so internally; the caller does not lock.
    * **Failure is the caller's to swallow.** A raising :meth:`record` is
      tolerated by :class:`AgentRunner`, which catches the exception, logs a
      ``WARNING``, and continues — an audit-sink outage MUST NOT abort a
      live agent run. Implementations SHOULD still raise on backend failure
      (rather than silently dropping) so the gap is visible in the Runner's
      ``audit.emit_failed`` log line.
    * **Tamper-evidence is the sink's job.** "Tamper-evident" is a property
      of the *sink's storage* (append-only table, hash-chaining, WORM
      bucket), not of the emission path. The emission path's contract is
      best-effort delivery plus a loud warning when delivery fails.
    * **No secrets in the payload.** Consumers and tools MUST NOT place
      credentials or raw secrets into :attr:`AuditEvent.payload`; the
      payload is persisted verbatim and may appear in logs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class AuditEvent(BaseModel):
    """A single audited agent action.

    The provider-agnostic shape passed to every :class:`AuditSink`. Like the
    other SDK models it sets ``extra="forbid"`` so a typo'd field is caught
    at construction rather than silently dropped.

    Attributes:
        session_id: Opaque session identifier the action belongs to.
        user_id: Optional end-user identifier. The Runner does not have a
            user-id channel today and always emits ``None``; consumers
            wanting per-user attribution set it via a wrapping sink.
        timestamp: When the action occurred (timezone-aware UTC). The Runner
            stamps this at emission time — it is the *action* time, not the
            time the row was persisted.
        event_type: Short tag for the kind of action. The four values the
            Runner emits are ``"session_start"``, ``"tool_invocation"``,
            ``"final_answer"``, and ``"error"``; the field is a free-form
            string (not a ``Literal``) so consumers can emit their own
            event types through the same sink.
        payload: Structured, event-specific detail. Always a real ``dict``
            (never ``None``); defaults to ``{}``. MUST NOT contain
            credentials or raw secrets — it is persisted and logged
            verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    user_id: str | None = None
    timestamp: datetime
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class AuditSink(Protocol):
    """Pluggable audit-trail backend.

    See the module docstring for the full contract. Implementations are
    duck-typed: any class providing the single async method below with a
    matching signature satisfies :func:`isinstance` against
    :class:`AuditSink` thanks to ``@runtime_checkable``.

    Note:
        ``@runtime_checkable`` :class:`Protocol` instances only check for
        method *presence*, not signature compatibility. Mypy ``--strict``
        catches signature mismatches at type-check time.
    """

    async def record(self, event: AuditEvent) -> None:
        """Persist a single :class:`AuditEvent`.

        Called once per audited action, in occurrence order. Implementations
        MUST NOT reorder events and MUST be safe to call concurrently from
        multiple runs sharing the sink.

        Args:
            event: The :class:`AuditEvent` to record.

        Raises:
            Exception: Implementations MAY raise on backend failure. The
                :class:`agent_sdk.runner.AgentRunner` catches the exception,
                logs a ``WARNING``, and continues — a sink failure never
                aborts an agent run.
        """
        ...


__all__ = ["AuditEvent", "AuditSink"]
