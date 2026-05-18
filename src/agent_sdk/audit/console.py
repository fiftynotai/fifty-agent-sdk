"""ConsoleAuditSink — structured-log audit sink for dev / single-tenant use.

:class:`ConsoleAuditSink` is the zero-dependency :class:`AuditSink`
implementation. It emits each :class:`AuditEvent` as one ``INFO`` structlog
record under the fixed logger name ``agent_sdk.audit``.

It is the right choice when audit logs are already centralized (a single
deployment shipping structlog output to a log aggregator) and a separate
audit database is not warranted. It is NOT tamper-evident on its own — the
durability and integrity guarantees are whatever the surrounding log
pipeline provides. For a durable, queryable trail use
:class:`agent_sdk.audit.sql.SqlAuditSink`.
"""

from __future__ import annotations

from typing import Final

import structlog

from agent_sdk.audit.protocol import AuditEvent

_log: Final = structlog.get_logger("agent_sdk.audit")
"""Module-level structured logger.

Bound to the fixed name ``agent_sdk.audit`` (NOT ``__name__``) so every
audit record — from this sink and from :mod:`agent_sdk.audit.sql` — shares
one logger name a consumer can route or filter on.
"""


class ConsoleAuditSink:
    """An :class:`AuditSink` that writes each event to structlog at ``INFO``.

    Satisfies :class:`agent_sdk.audit.protocol.AuditSink` structurally (no
    explicit inheritance needed thanks to ``@runtime_checkable``).

    No constructor arguments — the sink is stateless and safe to share
    across concurrent runs (structlog loggers are themselves thread- and
    task-safe).

    Caution:
        Not tamper-evident on its own. The integrity of the trail is only
        as strong as the log pipeline the records are shipped through. Use
        :class:`agent_sdk.audit.sql.SqlAuditSink` when a durable, queryable,
        append-only record is required.
    """

    async def record(self, event: AuditEvent) -> None:
        """Emit ``event`` as a single ``INFO`` structlog record.

        The record's event name is ``audit.event``; the
        :class:`AuditEvent` fields are spread as structured key/values
        (``timestamp`` rendered as an ISO-8601 string).

        Args:
            event: The :class:`AuditEvent` to log.
        """
        _log.info(
            "audit.event",
            session_id=event.session_id,
            user_id=event.user_id,
            event_type=event.event_type,
            timestamp=event.timestamp.isoformat(),
            payload=event.payload,
        )


__all__ = ["ConsoleAuditSink"]
