"""Route structlog through stdlib ``logging`` for one test, then restore it (BR-026).

Not a test module (``python_files = ["test_*.py"]``), like
``tests/loop/golden_capture.py``. The SDK never configures structlog; a host
may route it through stdlib ``logging``. :func:`stdlib_routed_structlog`
builds the configuration BR-024's sentinel measured: ``add_log_level`` and
one of three stdlib recipes as the processor chain, stdlib's
``LoggerFactory`` and ``BoundLogger``, and no logger cache.

* ``render_to_log_kwargs`` and ``render_to_log_args_and_kwargs`` (the
  second added in structlog 25.1.0) pass every event key apart from
  ``event``, ``exc_info``, ``stack_info`` and ``stacklevel`` (and
  ``positional_args`` for the second) to stdlib as the record's ``extra``
  (structlog 26.1.0 ``stdlib.py:945-1007``, by reading). Stdlib's
  ``Logger.makeRecord`` raises ``KeyError("Attempt to overwrite ...")`` for
  an ``extra`` key that is ``message``, ``asctime`` or already a record
  attribute, so a log call passing one raised before BR-026 (BR-026
  evidence, P2). Under these recipes a record carries the SDK's keys as
  attributes, and ``record.msg`` is the event name.
* ``ProcessorFormatter.wrap_for_formatter`` passes the event dict as
  ``record.msg`` and only ``_logger``/``_name`` as ``extra``, so the SDK's
  keys never reach ``makeRecord``. It is the control recipe.

The yielded list holds every record a ``fifty_agent_sdk`` logger produced.
The handler formats each record before keeping it, as a host's handler
would (``%(levelname)s %(name)s %(message)s``; a ``ProcessorFormatter``
with ``ConsoleRenderer(colors=False)`` for the control recipe). Under
``render_to_log_kwargs`` and ``render_to_log_args_and_kwargs`` the kept
record carries ``record.message``; under ``wrap_for_formatter`` it does not
(measured with structlog 26.1.0 on CPython 3.11.15 and 3.14.3, BR-028
evidence; B19 re-renders the record instead), so assert neither its
presence nor its absence across recipes. Read a field with
:func:`record_field`.

Restoring: the handler is removed and the ``fifty_agent_sdk`` logger's
level and ``propagate`` come back, as does ``logging.raiseExceptions``
(set to ``True`` inside, so a handler error is reported, not silenced).
structlog is reset to its defaults and, if it was configured before,
configured again with the previous values, as BR-024's
``strict_log_stream`` does. ``cache_logger_on_first_use`` stays ``False``:
with a cache, the SDK's module-level ``_log`` proxies would keep the stdlib
chain after the test (S4 pins the restore). The root logger is not touched,
so propagated records still reach pytest's own handlers. Stdlib creates a
``fifty_agent_sdk.<module>`` logger the first time that module logs through
the helper; it keeps stdlib's defaults (level ``NOTSET``, no handler,
propagating), and stdlib never removes a logger. Rows that stand
for stdlib's default level pass ``logging.WARNING`` explicitly rather than
leaving the logger unset, because the root level is global state another
test could change.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

import pytest
import structlog

SDK_LOGGER: Final = "fifty_agent_sdk"

HAS_ARGS_AND_KWARGS: Final = hasattr(structlog.stdlib, "render_to_log_args_and_kwargs")

# The two recipes that pass the SDK's keys as ``extra``, and the control recipe.
EXTRA_RECIPES: Final = [
    pytest.param("render_to_log_kwargs", id="render_to_log_kwargs"),
    pytest.param(
        "render_to_log_args_and_kwargs",
        id="render_to_log_args_and_kwargs",
        marks=pytest.mark.skipif(
            not HAS_ARGS_AND_KWARGS,
            reason="structlog before 25.1.0 has no render_to_log_args_and_kwargs",
        ),
    ),
]
ALL_RECIPES: Final = [
    *EXTRA_RECIPES,
    pytest.param("wrap_for_formatter", id="wrap_for_formatter"),
]


class _RecordingHandler(logging.Handler):
    """Formats each record, as a host handler would, and keeps it."""

    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(logging.NOTSET)
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.format(record)
        self._records.append(record)


def _renderer(recipe: str) -> Any:
    if recipe == "render_to_log_kwargs":
        return structlog.stdlib.render_to_log_kwargs
    if recipe == "render_to_log_args_and_kwargs":
        return structlog.stdlib.render_to_log_args_and_kwargs
    if recipe == "wrap_for_formatter":
        return structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    raise ValueError(f"unknown recipe {recipe!r}")


def _formatter(recipe: str) -> logging.Formatter:
    if recipe == "wrap_for_formatter":
        return structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False),
            ]
        )
    return logging.Formatter("%(levelname)s %(name)s %(message)s")


@contextmanager
def stdlib_routed_structlog(recipe: str, level: int) -> Iterator[list[logging.LogRecord]]:
    """Route structlog through stdlib with ``recipe``; yield the SDK's records at ``level``."""
    renderer = _renderer(recipe)
    was_configured = structlog.is_configured()
    previous = structlog.get_config()
    sdk_logger = logging.getLogger(SDK_LOGGER)
    previous_level = sdk_logger.level
    previous_propagate = sdk_logger.propagate
    previous_raise = logging.raiseExceptions
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    handler.setFormatter(_formatter(recipe))

    structlog.reset_defaults()
    structlog.configure(
        processors=[structlog.stdlib.add_log_level, renderer],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
    logging.raiseExceptions = True
    sdk_logger.addHandler(handler)
    sdk_logger.setLevel(level)
    try:
        yield records
    finally:
        sdk_logger.removeHandler(handler)
        sdk_logger.setLevel(previous_level)
        sdk_logger.propagate = previous_propagate
        logging.raiseExceptions = previous_raise
        structlog.reset_defaults()
        if was_configured:
            structlog.configure(**previous)


def event_of(record: logging.LogRecord) -> Any:
    """The structlog event name: ``record.msg``, or its ``event`` under the control recipe."""
    if isinstance(record.msg, dict):
        return record.msg.get("event")
    return record.msg


def records_for(records: list[logging.LogRecord], event: str) -> list[logging.LogRecord]:
    """The records whose structlog event is ``event``, in emission order."""
    return [record for record in records if event_of(record) == event]


def record_field(record: logging.LogRecord, key: str) -> Any:
    """A structlog key: a record attribute under the ``extra`` recipes, an event-dict entry under the control."""
    if isinstance(record.msg, dict):
        return record.msg[key]
    return record.__dict__[key]


def record_keys(record: logging.LogRecord) -> set[str]:
    """The structlog keys a record carries apart from ``event`` (control recipe only)."""
    assert isinstance(record.msg, dict), "record_keys reads the control recipe's event dict"
    return set(record.msg) - {"event"}
