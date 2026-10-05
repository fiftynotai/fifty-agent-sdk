"""Configure one of structlog's own processor chains for one test, then restore it (BR-027).

Not a test module (``python_files = ["test_*.py"]``), like
``tests/stdlib_routing.py``. The SDK never configures structlog; a host may
keep structlog's default configuration or add its own processors.
:func:`configured_structlog` builds either on a ``StringIO``, so a test can
read what a real renderer wrote:

* ``chain=None`` is structlog's default configuration as it is, apart from
  the output stream: ``merge_contextvars``, ``add_log_level``,
  ``StackInfoRenderer``, ``set_exc_info``,
  ``TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)`` and ``ConsoleRenderer``
  (measured on structlog 24.1.0 and 26.1.0; BR-027 evidence, P0). The
  helper asserts that the last default processor is a ``ConsoleRenderer``,
  so a structlog release whose default chain no longer ends in one fails
  loudly here; A1b's first-column check covers the default
  ``TimeStamper``'s format.
* ``chain`` is called with that default list and returns the processors to
  use. :data:`CHAINS` holds the chains BR-027's tests run.

Read the output by event, never by position: other SDK lines share the
buffer. :func:`audit_json_lines` parses the ``audit.event`` lines of a JSON
chain; :func:`console_lines` returns the lines of a ``ConsoleRenderer`` that
hold an event, with ANSI colour codes removed (by reading structlog
26.1.0's ``_config.py``, the default ``ConsoleRenderer`` decides on colours
when structlog is imported, and ``FORCE_COLOR`` turns them on unless
``NO_COLOR`` is set).

Restoring: structlog is reset to its defaults and, if it was configured
before, configured again with the previous values, as BR-024's
``strict_log_stream`` and BR-026's ``stdlib_routed_structlog`` do.
``cache_logger_on_first_use`` stays ``False``: with a cache, the SDK's
module-level ``_log`` proxies kept the first test's chain and buffer, and
later audit lines missed their own buffers and ``capture_logs`` (BR-027
battery, MH2). A5 in ``tests/audit/test_console.py`` pins the restore.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final

import structlog
from structlog.processors import JSONRenderer, MaybeTimeStamper, TimeStamper, add_log_level
from structlog.typing import Processor

ChainBuilder = Callable[[list[Processor]], list[Processor]]

# The default processors with a JSON renderer (so the default configuration's own
# ``TimeStamper`` runs), and three chains a host may build.
CHAINS: Final[dict[str, ChainBuilder]] = {
    "default_processors_json": lambda defaults: [*defaults[:-1], JSONRenderer()],
    "timestamper_iso": lambda _: [add_log_level, TimeStamper(fmt="iso"), JSONRenderer()],
    "timestamper_unix": lambda _: [add_log_level, TimeStamper(), JSONRenderer()],
    "maybe_timestamper": lambda _: [add_log_level, MaybeTimeStamper(fmt="iso"), JSONRenderer()],
}

_ANSI: Final = re.compile(r"\x1b\[[0-9;]*m")


@contextmanager
def configured_structlog(chain: ChainBuilder | None = None) -> Iterator[io.StringIO]:
    """Configure structlog's defaults, or ``chain(defaults)``, on a ``StringIO``; yield it."""
    was_configured = structlog.is_configured()
    previous = structlog.get_config()
    buffer = io.StringIO()
    try:
        structlog.reset_defaults()
        defaults = list(structlog.get_config()["processors"])
        assert isinstance(defaults[-1], structlog.dev.ConsoleRenderer), defaults
        processors: dict[str, Any] = {} if chain is None else {"processors": chain(defaults)}
        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=buffer),
            cache_logger_on_first_use=False,
            **processors,
        )
        yield buffer
    finally:
        structlog.reset_defaults()
        if was_configured:
            structlog.configure(**previous)


def audit_json_lines(buffer: io.StringIO) -> list[dict[str, Any]]:
    """The parsed ``audit.event`` lines of a JSON chain, in emission order."""
    lines = [json.loads(line) for line in buffer.getvalue().splitlines() if line]
    return [line for line in lines if line.get("event") == "audit.event"]


def console_lines(buffer: io.StringIO, event: str) -> list[str]:
    """The ``ConsoleRenderer`` lines that hold ``event``, ANSI colour codes removed."""
    text = _ANSI.sub("", buffer.getvalue())
    return [line for line in text.splitlines() if f" {event} " in line]
