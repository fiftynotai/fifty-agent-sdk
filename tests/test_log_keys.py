"""No logging call in the package passes a key stdlib ``logging`` refuses as ``extra`` (BR-026, AC-1) or a :data:`PROCESSOR_OWNED` key (BR-027, AC-2).

When a host routes structlog through stdlib ``logging`` with
``structlog.stdlib.render_to_log_kwargs`` or
``render_to_log_args_and_kwargs``, every event key apart from ``event``,
``exc_info``, ``stack_info`` and ``stacklevel`` (and ``positional_args``
for the second) becomes the record's ``extra`` (by reading structlog
26.1.0), and ``Logger.makeRecord`` raises
``KeyError("Attempt to overwrite ...")`` for a key that is ``message``,
``asctime`` or already an attribute of the new record.
Before BR-026 five keyword arguments in four log lines did that
(BR-026 evidence, P1 and P2).

**The reserved set** (:data:`RESERVED`) is built on the running
interpreter: the attributes of a fresh ``logging.LogRecord`` (built from
the class, not ``logging.getLogRecordFactory()``, which a host or a test
can replace), plus ``message`` and ``asctime``, which ``makeRecord``
refuses by name, plus :data:`_ADDED_IN_LATER_PYTHONS`. CPython 3.12 added
``LogRecord.taskName``: measured, CPython 3.11.15 sets 20 attributes and
accepts ``extra={"taskName": ...}``, while 3.13.2 and 3.14.3 set 21 and
refuse it (BR-026 evidence, P0). The union refuses ``taskName`` on
every interpreter, so CI's 3.11 job fails on a key that would raise on a
3.12 or later host (K6). ``exc_info`` and ``stack_info`` stay in the set although both
recipes pass them to stdlib as arguments, not ``extra``: a recipe written
by hand could put them in ``extra``, and no SDK call passes them (the
sweep does not see the ``exc_info`` that structlog's ``exception()`` adds
itself; no SDK call uses ``exception()``).
``stacklevel`` is not a record attribute, so it is not in the set. K2
passes every key the sweep collected through a real ``makeRecord`` too.

**The sweep** parses every ``.py`` file of the imported package
(``fifty_agent_sdk.__file__``'s directory, so a test run over another
tree's ``src`` sweeps that tree). A *logger receiver* is the target of an
assignment from a call ending in ``get_logger`` or ``wrap_logger``, from
``.bind(...)``/``.new(...)`` on a receiver, or from a receiver itself (to
a fixpoint, so ``log = _log.bind(...)`` and ``log2 = _log`` count). A
*swept call* is a log method
(``debug``, ``info``, ``warning``, ``warn``, ``error``, ``exception``,
``critical``, ``fatal``, ``msg``, ``log`` and their ``a``-prefixed async
forms), ``bind`` or ``new`` on a receiver or on a chain that builds one,
a call ending in ``bind_contextvars`` or ``bound_contextvars``, or (since
BR-027) a call ending in ``get_logger`` or ``wrap_logger`` that passes
keyword arguments. Its keys are the keyword names wherever they sit in a
multi-line call, and the string keys of a ``**{...}`` literal. A
``wrap_logger`` call's own parameters (``logger``, ``processors``,
``wrapper_class``, ``context_class``, ``cache_logger_on_first_use``,
``logger_factory_args``) are left out: structlog binds its other keywords
as initial values on every line the logger writes (by reading structlog
26.1.0 ``_config.py``). ``get_logger`` passes its keywords on to
``wrap_logger`` (``_config.py:143``), so none of those parameters is an
initial value there (four become ``wrap_logger``'s arguments; ``logger``
and ``logger_factory_args`` raise ``TypeError``); the sweep reads every
``get_logger`` keyword,
which can over-report a key but cannot miss one. A logger-factory call is
reported under the factory's name. The package's 11 calls to a logger
factory pass only a name (BR-027 evidence, P1). It fails closed: a ``**`` of
anything but a literal with string keys is *unverifiable*; a
logging-style call (a log method called with a constant string first
argument or with keywords, or ``bind``/``new`` called with keywords) on
anything that is not a receiver (apart from ``warnings``) is an
*unrecognised receiver*; any other method called on a receiver is an
*unrecognised method* (``unbind`` and ``try_unbind`` pass without
keywords); and any call ending in ``getLogger`` is a *stdlib logger*
(teach the sweep about ``extra=`` before adding one). Three uses whose
later calls the sweep could not follow are findings too: an *uncalled
logger attribute* (``meth = _log.info``, ``functools.partial(_log.info,
...)``), a *getattr on a logger* (``getattr(_log, level)``), and a
*logger used as a value*: a receiver, or a call that builds a logger,
anywhere but as the base of an attribute, the value of an assignment or
a discarded expression statement (passed to a function, returned, stored
in a container), because the code that receives it does not know it is a
logger. K1 fails on any of these. K3 and K4 anchor the sweep to calls it
must find, and K7-K9 feed it synthetic sources: each sweep mutant in
BR-026's battery (M11-M18, M21-M27) turned at least one of them red, M16
on CPython 3.11 only (BR-026 evidence).

**The processor-owned set** (:data:`PROCESSOR_OWNED`, BR-027) holds these
28 keys, which structlog 26.1.0's own processors, renderers and logger
methods write into the event, remove from it or read with a meaning of
their own, under their default names (by reading structlog 26.1.0; the
literal maps each key to its writer): ``event`` (the logger methods,
``EventRenamer``, ``ConsoleRenderer``); ``timestamp`` (``TimeStamper``,
``MaybeTimeStamper``); ``level``, ``level_number`` and ``logger``
(``add_log_level``, ``add_log_level_number``, ``add_logger_name``);
``logger_name`` (a ``ConsoleRenderer`` column); ``exception``,
``exc_info``, ``stack`` and ``stack_info`` (``format_exc_info``,
``set_exc_info``, the loggers' ``exception()``, ``StackInfoRenderer``);
``stacklevel`` and ``positional_args`` (``render_to_log_kwargs``,
``render_to_log_args_and_kwargs``, the stdlib ``BoundLogger``,
``PositionalArgumentsFormatter``); ``log_level`` (``capture_logs``);
``_record``, ``_from_structlog``, ``_logger`` and ``_name``
(``ProcessorFormatter``, ``ExtraAdder``); and the eleven
``CallsiteParameterAdder`` keys (``pathname``, ``filename``, ``module``,
``qual_module``, ``func_name``, ``qual_name``, ``lineno``, ``thread``,
``thread_name``, ``process``, ``process_name``). A processor that owns one
replaces, removes or reinterprets the SDK's value, for ``timestamp`` with
no error (for some values of other keys a processor raises instead: under
structlog's default configuration ``ConsoleRenderer`` raises ``TypeError``
for an ``exception`` that is neither a string nor ``None``, unless that
call's ``exc_info`` resolves to an exception (by reading structlog 26.1.0
``dev.py:954-958``; measured for ``False``)): before
BR-027, ``audit.event`` logged the audit event's time as ``timestamp``,
which a ``TimeStamper`` replaced with the log time
(``tests/audit/test_console.py``). The set is a literal, so it can be
checked against its source line by line. K11 runs 18 probes of these
writers on the installed structlog and checks that every key a probe
touches is in the set, that the set holds a floor and every
``CallsiteParameter`` of the installed release, and that the probes
together touch the whole set apart from the keys whose writer the
installed release predates
(:data:`_WRITER_SINCE`: none on 26.1.0; ``qual_name``, ``qual_module`` and
``stacklevel`` on 24.1.0, measured). Eight keys are in both sets
(``exc_info``, ``stack_info``, ``pathname``, ``filename``, ``module``,
``lineno``, ``thread``, ``process``): :data:`RESERVED` holds them because
``makeRecord`` refuses them, this set because a processor owns them, and
the sets stay separate because K5 requires ``makeRecord`` to refuse every
:data:`RESERVED` key, which it does not for ``timestamp``. K10 fails on a
processor-owned key in a swept call and names its writer; K12 anchors the
``audit.event`` call and its five keys; K13 feeds the shared sweep
synthetic sources, with near misses as its control; K14 passes every
swept key through structlog's key-writing processors and
``render_to_log_kwargs``, independently of the literal, as K2 does for
:data:`RESERVED`.

What this does NOT pin: keys a host's own processor adds (six of the keys
structlog 26.1.0's ``CallsiteParameterAdder`` can add are record
attributes, and with it before ``render_to_log_kwargs`` the registry's
overwrite warning still raises; BR-026 evidence, P5), attributes a
host's ``LogRecord`` factory adds, and keys a host binds with ``bind_contextvars``; a logger
the module neither builds with a call ending in ``get_logger`` or
``wrap_logger`` nor takes from one of its own receivers (a host factory's
logger, or one received as a parameter) when it is used without a
logging-style call, for example ``getattr(logger, level)(...)`` or a
stored ``logger.info``; and code the AST cannot read, such as ``eval``.
For the processor-owned set: a name a host chooses for a configurable key
(``TimeStamper(key=...)``, ``MaybeTimeStamper(key=...)``,
``EventRenamer(to=..., replace_by=...)``, ``ConsoleRenderer(timestamp_key=...,
event_key=...)``); the other keys ``ExtraAdder`` (without an ``allow``
list) copies from a stdlib record: a host's own ``extra`` keys, and
``message`` and ``asctime`` once an earlier formatter set them (measured,
BR-027 evidence, P0; :data:`RESERVED` holds both); processors a
host writes or another package ships; and ``structlog.twisted``. K14
reports 21 of the 28 keys on structlog 26.1.0; the other seven rest on
K10 (K14's docstring lists them).
The behavioural pins are ``tests/loop/test_loop_stdlib_logging.py``,
``tests/tools/test_tools_stdlib_logging.py`` and, for BR-027,
``tests/audit/test_console.py`` and ``tests/runner/test_runner_audit.py``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import re
import sys
import textwrap
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Final

import pytest
import structlog
from structlog.dev import ConsoleRenderer, set_exc_info
from structlog.processors import (
    CallsiteParameter,
    CallsiteParameterAdder,
    MaybeTimeStamper,
    StackInfoRenderer,
    TimeStamper,
    format_exc_info,
)
from structlog.stdlib import (
    ExtraAdder,
    PositionalArgumentsFormatter,
    ProcessorFormatter,
    add_log_level,
    add_log_level_number,
    add_logger_name,
    render_to_log_kwargs,
)

import fifty_agent_sdk

# --- The reserved set ---------------------------------------------------------------------


def _record_attributes() -> frozenset[str]:
    """The attributes ``logging.LogRecord.__init__`` sets on this interpreter."""
    return frozenset(vars(logging.LogRecord("probe", logging.INFO, __file__, 1, "m", None, None)))


_RUNTIME_RECORD_KEYS: Final = _record_attributes()
_REFUSED_BY_NAME: Final = frozenset({"message", "asctime"})
_ADDED_IN_LATER_PYTHONS: Final = frozenset({"taskName"})  # LogRecord.taskName, CPython 3.12+
RESERVED: Final = _RUNTIME_RECORD_KEYS | _REFUSED_BY_NAME | _ADDED_IN_LATER_PYTHONS

# --- The processor-owned set (BR-027) -----------------------------------------------------

# The keys structlog 26.1.0's own processors, renderers and logger methods write into the event,
# remove from it or read with a meaning of their own, under their default names, each with its
# writer (by reading structlog 26.1.0; BR-027 plan section 1.2). A literal, not derived at
# runtime; K11 runs each writer against the running structlog.
PROCESSOR_OWNED: Final[dict[str, str]] = {
    "event": "the logger method's event argument; EventRenamer; ConsoleRenderer column",
    "timestamp": "TimeStamper, MaybeTimeStamper; ConsoleRenderer column",
    "level": "add_log_level; ConsoleRenderer column",
    "level_number": "add_log_level_number",
    "logger": "add_logger_name; ConsoleRenderer column",
    "logger_name": "ConsoleRenderer column",
    "exception": "format_exc_info, dict_tracebacks (ExceptionRenderer); ConsoleRenderer",
    "exc_info": "set_exc_info, the loggers' exception(); ExceptionRenderer, render_to_log_kwargs",
    "stack": "StackInfoRenderer; ConsoleRenderer",
    "stack_info": "StackInfoRenderer, render_to_log_kwargs",
    "stacklevel": "render_to_log_kwargs, render_to_log_args_and_kwargs",
    "positional_args": "stdlib BoundLogger; PositionalArgumentsFormatter",
    "log_level": "capture_logs (LogCapture)",
    "_record": "ProcessorFormatter",
    "_from_structlog": "ProcessorFormatter",
    "_logger": "ProcessorFormatter.wrap_for_formatter, ExtraAdder",
    "_name": "ProcessorFormatter.wrap_for_formatter, ExtraAdder",
    "pathname": "CallsiteParameterAdder",
    "filename": "CallsiteParameterAdder",
    "module": "CallsiteParameterAdder",
    "qual_module": "CallsiteParameterAdder",
    "func_name": "CallsiteParameterAdder",
    "qual_name": "CallsiteParameterAdder",
    "lineno": "CallsiteParameterAdder",
    "thread": "CallsiteParameterAdder",
    "thread_name": "CallsiteParameterAdder",
    "process": "CallsiteParameterAdder",
    "process_name": "CallsiteParameterAdder",
}

# --- The sweep ----------------------------------------------------------------------------

_SYNC_LOG_METHODS: Final = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "msg", "log"}
)
_LOG_METHODS: Final = _SYNC_LOG_METHODS | {"a" + name for name in _SYNC_LOG_METHODS}
_BINDERS: Final = frozenset({"bind", "new"})
_KEYLESS_METHODS: Final = frozenset({"unbind", "try_unbind"})
_CONTEXTVAR_BINDERS: Final = ("bind_contextvars", "bound_contextvars")
_LOGGER_FACTORIES: Final = ("get_logger", "wrap_logger")
# ``wrap_logger``'s own parameters; every other keyword is an initial value (structlog 26.1.0
# and 24.1.0 ``_config.py``, by reading). ``get_logger`` passes its keywords on to
# ``wrap_logger``, so none of these is an initial value there (four become its arguments;
# ``logger`` and ``logger_factory_args`` raise ``TypeError``); the sweep still reads every
# ``get_logger`` keyword (it can over-report, never miss one).
_WRAP_LOGGER_PARAMETERS: Final = frozenset(
    {
        "logger",
        "processors",
        "wrapper_class",
        "context_class",
        "cache_logger_on_first_use",
        "logger_factory_args",
    }
)
_STDLIB_FACTORY: Final = "getLogger"


@dataclass(frozen=True)
class LogCall:
    """One swept call: where it is, what it is called on, its event and its keys with their lines."""

    path: str
    line: int
    receiver: str
    method: str
    event: str | None
    keys: tuple[tuple[str, int], ...]

    @property
    def key_names(self) -> set[str]:
        return {key for key, _ in self.keys}


@dataclass
class ModuleSweep:
    """The receivers, swept calls and fail-closed findings of one source file."""

    path: str
    receivers: set[str] = field(default_factory=set)
    calls: list[LogCall] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)


def _dotted(node: ast.expr) -> str | None:
    return ast.unparse(node) if isinstance(node, (ast.Name, ast.Attribute)) else None


def _builds_logger(node: ast.expr, receivers: set[str]) -> bool:
    """A call to a logger factory, or ``bind``/``new`` on a logger."""
    if not isinstance(node, ast.Call):
        return False
    name = _dotted(node.func)
    if name is not None and name.endswith(_LOGGER_FACTORIES):
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in _BINDERS
        and _is_logger(node.func.value, receivers)
    )


def _is_logger(node: ast.expr, receivers: set[str]) -> bool:
    name = _dotted(node)
    return (name is not None and name in receivers) or _builds_logger(node, receivers)


def _receivers(tree: ast.Module) -> set[str]:
    """Targets assigned a logger, to a fixpoint (``log = _log.bind(...)`` or ``log2 = _log`` after ``_log = get_logger()``)."""
    assignments: list[tuple[list[ast.expr], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            assignments.append((node.targets, node.value))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            assignments.append(([node.target], node.value))
    receivers: set[str] = set()
    changed = True
    while changed:
        changed = False
        for targets, value in assignments:
            if not _is_logger(value, receivers):
                continue
            for target in targets:
                name = ast.unparse(target)
                if name not in receivers:
                    receivers.add(name)
                    changed = True
    return receivers


def _swept_call(
    node: ast.Call, path: str, receiver: str, method: str, findings: list[str]
) -> LogCall:
    keys: list[tuple[str, int]] = []
    for keyword in node.keywords:
        if keyword.arg is not None:
            keys.append((keyword.arg, keyword.lineno))
            continue
        value = keyword.value
        if isinstance(value, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys
        ):
            keys.extend((k.value, k.lineno) for k in value.keys if isinstance(k, ast.Constant))
        else:
            findings.append(f"{path}:{keyword.lineno} unverifiable: **{ast.unparse(value)}")
    position = 1 if method in ("log", "alog") else 0
    event_node = node.args[position] if len(node.args) > position else None
    event = (
        event_node.value
        if isinstance(event_node, ast.Constant) and isinstance(event_node.value, str)
        else None
    )
    return LogCall(path, node.lineno, receiver, method, event, tuple(keys))


def sweep_source(source: str, path: str) -> ModuleSweep:
    """Sweep one module's source (pure: K7-K9 feed it synthetic sources)."""
    tree = ast.parse(source)
    result = ModuleSweep(path, receivers=_receivers(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name is not None and name.endswith(_STDLIB_FACTORY):
            result.findings.append(f"{path}:{node.lineno} stdlib logger: {name}")
            continue
        if name is not None and name.endswith(_CONTEXTVAR_BINDERS):
            method = name.rsplit(".", 1)[-1]
            result.calls.append(_swept_call(node, path, name, method, result.findings))
            continue
        if name is not None and name.endswith(_LOGGER_FACTORIES):
            # BR-027: keywords to a logger factory are initial values on every line the logger
            # writes. Before the ``ast.Attribute`` filter, so a bare ``get_logger(...)`` counts.
            if node.keywords:
                method = name.rsplit(".", 1)[-1]
                call = _swept_call(node, path, name, method, result.findings)
                own = _WRAP_LOGGER_PARAMETERS if method == "wrap_logger" else frozenset()
                keys = tuple((key, line) for key, line in call.keys if key not in own)
                result.calls.append(replace(call, event=None, keys=keys))
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        receiver = node.func.value
        if _is_logger(receiver, result.receivers):
            if method in _LOG_METHODS or method in _BINDERS:
                result.calls.append(
                    _swept_call(node, path, ast.unparse(receiver), method, result.findings)
                )
            elif method not in _KEYLESS_METHODS or node.keywords:
                result.findings.append(
                    f"{path}:{node.lineno} unrecognised method: {ast.unparse(node.func)}"
                )
            continue
        first_is_text = (
            bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        )
        looks_like_logging = (
            method in _LOG_METHODS and (first_is_text or bool(node.keywords))
        ) or (method in _BINDERS and bool(node.keywords))
        if looks_like_logging and ast.unparse(receiver) != "warnings":
            result.findings.append(
                f"{path}:{node.lineno} unrecognised receiver: {ast.unparse(node.func)}"
            )
    result.findings.extend(_uses_the_sweep_cannot_follow(tree, path, result.receivers))
    return result


# A logger may be the value of these without leaving the sweep's sight: an assignment makes a
# new receiver, and an expression statement discards it.
_BINDING_PARENTS: Final = (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.Expr)


def _uses_the_sweep_cannot_follow(tree: ast.Module, path: str, receivers: set[str]) -> list[str]:
    """Logger uses whose later calls the sweep could not read: it reports them instead.

    * an *uncalled logger attribute*: ``meth = _log.info``, or
      ``functools.partial(_log.info, ...)``;
    * a *getattr on a logger*: ``getattr(_log, level)``;
    * a *logger used as a value*: a receiver, or a call that builds a
      logger, anywhere but as the base of an attribute or the value of an
      assignment (passed to a function, returned, stored in a container,
      aliased by ``log2 = _log``), because the code that receives it does
      not know it is a logger.
    """
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    findings: list[str] = []
    for node in ast.walk(tree):
        parent = parents.get(id(node))
        if (
            isinstance(node, ast.Attribute)
            and id(node) not in called
            and _is_logger(node.value, receivers)
        ):
            findings.append(f"{path}:{node.lineno} uncalled logger attribute: {ast.unparse(node)}")
            continue
        if isinstance(node, ast.Call) and _dotted(node.func) == "getattr":
            if node.args and _is_logger(node.args[0], receivers):
                findings.append(f"{path}:{node.lineno} getattr on a logger: {ast.unparse(node)}")
            continue
        is_receiver = (
            isinstance(node, (ast.Name, ast.Attribute))
            and isinstance(node.ctx, ast.Load)
            and ast.unparse(node) in receivers
        )
        if not (is_receiver or (isinstance(node, ast.expr) and _builds_logger(node, receivers))):
            continue
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue  # ``_log.info``, ``_log.bind(...).info``: checked as calls or attributes
        if isinstance(parent, _BINDING_PARENTS) and parent.value is node:
            continue  # ``log = _log.bind(...)``: a new receiver; ``_log.bind(x=1)``: discarded
        if (
            isinstance(parent, ast.Call)
            and _dotted(parent.func) == "getattr"
            and parent.args
            and parent.args[0] is node
        ):
            continue  # reported as a getattr on a logger
        findings.append(f"{path}:{node.lineno} logger used as a value: {ast.unparse(node)}")
    return findings


def sweep_package(package_dir: Path) -> list[ModuleSweep]:
    """Sweep every ``.py`` file under ``package_dir``; paths relative to its parent."""
    return [
        sweep_source(
            path.read_text(encoding="utf-8"), path.relative_to(package_dir.parent).as_posix()
        )
        for path in sorted(package_dir.rglob("*.py"))
    ]


def reserved_key_findings(calls: list[LogCall], reserved: frozenset[str] = RESERVED) -> list[str]:
    """``path:line event key`` for every reserved key a swept call passes."""
    return [
        f"{call.path}:{line} {call.event or call.method} {key}"
        for call in calls
        for key, line in call.keys
        if key in reserved
    ]


def processor_owned_findings(calls: list[LogCall]) -> list[str]:
    """``path:line event key (writer)`` for every :data:`PROCESSOR_OWNED` key a swept call passes."""
    return [
        f"{finding} ({PROCESSOR_OWNED[finding.rsplit(' ', 1)[1]]})"
        for finding in reserved_key_findings(calls, frozenset(PROCESSOR_OWNED))
    ]


_PACKAGE_DIR: Final = Path(fifty_agent_sdk.__file__).resolve().parent


@pytest.fixture(scope="module")
def package_sweep() -> list[ModuleSweep]:
    return sweep_package(_PACKAGE_DIR)


def _all_calls(sweeps: list[ModuleSweep]) -> list[LogCall]:
    return [call for sweep in sweeps for call in sweep.calls]


# --- K1/K2: the package -------------------------------------------------------------------


def test_no_sdk_log_call_passes_a_key_stdlib_logging_reserves(
    package_sweep: list[ModuleSweep],
) -> None:
    """K1: no swept call passes a reserved key, and nothing failed closed (BR-026, AC-1).

    Before BR-026 this listed the five keywords the brief renamed (``name``
    in both ``tool_invoked`` calls, ``tool overwritten`` and
    ``mcp.tool_overwrite``; ``message`` in ``mcp.refresh_failed``).
    """
    offenders = reserved_key_findings(_all_calls(package_sweep))
    findings = [finding for sweep in package_sweep for finding in sweep.findings]
    assert offenders == [], "logging keys stdlib refuses as extra:\n" + "\n".join(offenders)
    assert findings == [], "calls the sweep cannot verify:\n" + "\n".join(findings)


def test_every_swept_key_is_accepted_by_make_record(package_sweep: list[ModuleSweep]) -> None:
    """K2: stdlib itself accepts every key the package's logging calls pass, on this interpreter.

    Independent of :data:`RESERVED`'s derivation: each key goes through a
    real ``Logger.makeRecord(..., extra={key: None})``. Before BR-026 it
    refused ``message`` and ``name``.
    """
    keys = sorted({key for call in _all_calls(package_sweep) for key in call.key_names})
    assert keys, "the sweep collected no keys"
    logger = logging.Logger("probe")
    refused = []
    for key in keys:
        try:
            logger.makeRecord("probe", logging.INFO, __file__, 1, "m", (), None, extra={key: None})
        except KeyError as exc:
            refused.append(f"{key}: {exc}")
    assert refused == []


# --- K3/K4: anchors -----------------------------------------------------------------------


def _calls(sweeps: list[ModuleSweep], path: str, event: str | None, method: str) -> list[LogCall]:
    return [
        call
        for call in _all_calls(sweeps)
        if call.path == path and call.event == event and call.method == method
    ]


def test_sweep_finds_the_calls_this_brief_renamed_and_the_bound_logger_calls(
    package_sweep: list[ModuleSweep],
) -> None:
    """K3: the sweep reaches the five calls BR-026 changed, the bound logger ``log`` and both ``bind`` calls.

    Anchored by event, not line. Each renamed call is checked through a key
    BR-026 did not rename, so a sweep that reads no keywords from a
    multi-line call fails here as well as in K7.
    """
    invoked = _calls(package_sweep, "fifty_agent_sdk/loop.py", "tool_invoked", "debug")
    assert len(invoked) == 2
    assert all({"call_id", "run_id"} <= call.key_names and len(call.keys) == 3 for call in invoked)
    (overwritten,) = _calls(
        package_sweep, "fifty_agent_sdk/tools/registry.py", "tool overwritten", "warning"
    )
    assert len(overwritten.keys) == 1
    (mcp_overwrite,) = _calls(
        package_sweep, "fifty_agent_sdk/tools/mcp_provider.py", "mcp.tool_overwrite", "warning"
    )
    assert "reason" in mcp_overwrite.key_names and len(mcp_overwrite.keys) == 2
    (refresh_failed,) = _calls(
        package_sweep, "fifty_agent_sdk/tools/mcp_provider.py", "mcp.refresh_failed", "warning"
    )
    assert "wrapped" in refresh_failed.key_names and len(refresh_failed.keys) == 2

    call_ok = _calls(package_sweep, "fifty_agent_sdk/mcp/client.py", "mcp.call_ok", "debug")
    assert [call.receiver for call in call_ok] == ["log", "log"]
    binds = _calls(package_sweep, "fifty_agent_sdk/mcp/client.py", None, "bind")
    assert [call.receiver for call in binds] == ["_log", "_log"]
    assert all(call.key_names == {"method", "server_url"} for call in binds)


_LOGGER_MODULES: Final = frozenset(
    {
        "fifty_agent_sdk/audit/console.py",
        "fifty_agent_sdk/audit/sql.py",
        "fifty_agent_sdk/interventions.py",
        "fifty_agent_sdk/loop.py",
        "fifty_agent_sdk/mcp/client.py",
        "fifty_agent_sdk/observability/hooks.py",
        "fifty_agent_sdk/runner.py",
        "fifty_agent_sdk/state/redis.py",
        "fifty_agent_sdk/state/sql.py",
        "fifty_agent_sdk/tools/mcp_provider.py",
        "fifty_agent_sdk/tools/registry.py",
    }
)


def test_every_module_with_a_logger_has_swept_calls(package_sweep: list[ModuleSweep]) -> None:
    """K4: the 11 modules that assign a structlog logger are found, and each yields a swept call."""
    with_receivers = {sweep.path: sweep for sweep in package_sweep if sweep.receivers}
    assert set(with_receivers) >= _LOGGER_MODULES
    empty = sorted(path for path, sweep in with_receivers.items() if not sweep.calls)
    assert empty == []


# --- K5/K6: the reserved set --------------------------------------------------------------


def test_reserved_set_is_what_make_record_refuses() -> None:
    """K5: every key in :data:`RESERVED` is refused by ``makeRecord`` here, except a later Python's attribute.

    The floor makes a broken derivation fail instead of passing an empty
    set. A key from :data:`_ADDED_IN_LATER_PYTHONS` that this interpreter's
    records lack (``taskName`` on 3.11) is accepted, which is why the union
    exists.
    """
    assert {"name", "msg", "args", "levelname", "message", "asctime"} <= RESERVED
    logger = logging.Logger("probe")
    for key in sorted(RESERVED):
        if key in _ADDED_IN_LATER_PYTHONS - _RUNTIME_RECORD_KEYS:
            record = logger.makeRecord(
                "probe", logging.INFO, __file__, 1, "m", (), None, extra={key: None}
            )
            assert key in record.__dict__
            continue
        with pytest.raises(KeyError, match=f"Attempt to overwrite '{key}' in LogRecord"):
            logger.makeRecord("probe", logging.INFO, __file__, 1, "m", (), None, extra={key: None})


def test_reserved_set_includes_task_name_on_every_interpreter() -> None:
    """K6: ``taskName`` is reserved everywhere; it is a record attribute from CPython 3.12 on."""
    assert "taskName" in RESERVED
    if sys.version_info >= (3, 12):
        assert "taskName" in _RUNTIME_RECORD_KEYS
    else:
        assert "taskName" not in _RUNTIME_RECORD_KEYS


# --- K7-K9: the sweep on synthetic sources ------------------------------------------------

_PRELUDE: Final = "import logging\nimport structlog\n_log = structlog.get_logger(__name__)\n"


def _sweep(body: str) -> ModuleSweep:
    return sweep_source(_PRELUDE + textwrap.dedent(body), "probe.py")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            """
            _log.info(
                "e",
                name=1,
            )
            """,
            ["probe.py:7 e name"],
            id="multi_line_call",
        ),
        pytest.param(
            'log = _log.bind(a=1)\nlog.info("e", name=1)\n',
            ["probe.py:5 e name"],
            id="bound_logger",
        ),
        pytest.param("_log.bind(levelname=1)\n", ["probe.py:4 bind levelname"], id="bind_key"),
        pytest.param("_log.new(funcName=1)\n", ["probe.py:4 new funcName"], id="new_key"),
        pytest.param(
            '_log.bind(x=1).warning("e", msg=1)\n', ["probe.py:4 e msg"], id="chained_bind"
        ),
        pytest.param(
            """
            class C:
                def __init__(self):
                    self._log = structlog.get_logger()

                def f(self):
                    self._log.debug("e", module=1)
            """,
            ["probe.py:10 e module"],
            id="attribute_receiver",
        ),
        pytest.param(
            'async def f():\n    await _log.ainfo("e", lineno=1)\n',
            ["probe.py:5 e lineno"],
            id="async_method",
        ),
        pytest.param(
            '_log.log(logging.INFO, "e", taskName=1)\n', ["probe.py:4 e taskName"], id="log_method"
        ),
        pytest.param(
            'structlog.get_logger().error("e", filename=1)\n',
            ["probe.py:4 e filename"],
            id="factory_chain",
        ),
        pytest.param(
            "structlog.contextvars.bind_contextvars(process=1)\n",
            ["probe.py:4 bind_contextvars process"],
            id="bind_contextvars",
        ),
        pytest.param(
            "with structlog.contextvars.bound_contextvars(thread=1):\n    pass\n",
            ["probe.py:4 bound_contextvars thread"],
            id="bound_contextvars",
        ),
        pytest.param(
            '_log.info("e", **{"asctime": 1})\n', ["probe.py:4 e asctime"], id="dict_splat"
        ),
        pytest.param(
            'log2 = _log\nlog2.info("e", name=1)\n', ["probe.py:5 e name"], id="aliased_logger"
        ),
        pytest.param(
            "structlog.get_logger(name=1)\n",
            ["probe.py:4 get_logger name"],
            id="get_logger_initial_value",
        ),
        pytest.param(
            'get_logger("x", levelname=1)\n',
            ["probe.py:4 get_logger levelname"],
            id="bare_get_logger_initial_value",
        ),
        pytest.param(
            "structlog.wrap_logger(None, processors=[], msg=1)\n",
            ["probe.py:4 wrap_logger msg"],
            id="wrap_logger_initial_value",
        ),
    ],
)
def test_sweep_reports_reserved_keys_in_every_call_shape(body: str, expected: list[str]) -> None:
    """K7: the sweep reports a reserved key in each call shape it claims to read."""
    sweep = _sweep(body)
    assert reserved_key_findings(sweep.calls) == expected
    assert sweep.findings == []


@pytest.mark.parametrize(
    ("body", "kind"),
    [
        pytest.param('_log.info("e", **kwargs)\n', "unverifiable", id="splat_of_a_name"),
        pytest.param(
            '_log.info("e", **{"a": 1, **rest})\n', "unverifiable", id="splat_inside_a_literal"
        ),
        pytest.param('other.info("e", ok=1)\n', "unrecognised receiver", id="unknown_receiver"),
        pytest.param("other.bind(ok=1)\n", "unrecognised receiver", id="unknown_bind"),
        pytest.param('_log.custom("e", ok=1)\n', "unrecognised method", id="unknown_method"),
        pytest.param('logging.getLogger("x")\n', "stdlib logger", id="stdlib_logger"),
        pytest.param("meth = _log.info\n", "uncalled logger attribute", id="stored_method"),
        pytest.param(
            'functools.partial(_log.info, "e")\n', "uncalled logger attribute", id="partial"
        ),
        pytest.param(
            "log = _log.bind(a=1); meth = log.warning\n",
            "uncalled logger attribute",
            id="stored_bound_method",
        ),
        pytest.param('getattr(_log, "info")("e", name=1)\n', "getattr on a logger", id="getattr"),
        pytest.param(
            'getattr(_log.bind(a=1), "info")\n', "getattr on a logger", id="getattr_bound"
        ),
        pytest.param("helper(_log)\n", "logger used as a value", id="passed_to_a_call"),
        pytest.param("loggers = [_log]\n", "logger used as a value", id="stored_in_a_list"),
        pytest.param("f = lambda: _log\n", "logger used as a value", id="returned"),
        pytest.param(
            "helper(structlog.get_logger())\n", "logger used as a value", id="built_and_passed"
        ),
        pytest.param(
            "structlog.get_logger(**kw)\n", "unverifiable", id="get_logger_splat_of_a_name"
        ),
    ],
)
def test_sweep_fails_closed_on_calls_it_cannot_verify(body: str, kind: str) -> None:
    """K8: a call whose keys the sweep cannot read, whose receiver it does not know, or a logger use it cannot follow, is a finding."""
    findings = _sweep(body).findings
    assert len(findings) == 1
    assert findings[0].startswith("probe.py:4 " + kind + ":")


def test_sweep_ignores_near_miss_keys_and_non_logger_calls() -> None:
    """K9 (control): case-sensitive near misses and calls that are not logging are not reported."""
    sweep = _sweep(
        """
        import math
        import warnings
        _log.info("e", task_name=1, processname=1, Name=1, name_=1, message_id=1)
        math.log(2.0)
        warnings.warn("w")
        obj.info()
        _log.unbind("a")
        x = result.error
        y = getattr(obj, "info")
        z = obj.info
        log3 = _log
        _log.bind(discarded=1)
        """
    )
    assert reserved_key_findings(sweep.calls) == []
    assert sweep.findings == []
    (call,) = [call for call in sweep.calls if call.event == "e"]
    assert call.key_names == {"task_name", "processname", "Name", "name_", "message_id"}


# --- K10-K14: the processor-owned set (BR-027) --------------------------------------------


def test_no_sdk_log_call_passes_a_key_a_structlog_processor_owns(
    package_sweep: list[ModuleSweep],
) -> None:
    """K10: no swept call passes a key in :data:`PROCESSOR_OWNED` (BR-027, AC-2).

    Before BR-027 this listed one key: ``timestamp`` in ``audit.event``
    (``fifty_agent_sdk/audit/console.py:64``). The sweep is K1's, and K1
    reports what it cannot read.
    """
    offenders = processor_owned_findings(_all_calls(package_sweep))
    assert offenders == [], "logging keys a structlog processor owns:\n" + "\n".join(offenders)


# K11's writer rows. Each runs a real structlog object and returns the keys it wrote, removed,
# replaced or treated as its own, or ``None`` when this structlog release lacks the writer.

_PROBE_LOGGER: Final = logging.Logger("probe")  # not registered with stdlib's manager


def _touched(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """Keys a processor added, removed or replaced (compared by identity)."""
    return {
        key
        for key in before.keys() | after.keys()
        if key not in before or key not in after or after[key] is not before[key]
    }


def _run(
    processor: Callable[..., Any], event_dict: dict[str, Any], method: str = "info"
) -> set[str]:
    return _touched(event_dict, processor(_PROBE_LOGGER, method, dict(event_dict)))


def _logged(log: Callable[[Any], object], *, stdlib: bool = False) -> dict[str, Any]:
    """The event dict a logger method builds, before any processor changes it."""
    captured: list[dict[str, Any]] = []

    def capture(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        captured.append(dict(event_dict))
        raise structlog.DropEvent

    logger: Any
    if stdlib:
        stdlib_logger = logging.Logger("probe")
        stdlib_logger.setLevel(logging.DEBUG)
        logger = structlog.stdlib.BoundLogger(stdlib_logger, [capture], {})
    else:
        wrapper = structlog.make_filtering_bound_logger(logging.DEBUG)
        logger = wrapper(structlog.PrintLogger(io.StringIO()), [capture], {})
    log(logger)
    (event_dict,) = captured
    return event_dict


def _exc_info_tuple() -> Any:
    try:
        raise ValueError("probe")
    except ValueError:
        return sys.exc_info()


def _row_exception_methods() -> set[str]:
    native = _logged(lambda log: log.exception("e"))
    via_stdlib = _logged(lambda log: log.exception("e"), stdlib=True)
    return (set(native) | set(via_stdlib)) - {"event"}


def _row_render_to_log_kwargs() -> set[str]:
    sent = {"event": "e", "exc_info": object(), "stack_info": object(), "stacklevel": object()}
    out = render_to_log_kwargs(_PROBE_LOGGER, "info", dict(sent))
    return set(sent) - set(out["extra"])


def _row_render_to_log_args_and_kwargs() -> set[str] | None:
    render = getattr(structlog.stdlib, "render_to_log_args_and_kwargs", None)
    if render is None:  # added in structlog 25.1.0
        return None
    sent = {
        "event": "e",
        "positional_args": (),
        "exc_info": object(),
        "stack_info": object(),
        "stacklevel": object(),
    }
    _, kwargs = render(_PROBE_LOGGER, "info", dict(sent))
    return set(sent) - set(kwargs.get("extra", {}))


def _row_positional_arguments() -> set[str]:
    logged = _logged(lambda log: log.info("e %s", "x"), stdlib=True)
    formatted = PositionalArgumentsFormatter()(_PROBE_LOGGER, "info", dict(logged))
    return (set(logged) - {"event"}) | _touched(logged, formatted)


def _row_log_capture() -> set[str]:
    capture = structlog.testing.LogCapture()
    with contextlib.suppress(structlog.DropEvent):
        capture(_PROBE_LOGGER, "info", {"event": "e"})
    (entry,) = capture.entries
    return set(entry) - {"event"}


def _row_processor_formatter() -> set[str]:
    """The keys ``wrap_for_formatter`` passes as ``extra``, and those ``ProcessorFormatter`` and ``ExtraAdder`` add."""
    seen: list[set[str]] = []

    def grab(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        seen.append(set(event_dict))
        return event_dict

    args, kwargs = ProcessorFormatter.wrap_for_formatter(_PROBE_LOGGER, "info", {"event": "e"})
    record = logging.makeLogRecord({"msg": args[0], **kwargs["extra"]})
    ProcessorFormatter(processors=[grab, ExtraAdder(), grab, lambda *_: "x"]).format(record)
    before_extra_adder, after_extra_adder = seen
    return set(kwargs["extra"]) | (before_extra_adder - {"event"}) | after_extra_adder


def _row_console_renderer_columns() -> set[str] | None:
    columns = getattr(ConsoleRenderer(colors=False), "columns", None)
    if columns is None:  # structlog 24.1.0 keeps them private
        return None
    return {column.key for column in columns} - {""}  # "" is the column for the other keys


_RENDER_CONTROLS: Final = ("payload", "event_timestamp", "levels", "Timestamp")


def _row_console_renderer_output() -> set[str]:
    """Keys ``ConsoleRenderer`` writes as a column or removes, not as ``key=value``."""
    owned = set()
    for key in [*PROCESSOR_OWNED, *_RENDER_CONTROLS]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            line = ConsoleRenderer(colors=False)(_PROBE_LOGGER, "info", {"event": "e", key: "v"})
        if f" {key}=" not in line:
            owned.add(key)
    return owned


def _row_bound_event() -> set[str]:
    """A logger method sets ``event`` after applying the bound context, so a bound ``event`` is replaced."""
    logged = _logged(lambda log: log.bind(event="bound").info("call"))
    return {"event"} if logged["event"] == "call" else set()


_WRITER_ROWS: Final[dict[str, Callable[[], set[str] | None]]] = {
    "TimeStamper": lambda: _run(TimeStamper(), {"event": "e"}),
    "MaybeTimeStamper": lambda: _run(MaybeTimeStamper(), {"event": "e"}),
    "add_log_level": lambda: _run(add_log_level, {"event": "e"}),
    "add_log_level_number": lambda: _run(add_log_level_number, {"event": "e"}),
    "add_logger_name": lambda: _run(add_logger_name, {"event": "e"}),
    "format_exc_info": lambda: _run(format_exc_info, {"event": "e", "exc_info": _exc_info_tuple()}),
    "StackInfoRenderer": lambda: _run(StackInfoRenderer(), {"event": "e", "stack_info": True}),
    "set_exc_info": lambda: _run(set_exc_info, {"event": "e"}, "exception"),
    "exception_methods": _row_exception_methods,
    "CallsiteParameterAdder": lambda: _run(CallsiteParameterAdder(), {"event": "e"}),
    "render_to_log_kwargs": _row_render_to_log_kwargs,
    "render_to_log_args_and_kwargs": _row_render_to_log_args_and_kwargs,
    "PositionalArgumentsFormatter": _row_positional_arguments,
    "LogCapture": _row_log_capture,
    "ProcessorFormatter": _row_processor_formatter,
    "ConsoleRenderer_columns": _row_console_renderer_columns,
    "ConsoleRenderer_output": _row_console_renderer_output,
    "bound_event": _row_bound_event,
}

# Keys whose earliest writer among the rows arrived after structlog 24.1.0, the floor
# (``pyproject.toml``): ``CallsiteParameter.QUAL_NAME`` in 25.5.0 and ``QUAL_MODULE`` in 26.1.0,
# and ``render_to_log_kwargs`` taking ``stacklevel`` out of ``extra`` in 24.2.0 (``stackLevel``
# before). By reading structlog 26.1.0's version notes; measured on 24.1.0 (BR-027 evidence, P0).
_WRITER_SINCE: Final = {"qual_name": (25, 5), "qual_module": (26, 1), "stacklevel": (24, 2)}


def _structlog_version() -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", metadata.version("structlog"))
    assert match is not None
    return int(match[1]), int(match[2])


def _keys_without_a_writer_here() -> set[str]:
    return {key for key, since in _WRITER_SINCE.items() if _structlog_version() < since}


@pytest.mark.parametrize("row", list(_WRITER_ROWS))
def test_processor_owned_set_is_what_structlog_writes(row: str) -> None:
    """K11: each probe runs a real structlog writer here and touches only keys in :data:`PROCESSOR_OWNED` (BR-027)."""
    keys = _WRITER_ROWS[row]()
    if keys is None:
        pytest.skip(f"structlog {metadata.version('structlog')} has no {row}")
    assert keys, f"{row} touched no key"
    assert keys <= PROCESSOR_OWNED.keys(), sorted(keys - PROCESSOR_OWNED.keys())


def test_processor_owned_set_is_covered_and_complete() -> None:
    """K11: a floor, every ``CallsiteParameter`` of this structlog, and the set equals what the rows touch.

    Exactness allows only the keys of :data:`_WRITER_SINCE` whose writer
    this structlog release predates; on structlog 26.1.0 that is none.
    """
    floor = {"event", "timestamp", "level", "logger", "exception", "exc_info", "stack"}
    assert floor <= PROCESSOR_OWNED.keys()
    assert {parameter.value for parameter in CallsiteParameter} <= PROCESSOR_OWNED.keys()
    seen: set[str] = set()
    for probe in _WRITER_ROWS.values():
        seen |= probe() or set()
    assert seen <= PROCESSOR_OWNED.keys()
    assert PROCESSOR_OWNED.keys() - seen == _keys_without_a_writer_here()


def test_sweep_finds_the_audit_event_call_and_its_keys(package_sweep: list[ModuleSweep]) -> None:
    """K12: the sweep reaches ``audit.event`` and reads its five keys, the event's time as ``event_timestamp`` (BR-027)."""
    (call,) = _calls(package_sweep, "fifty_agent_sdk/audit/console.py", "audit.event", "info")
    assert call.key_names == {"session_id", "user_id", "event_type", "event_timestamp", "payload"}
    assert len(call.keys) == 5


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param('_log.info("e", timestamp=1)\n', "probe.py:4 e timestamp", id="log_method"),
        pytest.param("_log.bind(level=1)\n", "probe.py:4 bind level", id="bind_key"),
        pytest.param("_log.new(logger=1)\n", "probe.py:4 new logger", id="new_key"),
        pytest.param(
            "structlog.contextvars.bind_contextvars(func_name=1)\n",
            "probe.py:4 bind_contextvars func_name",
            id="bind_contextvars",
        ),
        pytest.param(
            '_log.info("e", **{"stacklevel": 1})\n', "probe.py:4 e stacklevel", id="dict_splat"
        ),
        pytest.param(
            'log = _log.bind(a=1)\nlog.info("e", log_level=1)\n',
            "probe.py:5 e log_level",
            id="bound_logger",
        ),
        pytest.param('_log.info("e", _record=1)\n', "probe.py:4 e _record", id="formatter_key"),
        pytest.param(
            "structlog.get_logger(exception=1)\n",
            "probe.py:4 get_logger exception",
            id="get_logger_initial_value",
        ),
    ],
)
def test_processor_owned_keys_are_reported_in_every_call_shape(body: str, expected: str) -> None:
    """K13: the shared sweep reports a processor-owned key in each call shape, with its writer (BR-027)."""
    sweep = _sweep(body)
    key = expected.rsplit(" ", 1)[1]
    assert processor_owned_findings(sweep.calls) == [f"{expected} ({PROCESSOR_OWNED[key]})"]
    assert sweep.findings == []


def test_processor_owned_findings_ignore_near_miss_keys() -> None:
    """K13 (control): near misses are not reported; matching is exact and case-sensitive."""
    near_misses = {
        "event_type",
        "event_timestamp",
        "timestamps",
        "Timestamp",
        "levels",
        "logger_id",
        "stack_depth",
        "func",
    }
    sweep = _sweep(f'_log.info("e", {", ".join(f"{key}=1" for key in sorted(near_misses))})\n')
    assert processor_owned_findings(sweep.calls) == []
    assert sweep.findings == []
    (call,) = sweep.calls
    assert call.key_names == near_misses


def _key_writing_chain() -> list[Callable[..., Any]]:
    """structlog's processors that write a key on every call, plus its exception, stack and positional-argument ones."""
    return [
        add_log_level,
        add_log_level_number,
        add_logger_name,
        TimeStamper(),
        CallsiteParameterAdder(),
        StackInfoRenderer(),
        format_exc_info,
        set_exc_info,
        PositionalArgumentsFormatter(),
    ]


def keys_structlog_processors_replace(keys: Iterable[str]) -> list[str]:
    """The keys whose value does not reach ``render_to_log_kwargs``'s ``extra`` as it was passed.

    Each key goes alone, with a fresh sentinel, through
    :func:`_key_writing_chain` for an ``exception`` call and then
    ``render_to_log_kwargs``. A processor that raises counts as replacing it.
    """
    replaced = []
    for key in keys:
        sentinel = object()
        event_dict: Any = {"event": "probe", key: sentinel}
        try:
            for processor in _key_writing_chain():
                event_dict = processor(_PROBE_LOGGER, "exception", event_dict)
            extra = render_to_log_kwargs(_PROBE_LOGGER, "exception", event_dict)["extra"]
            survived = extra.get(key) is sentinel
        except Exception:
            survived = False
        if not survived:
            replaced.append(key)
    return replaced


def test_every_swept_key_survives_structlog_processors(package_sweep: list[ModuleSweep]) -> None:
    """K14: every key the package's logging calls pass survives structlog's key-writing processors (BR-027).

    Independent of :data:`PROCESSOR_OWNED`, like K2 for :data:`RESERVED`.
    ``event`` is left out (K10 covers it). Of the 28 keys in the set it
    reports 21 on structlog 26.1.0 (18 on 24.1.0, which lacks
    ``qual_module``, ``qual_name`` and the ``stacklevel`` pop). It cannot
    see ``logger_name`` (a renderer column), ``exception`` and ``stack``
    (written only when an exception or ``stack_info`` is there),
    ``log_level`` (``capture_logs``), or ``_from_structlog``, ``_logger`` and
    ``_name`` (``ProcessorFormatter`` and ``ExtraAdder`` are not in its
    chain); K10 covers those through the literal (BR-027 evidence, K14
    scope).
    """
    keys = sorted({key for call in _all_calls(package_sweep) for key in call.key_names} - {"event"})
    assert keys, "the sweep collected no keys"
    assert keys_structlog_processors_replace(keys) == []


def test_structlog_processor_cross_check_reports_replaced_keys() -> None:
    """K14 (self-test): the cross-check reports a written key, a call-site key and, from structlog 24.2.0, ``stacklevel``.

    ``render_to_log_kwargs`` takes ``stacklevel`` out of ``extra`` from
    structlog 24.2.0 (``stackLevel`` before; BR-027 evidence, P0).
    """
    expected = ["timestamp", "func_name"]
    if "stacklevel" not in _keys_without_a_writer_here():
        expected.append("stacklevel")
    probe = ["timestamp", "func_name", "stacklevel", "event_timestamp", "event_type"]
    assert keys_structlog_processors_replace(probe) == expected
