"""Tests for the SQL surface's graceful-ImportError behaviour.

The ``sql`` extra (SQLAlchemy) is optional. When it is NOT installed,
attempting to use :class:`SqlStateStore` or :data:`sql_metadata` must
raise an :class:`ImportError` whose message references both the
``sqlalchemy`` package and the ``fifty-agent-sdk[sql]`` extras line. The
in-memory store and the rest of the SDK remain importable.

These tests simulate "missing extra" by patching ``sys.modules`` so the
top-level ``sqlalchemy`` import inside :mod:`fifty_agent_sdk.state.sql` fails,
then reloading the module via :mod:`importlib`. The patch is reverted
in fixture teardown so subsequent tests in the same run are unaffected.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def sqlalchemy_unavailable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend SQLAlchemy is not installed for the duration of this test.

    Removes any already-imported ``sqlalchemy*`` and ``fifty_agent_sdk.state.sql``
    modules from :data:`sys.modules`, and installs a ``None`` entry for
    ``sqlalchemy`` so the next ``import sqlalchemy`` raises
    :class:`ImportError`. Tearing down restores the prior state by
    reloading the real :mod:`fifty_agent_sdk.state.sql` so the rest of the
    test session sees a working SQL surface.
    """
    # Strip every cached sqlalchemy* module + fifty_agent_sdk.state.sql so the
    # next import path goes through our blocked entry. We collect first,
    # then mutate, to avoid mutating during iteration.
    removed: list[str] = [
        name
        for name in list(sys.modules)
        if name == "sqlalchemy"
        or name.startswith("sqlalchemy.")
        or name == "fifty_agent_sdk.state.sql"
    ]
    for name in removed:
        monkeypatch.delitem(sys.modules, name, raising=False)

    # `None` in sys.modules makes `import sqlalchemy` raise ImportError.
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)

    yield
    # monkeypatch will revert sys.modules entries on teardown — at which
    # point the real sqlalchemy is back. We also pop our half-built
    # fifty_agent_sdk.state.sql so the next caller re-imports it cleanly.
    sys.modules.pop("fifty_agent_sdk.state.sql", None)


def test_import_without_sqlalchemy_raises_clear_error(
    sqlalchemy_unavailable: None,
) -> None:
    """Direct ``import fifty_agent_sdk.state.sql`` without SQLAlchemy gives a clear ImportError."""
    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("fifty_agent_sdk.state.sql")
    msg = str(exc_info.value)
    assert "sqlalchemy" in msg.lower()
    assert "fifty-agent-sdk[sql]" in msg


def test_lazy_attribute_access_on_state_package_triggers_extras_hint(
    sqlalchemy_unavailable: None,
) -> None:
    """``fifty_agent_sdk.state.SqlStateStore`` (lazy) raises the same ImportError."""
    # `fifty_agent_sdk.state` itself does not import sqlalchemy, so a fresh import
    # of it must succeed even while sqlalchemy is blocked.
    sys.modules.pop("fifty_agent_sdk.state", None)
    state_pkg = importlib.import_module("fifty_agent_sdk.state")

    with pytest.raises(ImportError) as exc_info:
        _ = state_pkg.SqlStateStore  # triggers __getattr__
    msg = str(exc_info.value)
    assert "sqlalchemy" in msg.lower()
    assert "fifty-agent-sdk[sql]" in msg


def test_top_level_lazy_attribute_access_triggers_extras_hint(
    sqlalchemy_unavailable: None,
) -> None:
    """``fifty_agent_sdk.SqlStateStore`` (top-level lazy) raises the same ImportError."""
    sys.modules.pop("fifty_agent_sdk", None)
    pkg = importlib.import_module("fifty_agent_sdk")

    with pytest.raises(ImportError) as exc_info:
        _ = pkg.SqlStateStore  # triggers top-level __getattr__
    msg = str(exc_info.value)
    assert "sqlalchemy" in msg.lower()
    assert "fifty-agent-sdk[sql]" in msg


def test_lazy_attribute_access_for_sql_metadata_triggers_extras_hint(
    sqlalchemy_unavailable: None,
) -> None:
    """``fifty_agent_sdk.sql_metadata`` lazy access also surfaces the extras hint."""
    sys.modules.pop("fifty_agent_sdk", None)
    pkg = importlib.import_module("fifty_agent_sdk")

    with pytest.raises(ImportError) as exc_info:
        _ = pkg.sql_metadata
    msg = str(exc_info.value)
    assert "sqlalchemy" in msg.lower()
    assert "fifty-agent-sdk[sql]" in msg


def test_state_package_import_still_works_without_extras(
    sqlalchemy_unavailable: None,
) -> None:
    """``import fifty_agent_sdk.state`` succeeds and exposes ``MemoryStateStore``."""
    sys.modules.pop("fifty_agent_sdk.state", None)
    state_pkg = importlib.import_module("fifty_agent_sdk.state")
    # MemoryStateStore must be reachable; only the SQL surface is gated.
    assert state_pkg.MemoryStateStore is not None
    assert state_pkg.StateStore is not None


def test_unknown_attribute_still_raises_attribute_error(
    sqlalchemy_unavailable: None,
) -> None:
    """``__getattr__`` does NOT swallow access to non-SQL attribute names."""
    sys.modules.pop("fifty_agent_sdk.state", None)
    state_pkg = importlib.import_module("fifty_agent_sdk.state")
    with pytest.raises(AttributeError):
        _ = state_pkg.DefinitelyNotASymbol


def test_top_level_unknown_attribute_still_raises_attribute_error() -> None:
    """The top-level package's ``__getattr__`` rejects unknown names."""
    import fifty_agent_sdk

    with pytest.raises(AttributeError):
        _ = fifty_agent_sdk.DefinitelyNotASymbol
