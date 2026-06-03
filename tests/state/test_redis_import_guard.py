"""Tests for the Redis surface's graceful-ImportError behaviour.

The ``redis`` extra (redis-py) is optional. When it is NOT installed,
attempting to use :class:`RedisStateStore` must raise an
:class:`ImportError` whose message references both the ``redis`` package
and the ``agent-sdk[redis]`` extras line. The in-memory store and the
rest of the SDK remain importable.

These tests simulate "missing extra" by patching ``sys.modules`` so the
top-level ``redis`` import inside :mod:`agent_sdk.state.redis` fails,
then reloading the module via :mod:`importlib`. The patch is reverted
in fixture teardown so subsequent tests in the same run are unaffected.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator

import pytest


@pytest.fixture
def redis_unavailable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pretend redis-py is not installed for the duration of this test.

    Removes any already-imported ``redis*`` and ``agent_sdk.state.redis``
    modules from :data:`sys.modules`, and installs a ``None`` entry for
    ``redis`` so the next ``import redis`` raises :class:`ImportError`.
    The guard imports ``redis.asyncio`` and ``redis.exceptions`` as
    submodules, so every ``redis.*`` entry must be stripped too. Tearing
    down restores the prior state by reloading the real
    :mod:`agent_sdk.state.redis` so the rest of the test session sees a
    working Redis surface.
    """
    # Strip every cached redis* module + agent_sdk.state.redis so the next
    # import path goes through our blocked entry. We collect first, then
    # mutate, to avoid mutating during iteration.
    removed: list[str] = [
        name
        for name in list(sys.modules)
        if name == "redis" or name.startswith("redis.") or name == "agent_sdk.state.redis"
    ]
    for name in removed:
        monkeypatch.delitem(sys.modules, name, raising=False)

    # `None` in sys.modules makes `import redis` raise ImportError.
    monkeypatch.setitem(sys.modules, "redis", None)

    yield
    # monkeypatch will revert sys.modules entries on teardown — at which
    # point the real redis is back. We also pop our half-built
    # agent_sdk.state.redis so the next caller re-imports it cleanly.
    sys.modules.pop("agent_sdk.state.redis", None)


def test_import_without_redis_raises_clear_error(
    redis_unavailable: None,
) -> None:
    """Direct ``import agent_sdk.state.redis`` without redis gives a clear ImportError."""
    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("agent_sdk.state.redis")
    msg = str(exc_info.value)
    assert "redis" in msg.lower()
    assert "agent-sdk[redis]" in msg


def test_lazy_attribute_access_on_state_package_triggers_extras_hint(
    redis_unavailable: None,
) -> None:
    """``agent_sdk.state.RedisStateStore`` (lazy) raises the same ImportError."""
    # `agent_sdk.state` itself does not import redis, so a fresh import
    # of it must succeed even while redis is blocked.
    sys.modules.pop("agent_sdk.state", None)
    state_pkg = importlib.import_module("agent_sdk.state")

    with pytest.raises(ImportError) as exc_info:
        _ = state_pkg.RedisStateStore  # triggers __getattr__
    msg = str(exc_info.value)
    assert "redis" in msg.lower()
    assert "agent-sdk[redis]" in msg


def test_top_level_lazy_attribute_access_triggers_extras_hint(
    redis_unavailable: None,
) -> None:
    """``agent_sdk.RedisStateStore`` (top-level lazy) raises the same ImportError."""
    sys.modules.pop("agent_sdk", None)
    pkg = importlib.import_module("agent_sdk")

    with pytest.raises(ImportError) as exc_info:
        _ = pkg.RedisStateStore  # triggers top-level __getattr__
    msg = str(exc_info.value)
    assert "redis" in msg.lower()
    assert "agent-sdk[redis]" in msg


def test_state_package_import_still_works_without_extras(
    redis_unavailable: None,
) -> None:
    """``import agent_sdk.state`` succeeds and exposes ``MemoryStateStore``."""
    sys.modules.pop("agent_sdk.state", None)
    state_pkg = importlib.import_module("agent_sdk.state")
    # MemoryStateStore must be reachable; only the Redis surface is gated.
    assert state_pkg.MemoryStateStore is not None
    assert state_pkg.StateStore is not None


def test_unknown_attribute_still_raises_attribute_error(
    redis_unavailable: None,
) -> None:
    """``__getattr__`` does NOT swallow access to non-Redis attribute names."""
    sys.modules.pop("agent_sdk.state", None)
    state_pkg = importlib.import_module("agent_sdk.state")
    with pytest.raises(AttributeError):
        _ = state_pkg.DefinitelyNotASymbol
