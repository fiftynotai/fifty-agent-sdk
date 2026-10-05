"""Unit tests for :class:`fifty_agent_sdk.state.redis.RedisStateStore`.

Runs against an in-process ``fakeredis`` async server — no external
Redis instance is required, so these tests run in the default
``make test`` pass. Real-Redis behaviour (actual ``EXPIRE`` firing,
connection failures) is covered by the env-gated integration suite in
``test_redis_integration.py``.

These tests cover the documented contract from
:class:`fifty_agent_sdk.state.protocol.StateStore` plus the Redis-specific
commitments from BR-010:

* Round-trip preservation of all :class:`ChatMessage` fields and ordering.
* Empty / unknown session returns ``[]`` (never raises).
* Fresh-list-per-call (the defensive-copy invariant).
* Idempotent delete; delete is scoped to one session.
* The configured ``key_prefix`` is applied (default ``fifty_agent_sdk:state:``).
* Every mutation atomically refreshes all session keys to one sliding TTL;
  TTL-disabled mode emits no expiry commands.
* Every backend failure (:class:`redis.exceptions.RedisError`) is wrapped
  into :class:`StateStoreError` with the documented context shape.
* :class:`RedisStateStore` satisfies the :class:`StateStore` protocol.
* A surrogate code point in ``content``, ``name`` or ``tool_call_id`` is
  stored as its six-character escape, which reads back exactly like the
  same six characters typed; a message without one, or one holding a
  non-``str`` value set without validation, is stored as its own
  ``model_dump_json()``; ``tool_calls`` are serialised as before; a
  ``session_id`` holding one still raises ``UnicodeEncodeError`` (BR-025).

Fixture seam
    The ``store`` fixture builds a real :class:`RedisStateStore` and then
    reassigns ``store._client`` to a ``fakeredis.aioredis.FakeRedis``
    instance. This keeps the production constructor free of any
    test-only branching: the fake is injected from the outside via the
    private-attribute seam rather than through a constructor hook. The
    SQL test suite touches ``_owns_engine`` / ``_engine`` the same way,
    so this matches house style.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator
from typing import Any

import fakeredis
import pytest
import pytest_asyncio
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from redis.exceptions import RedisError, WatchError

from fifty_agent_sdk import ChatMessage, RedisStateStore, StateStore, StateStoreError, ToolCall
from fifty_agent_sdk.state.protocol import TRUNK_BRANCH_ID

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store() -> AsyncIterator[RedisStateStore]:
    """A :class:`RedisStateStore` whose client is an in-process fake.

    The store is constructed normally (so the production ``__init__`` is
    exercised), then its ``_client`` attribute is reassigned to a
    ``fakeredis.aioredis.FakeRedis`` instance. Injecting the fake through
    this private-attribute seam — rather than via a constructor hook —
    keeps the real constructor free of test-only branching.
    """
    s = RedisStateStore("redis://localhost:6379/0")
    # Fixture seam: swap in the in-process fake. ``decode_responses=True``
    # mirrors the production client so list members come back as ``str``.
    s._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield s
    finally:
        await s.aclose()


@pytest_asyncio.fixture
async def store_with_ttl() -> AsyncIterator[RedisStateStore]:
    """A :class:`RedisStateStore` configured with a 3600s per-session TTL.

    Uses the same fake-client seam as :func:`store`. The TTL value is
    large enough that it never elapses mid-test — TTL assertions check
    that the value is *set*, never that it *fires* (real expiry firing
    belongs to the integration suite).
    """
    s = RedisStateStore("redis://localhost:6379/0", ttl_seconds=3600)
    s._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield s
    finally:
        await s.aclose()


# ---------------------------------------------------------------------------
# Round-trip / read-path basics
# ---------------------------------------------------------------------------


async def test_get_empty_session_returns_empty_list(store: RedisStateStore) -> None:
    """Unknown session → empty list, not an error."""
    assert await store.get_messages("never-seen") == []


async def test_round_trip_preserves_ordering(store: RedisStateStore) -> None:
    """Appended messages come back in append order."""
    msgs = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content="c"),
        ChatMessage(role="assistant", content="d"),
        ChatMessage(role="user", content="e"),
    ]
    for m in msgs:
        await store.append("s1", m)

    got = await store.get_messages("s1")
    assert got == msgs
    assert [m.content for m in got] == ["a", "b", "c", "d", "e"]


async def test_round_trip_preserves_all_chat_message_fields(
    store: RedisStateStore,
) -> None:
    """All four optional/required ChatMessage fields survive a round-trip."""
    msg = ChatMessage(
        role="tool",
        content="result-body",
        name="search",
        tool_call_id="call-abc",
    )
    await store.append("s1", msg)
    got = await store.get_messages("s1")
    assert len(got) == 1
    assert got[0].role == "tool"
    assert got[0].content == "result-body"
    assert got[0].name == "search"
    assert got[0].tool_call_id == "call-abc"


async def test_round_trip_handles_optional_fields_as_none(
    store: RedisStateStore,
) -> None:
    """``name`` and ``tool_call_id`` are nullable and round-trip as None."""
    await store.append("s1", ChatMessage(role="user", content="hi"))
    got = await store.get_messages("s1")
    assert got[0].name is None
    assert got[0].tool_call_id is None


async def test_round_trip_allows_empty_content(store: RedisStateStore) -> None:
    """An assistant turn with only tool calls may have empty content."""
    await store.append("s1", ChatMessage(role="assistant", content=""))
    got = await store.get_messages("s1")
    assert got[0].content == ""


async def test_get_returns_new_list_object_each_call(store: RedisStateStore) -> None:
    """Every ``get_messages`` returns a freshly-built list (defensive copy)."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    second = await store.get_messages("s1")
    assert first is not second
    assert first == second


async def test_get_returns_defensive_copy(store: RedisStateStore) -> None:
    """Mutating the returned list does not affect future reads."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    first = await store.get_messages("s1")
    first.append(ChatMessage(role="user", content="EVIL"))
    first.clear()

    second = await store.get_messages("s1")
    assert len(second) == 1
    assert second[0].content == "a"


# ---------------------------------------------------------------------------
# Delete semantics
# ---------------------------------------------------------------------------


async def test_delete_removes_the_session(store: RedisStateStore) -> None:
    """Deleting a session clears its message list."""
    for index in range(3):
        await store.append("s1", ChatMessage(role="user", content=f"m{index}"))
    await store.delete("s1")
    assert await store.get_messages("s1") == []


async def test_delete_unknown_session_is_silent_noop(store: RedisStateStore) -> None:
    """Deleting a session that was never created must not raise."""
    await store.delete("never-seen")  # no exception
    assert await store.get_messages("never-seen") == []


async def test_delete_does_not_affect_other_sessions(store: RedisStateStore) -> None:
    """Deleting one session leaves siblings untouched."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    await store.append("s2", ChatMessage(role="user", content="b"))
    await store.delete("s1")
    assert await store.get_messages("s1") == []
    s2 = await store.get_messages("s2")
    assert len(s2) == 1
    assert s2[0].content == "b"


# ---------------------------------------------------------------------------
# Key layout / prefixing
# ---------------------------------------------------------------------------


async def test_default_key_prefix_is_applied(store: RedisStateStore) -> None:
    """With no override, keys are namespaced under ``fifty_agent_sdk:state:``."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    keys = await store._client.keys("*")
    # The trunk's messages live in the bare key; a :branches existence marker
    # also exists (BR-003). Both must carry the configured prefix.
    assert "fifty_agent_sdk:state:s1" in keys
    assert all(k.startswith("fifty_agent_sdk:state:") for k in keys)


async def test_custom_key_prefix_is_applied() -> None:
    """A custom ``key_prefix`` is used verbatim when forming Redis keys."""
    s = RedisStateStore("redis://localhost:6379/0", key_prefix="custom:ns:")
    s._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        await s.append("s1", ChatMessage(role="user", content="a"))
        keys = await s._client.keys("*")
        assert "custom:ns:s1" in keys
        assert all(k.startswith("custom:ns:") for k in keys)
    finally:
        await s.aclose()


def test_key_helper_concatenates_prefix_and_session_id() -> None:
    """``_key`` joins the configured prefix and the opaque session id."""
    s = RedisStateStore("redis://localhost:6379/0", key_prefix="p:")
    assert s._key("abc") == "p:abc"


def test_default_key_prefix_constant() -> None:
    """The default prefix is exactly ``fifty_agent_sdk:state:`` per the brief."""
    s = RedisStateStore("redis://localhost:6379/0")
    assert s._key("abc") == "fifty_agent_sdk:state:abc"


# ---------------------------------------------------------------------------
# TTL semantics
# ---------------------------------------------------------------------------


async def test_append_sets_ttl_when_configured(
    store_with_ttl: RedisStateStore,
) -> None:
    """An append with ``ttl_seconds`` set leaves a positive, bounded TTL."""
    await store_with_ttl.append("s1", ChatMessage(role="user", content="a"))
    ttl = await store_with_ttl._client.ttl("fifty_agent_sdk:state:s1")
    assert 0 < ttl <= 3600


async def test_ttl_is_refreshed_on_every_append(
    store_with_ttl: RedisStateStore,
) -> None:
    """Each append re-issues EXPIRE so the expiry window slides forward."""
    key = "fifty_agent_sdk:state:s1"
    await store_with_ttl.append("s1", ChatMessage(role="user", content="a"))
    await store_with_ttl.append("s1", ChatMessage(role="user", content="b"))
    # After the second append the TTL is still set and bounded by the
    # configured value — proving EXPIRE was re-issued, not left to decay.
    ttl = await store_with_ttl._client.ttl(key)
    assert 0 < ttl <= 3600


async def test_no_ttl_when_ttl_seconds_is_none(store: RedisStateStore) -> None:
    """With ``ttl_seconds=None`` the key exists with no expiry (TTL == -1)."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    # Redis returns -1 for a key that exists but has no associated expiry.
    ttl = await store._client.ttl("fifty_agent_sdk:state:s1")
    assert ttl == -1


async def test_switch_branch_applies_ttl_to_active_pointer(
    store_with_ttl: RedisStateStore,
) -> None:
    """``switch_branch`` must not leave the ``:active`` pointer without a TTL.

    Regression: the pointer was written with a bare ``SET``, which both
    stripped any TTL the key already carried and left a first-time key with
    none — so the active head could outlive the session it belongs to.
    """
    await store_with_ttl.append("s1", ChatMessage(role="user", content="a"))
    branch = await store_with_ttl.fork("s1", from_sequence=1)
    await store_with_ttl.switch_branch("s1", branch)
    ttl = await store_with_ttl._client.ttl("fifty_agent_sdk:state:s1:active")
    assert 0 < ttl <= 3600
    # Switching back re-applies the TTL just the same.
    await store_with_ttl.switch_branch("s1", "trunk")
    ttl = await store_with_ttl._client.ttl("fifty_agent_sdk:state:s1:active")
    assert 0 < ttl <= 3600


async def test_switch_branch_without_ttl_leaves_pointer_durable(
    store: RedisStateStore,
) -> None:
    """With ``ttl_seconds=None`` the ``:active`` pointer gets no expiry."""
    await store.append("s1", ChatMessage(role="user", content="a"))
    branch = await store.fork("s1", from_sequence=1)
    await store.switch_branch("s1", branch)
    ttl = await store._client.ttl("fifty_agent_sdk:state:s1:active")
    assert ttl == -1


async def test_fork_applies_ttl_to_branches_registry(
    store_with_ttl: RedisStateStore,
) -> None:
    """Forking a pre-BR-004 session gives the new ``:branches`` hash the TTL.

    Regression: the registry hash was created via HSETNX/HSET with no
    ``EXPIRE``, so forking a legacy single-list session (whose bare list
    predates the registry) left the registry durable while the rest of the
    session expired.
    """
    # Pre-BR-004 layout: a bare message list with no registry hash.
    await store_with_ttl._client.rpush(
        "fifty_agent_sdk:state:legacy",
        ChatMessage(role="user", content="old").model_dump_json(),
    )
    await store_with_ttl.fork("legacy", from_sequence=1)
    ttl = await store_with_ttl._client.ttl("fifty_agent_sdk:state:legacy:branches")
    assert 0 < ttl <= 3600


async def _session_pttls(store: RedisStateStore, session_id: str) -> list[int]:
    keys = await store._client.keys(f"fifty_agent_sdk:state:{session_id}*")
    return [int(await store._client.pttl(key)) for key in keys]


def _assert_synchronized_ttls(ttls: list[int]) -> None:
    assert ttls
    assert min(ttls) > 3_500_000
    assert max(ttls) - min(ttls) <= 100


async def _seed_branched_session(store: RedisStateStore) -> str:
    await store.append("sync", ChatMessage(role="user", content="trunk"))
    branch = await store.fork("sync", from_sequence=1)
    await store.switch_branch("sync", branch)
    await store.append("sync", ChatMessage(role="user", content="fork"))
    return branch


async def _shorten_session_ttls(store: RedisStateStore) -> None:
    keys = await store._client.keys("fifty_agent_sdk:state:sync*")
    for key in keys:
        await store._client.pexpire(key, 10_000)


async def test_append_refreshes_every_session_key_to_one_ttl(
    store_with_ttl: RedisStateStore,
) -> None:
    """BR-015 append atomically slides all branch/session keys together."""
    await _seed_branched_session(store_with_ttl)
    await _shorten_session_ttls(store_with_ttl)
    await store_with_ttl.append("sync", ChatMessage(role="user", content="again"))
    _assert_synchronized_ttls(await _session_pttls(store_with_ttl, "sync"))


async def test_fork_refreshes_legacy_and_registry_keys_to_one_ttl(
    store_with_ttl: RedisStateStore,
) -> None:
    """BR-015 forking a legacy session cannot create a longer-lived registry."""
    key = "fifty_agent_sdk:state:legacy-sync"
    await store_with_ttl._client.rpush(
        key, ChatMessage(role="user", content="old").model_dump_json()
    )
    await store_with_ttl._client.pexpire(key, 10_000)
    await store_with_ttl.fork("legacy-sync", from_sequence=1)
    _assert_synchronized_ttls(await _session_pttls(store_with_ttl, "legacy-sync"))


async def test_switch_refreshes_every_session_key_to_one_ttl(
    store_with_ttl: RedisStateStore,
) -> None:
    """BR-015 switching cannot create an active pointer that outlives messages."""
    branch = await _seed_branched_session(store_with_ttl)
    await _shorten_session_ttls(store_with_ttl)
    await store_with_ttl.switch_branch("sync", branch)
    _assert_synchronized_ttls(await _session_pttls(store_with_ttl, "sync"))


async def test_truncate_refreshes_every_session_key_to_one_ttl(
    store_with_ttl: RedisStateStore,
) -> None:
    """BR-015 truncate uses the same whole-session sliding TTL transaction."""
    await _seed_branched_session(store_with_ttl)
    await _shorten_session_ttls(store_with_ttl)
    await store_with_ttl.truncate_after("sync", 1)
    _assert_synchronized_ttls(await _session_pttls(store_with_ttl, "sync"))


async def test_ttl_disabled_mode_keeps_all_session_keys_durable(store: RedisStateStore) -> None:
    """BR-015 ttl_seconds=None emits no expiry for any mutation-created key."""
    branch = await _seed_branched_session(store)
    await store.switch_branch("sync", branch)
    await store.truncate_after("sync", 1)
    assert set(await _session_pttls(store, "sync")) == {-1}


async def test_mutation_retries_from_fresh_snapshot_after_watch_conflict(
    store_with_ttl: RedisStateStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-015 a WatchError retries without duplicating the state mutation."""
    original = store_with_ttl._mutation_snapshot
    calls = 0

    async def conflict_once(pipe: Any, session_id: str) -> Any:
        nonlocal calls
        calls += 1
        snapshot = await original(pipe, session_id)
        if calls == 1:
            await store_with_ttl._client.hset(
                store_with_ttl._branches_key(session_id),
                "trunk",
                json.dumps(
                    {
                        "parent_branch_id": None,
                        "forked_from_sequence": None,
                        "created_at": None,
                    }
                ),
            )
        return snapshot

    monkeypatch.setattr(store_with_ttl, "_mutation_snapshot", conflict_once)
    await store_with_ttl.append("retry", ChatMessage(role="user", content="once"))
    assert calls == 2
    assert [message.content for message in await store_with_ttl.get_messages("retry")] == ["once"]


async def test_append_retries_and_reroutes_when_active_pointer_changes(
    store_with_ttl: RedisStateStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-015 active-pointer conflicts retry before choosing the append target."""
    await store_with_ttl.append("route", ChatMessage(role="user", content="trunk"))
    branch = await store_with_ttl.fork("route", from_sequence=1)
    original = store_with_ttl._mutation_snapshot
    calls = 0

    async def switch_during_first_snapshot(pipe: Any, session_id: str) -> Any:
        nonlocal calls
        calls += 1
        snapshot = await original(pipe, session_id)
        if calls == 1:
            await store_with_ttl._client.set(store_with_ttl._active_key(session_id), branch)
        return snapshot

    monkeypatch.setattr(store_with_ttl, "_mutation_snapshot", switch_during_first_snapshot)
    await store_with_ttl.append("route", ChatMessage(role="user", content="routed"))

    assert calls == 2
    trunk = await store_with_ttl.get_messages("route", branch_id="trunk")
    fork = await store_with_ttl.get_messages("route", branch_id=branch)
    assert [message.content for message in trunk] == ["trunk"]
    assert [message.content for message in fork] == ["trunk", "routed"]


async def test_append_retries_when_existing_fork_list_changes(
    store_with_ttl: RedisStateStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-015 fork-list conflicts retry without lost, reordered, or duplicate writes."""
    branch = await _seed_branched_session(store_with_ttl)
    await _shorten_session_ttls(store_with_ttl)
    original = store_with_ttl._mutation_snapshot
    calls = 0

    async def append_during_first_snapshot(pipe: Any, session_id: str) -> Any:
        nonlocal calls
        calls += 1
        snapshot = await original(pipe, session_id)
        if calls == 1:
            await store_with_ttl._client.rpush(
                store_with_ttl._msgs_key(session_id, branch),
                ChatMessage(role="user", content="concurrent").model_dump_json(),
            )
        return snapshot

    monkeypatch.setattr(store_with_ttl, "_mutation_snapshot", append_during_first_snapshot)
    await store_with_ttl.append("sync", ChatMessage(role="user", content="requested"))

    assert calls == 2
    trunk = await store_with_ttl.get_messages("sync", branch_id="trunk")
    fork = await store_with_ttl.get_messages("sync", branch_id=branch)
    assert [message.content for message in trunk] == ["trunk"]
    assert [message.content for message in fork] == [
        "trunk",
        "fork",
        "concurrent",
        "requested",
    ]
    _assert_synchronized_ttls(await _session_pttls(store_with_ttl, "sync"))


async def test_mutation_wraps_watch_error_after_bounded_retries(
    store_with_ttl: RedisStateStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-015 sustained optimistic-lock contention fails after three attempts."""
    calls = 0

    async def always_conflict(_pipe: Any, _session_id: str) -> Any:
        nonlocal calls
        calls += 1
        raise WatchError("forced conflict")

    monkeypatch.setattr(store_with_ttl, "_mutation_snapshot", always_conflict)
    with pytest.raises(StateStoreError) as excinfo:
        await store_with_ttl.append("retry", ChatMessage(role="user", content="never"))
    assert calls == 3
    assert excinfo.value.context["wrapped"] == "WatchError"


def test_non_positive_ttl_seconds_rejected() -> None:
    """``ttl_seconds <= 0`` raises at construction instead of deleting data.

    Regression: the check was ``is not None``, so ``ttl_seconds=0`` issued
    ``EXPIRE 0`` inside the append transaction — deleting every session key
    on the first append.
    """
    for bad in (0, -1, -3600):
        with pytest.raises(ValueError, match="positive integer or None"):
            RedisStateStore("redis://localhost:6379/0", ttl_seconds=bad)


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


async def test_get_messages_wraps_redis_error(
    store: RedisStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RedisError from LRANGE surfaces as a wrapped StateStoreError."""

    async def boom(*_args: object, **_kwargs: object) -> object:
        raise RedisError("backend down")

    monkeypatch.setattr(store._client, "lrange", boom)

    with pytest.raises(StateStoreError) as exc_info:
        await store.get_messages("sid-1")
    err = exc_info.value
    assert err.context["operation"] == "get_messages"
    assert err.context["session_id"] == "sid-1"
    assert err.context["wrapped"] == "RedisError"
    assert isinstance(err.__cause__, RedisError)


async def test_append_wraps_redis_error(
    store: RedisStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RedisError raised during the append pipeline is wrapped."""

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RedisError("pipeline failed")

    # The pipeline is opened via ``self._client.pipeline(...)``; making
    # that call raise exercises the append try/except path.
    monkeypatch.setattr(store._client, "pipeline", boom)

    with pytest.raises(StateStoreError) as exc_info:
        await store.append("sid-2", ChatMessage(role="user", content="x"))
    err = exc_info.value
    assert err.context["operation"] == "append"
    assert err.context["session_id"] == "sid-2"
    assert err.context["wrapped"] == "RedisError"
    assert isinstance(err.__cause__, RedisError)


async def test_delete_wraps_redis_error(
    store: RedisStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RedisError from DEL surfaces as a wrapped StateStoreError."""

    async def boom(*_args: object, **_kwargs: object) -> object:
        raise RedisError("del failed")

    monkeypatch.setattr(store._client, "delete", boom)

    with pytest.raises(StateStoreError) as exc_info:
        await store.delete("sid-3")
    err = exc_info.value
    assert err.context["operation"] == "delete"
    assert err.context["session_id"] == "sid-3"
    assert err.context["wrapped"] == "RedisError"
    assert isinstance(err.__cause__, RedisError)


async def test_wrapped_error_carries_underlying_class_name(
    store: RedisStateStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``context['wrapped']`` records the concrete RedisError subclass."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    async def boom(*_args: object, **_kwargs: object) -> object:
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr(store._client, "lrange", boom)

    with pytest.raises(StateStoreError) as exc_info:
        await store.get_messages("sid-1")
    # ConnectionError is a RedisError subclass — caught and named precisely.
    assert exc_info.value.context["wrapped"] == "ConnectionError"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


async def test_redis_store_satisfies_state_store_protocol(
    store: RedisStateStore,
) -> None:
    """:class:`RedisStateStore` matches the :class:`StateStore` runtime protocol."""
    assert isinstance(store, StateStore)


# ---------------------------------------------------------------------------
# Text UTF-8 cannot encode (BR-025)
# ---------------------------------------------------------------------------
#
# ChatMessage.model_dump_json() raises PydanticSerializationError for a
# surrogate code point (U+D800-U+DFFF), and append called it before its
# RedisError wrapping, so before BR-025 such a message raised out of append
# as it was, with nothing stored (BR-025 evidence, P2). Since BR-025 append
# dumps a copy whose content, name and tool_call_id carry the six-character
# escape, which cannot be told apart from the same six characters typed
# (pinned below). Expected escapes are literals, never computed with
# escape_surrogates.

_SURROGATE = chr(0xD800)
_ESCAPE = "\\ud800"  # the six characters written for chr(0xD800)
_ARABIC = "مرحبا بالعالم "

# Messages with no surrogate code point anywhere: BR-025 probe P4's corpus
# without its last message, which holds one in tool_calls (see the
# tool_calls tests below).
_PLAIN_MESSAGES = [
    pytest.param(ChatMessage(role="user", content="hello world"), id="user_ascii"),
    pytest.param(ChatMessage(role="assistant", content=_ARABIC * 100), id="arabic_x100"),
    pytest.param(
        ChatMessage(role="assistant", content="smile " + chr(0x1F600) + " ok"), id="astral_emoji"
    ),
    pytest.param(ChatMessage(role="assistant", content="del" + chr(0x7F) + "end"), id="u007f"),
    pytest.param(ChatMessage(role="assistant", content="a\\ud800b"), id="typed_escape"),
    pytest.param(
        ChatMessage(role="user", content="q\"'\\\n\t\\\\end"), id="quotes_backslashes_newline_tab"
    ),
    pytest.param(ChatMessage(role="assistant", content=""), id="empty"),
    pytest.param(
        ChatMessage(role="tool", content="result", name="بحث", tool_call_id="call_1"),
        id="tool_nonascii_name",
    ),
    pytest.param(ChatMessage(role="user", content="hi", name="مستخدم"), id="user_named_nonascii"),
    pytest.param(
        ChatMessage(role="tool", content="r", name="find\\ud800", tool_call_id="c\\udfff"),
        id="tool_typed_escape_name",
    ),
    pytest.param(
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    name="search",
                    args={"q": {"n": [1e16, 0.1, None], "مفتاح": "قيمة", "e": None}},
                    id="call_a",
                ),
                ToolCall(name="lookup", args={"id": 7}),
            ],
        ),
        id="tool_calls",
    ),
    pytest.param(ChatMessage(role="assistant", content="x", tool_calls=[]), id="empty_tool_calls"),
    pytest.param(
        ChatMessage(role="system", content=("You are a helpful agent. " * 410)[:10240]),
        id="system_10kb",
    ),
    pytest.param(
        ChatMessage(
            role="assistant",
            content=(
                '{"thought": "t", "action": "final", "tool_name": null, '
                '"tool_args": null, "answer": "' + _ARABIC + '"}'
            ),
        ),
        id="json_envelope",
    ),
]


@pytest_asyncio.fixture
async def store_and_raw() -> AsyncIterator[tuple[RedisStateStore, Any]]:
    """A store and a ``decode_responses=False`` client on the same, fresh ``FakeServer``.

    The raw client reads the list entries as the bytes Redis holds.
    """
    server = fakeredis.FakeServer()
    s = RedisStateStore("redis://localhost:6379/0")
    s._client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    raw = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    try:
        yield s, raw
    finally:
        await raw.aclose()
        await s.aclose()


async def _raw_entries(store: RedisStateStore, raw: Any) -> list[bytes]:
    """The trunk list's entries for session ``s1``, through the store's own key helper."""
    entries: list[bytes] = await raw.lrange(store._msgs_key("s1", TRUNK_BRANCH_ID), 0, -1)
    return entries


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param(
            ChatMessage(role="assistant", content="a" + _SURROGATE + "b"),
            ChatMessage(role="assistant", content="a" + _ESCAPE + "b"),
            id="content_assistant",
        ),
        pytest.param(
            ChatMessage(role="user", content="a" + _SURROGATE + "b"),
            ChatMessage(role="user", content="a" + _ESCAPE + "b"),
            id="content_user",
        ),
        pytest.param(
            ChatMessage(role="system", content="a" + _SURROGATE + "b"),
            ChatMessage(role="system", content="a" + _ESCAPE + "b"),
            id="content_system",
        ),
        pytest.param(
            ChatMessage(role="tool", content="r", name="find" + _SURROGATE, tool_call_id="call_1"),
            ChatMessage(role="tool", content="r", name="find" + _ESCAPE, tool_call_id="call_1"),
            id="name_tool",
        ),
        pytest.param(
            ChatMessage(role="tool", content="r", name="find", tool_call_id="c" + _SURROGATE),
            ChatMessage(role="tool", content="r", name="find", tool_call_id="c" + _ESCAPE),
            id="tool_call_id_tool",
        ),
    ],
)
async def test_append_text_holding_a_surrogate_is_stored_escaped(
    store_and_raw: tuple[RedisStateStore, Any], message: ChatMessage, expected: ChatMessage
) -> None:
    """A surrogate code point in ``content``, ``name`` or ``tool_call_id`` is stored as its six-character escape (BR-025, AC-1/AC-4).

    ``append`` does not raise; the list entry is the escaped message's
    ``model_dump_json()``, and ``get_messages`` returns the escaped text. On
    the ``cd4b811`` ``src`` ``append`` raised ``PydanticSerializationError``
    (BR-025 evidence, P2).
    """
    store, raw = store_and_raw

    await store.append("s1", message)

    assert await _raw_entries(store, raw) == [expected.model_dump_json().encode("utf-8")]
    assert await store.get_messages("s1") == [expected]


@pytest.mark.parametrize("message", _PLAIN_MESSAGES)
async def test_append_without_surrogates_stores_the_same_entry(
    store_and_raw: tuple[RedisStateStore, Any], message: ChatMessage
) -> None:
    """A message with no surrogate code point is stored as before BR-025: the entry is its own ``model_dump_json()`` (AC-3).

    BR-025 probe P4 compared these entries' bytes before and after the
    change for the same messages and found them identical.
    """
    store, raw = store_and_raw

    await store.append("s1", message)

    assert await _raw_entries(store, raw) == [message.model_dump_json().encode("utf-8")]
    assert await store.get_messages("s1") == [message]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(ToolCall(name="find" + _SURROGATE, args={"q": "x"}, id="call_1"), id="name"),
        pytest.param(ToolCall(name="find", args={"q": "x"}, id="c" + _SURROGATE), id="id"),
        pytest.param(
            ToolCall(name="find", args={"q": "v" + _SURROGATE}, id="call_1"), id="args_value"
        ),
    ],
)
async def test_tool_calls_holding_a_surrogate_still_raise_raw(
    store_and_raw: tuple[RedisStateStore, Any], call: ToolCall
) -> None:
    """A surrogate code point in a tool call's name, id or argument value still raises ``PydanticSerializationError``, not ``StateStoreError`` (BR-025 decision D3).

    Pins a documented gap, not a decision: BR-025 escapes only ``content``,
    ``name`` and ``tool_call_id``, and nothing is stored. ``AgentRunner``
    persists no ``tool_calls``. If this test changes, the CHANGELOG's BR-025
    *Not covered* item changes with it.
    """
    store, raw = store_and_raw

    with pytest.raises(PydanticSerializationError) as exc:
        await store.append("s1", ChatMessage(role="assistant", content="", tool_calls=[call]))

    assert not isinstance(exc.value, StateStoreError)
    assert await _raw_entries(store, raw) == []
    assert await store.get_messages("s1") == []


async def test_tool_call_argument_key_holding_a_surrogate_is_stored_as_replacement_characters(
    store_and_raw: tuple[RedisStateStore, Any],
) -> None:
    """A surrogate code point in an argument's key does not raise: the key is stored with three U+FFFD (BR-025 decision D3; pre-existing).

    Pins a documented gap, not a decision: pydantic's ``model_dump_json()``
    writes such a key so (pydantic 2.13.4 and 2.13.5, BR-025 evidence, P2).
    If this test changes, the CHANGELOG's BR-025 *Not covered* item changes
    with it.
    """
    store, _raw = store_and_raw
    call = ToolCall(name="find", args={"k" + _SURROGATE: 1}, id="call_1")

    await store.append("s1", ChatMessage(role="assistant", content="", tool_calls=[call]))

    got = await store.get_messages("s1")
    assert got[0].tool_calls == [
        ToolCall(name="find", args={"k" + chr(0xFFFD) * 3: 1}, id="call_1")
    ]


@pytest.mark.parametrize(
    "content",
    [pytest.param("a" + _ESCAPE, id="typed"), pytest.param("a" + _SURROGATE, id="code_point")],
)
async def test_escaped_and_typed_text_read_back_the_same(
    store_and_raw: tuple[RedisStateStore, Any], content: str
) -> None:
    """The six characters typed and a surrogate code point both read back as the six characters (BR-025, AC-4).

    Nothing decodes the escape on read, so the two cannot be told apart.
    """
    store, _raw = store_and_raw

    await store.append("s1", ChatMessage(role="assistant", content=content))

    got = await store.get_messages("s1")
    assert got[0].content == "a" + _ESCAPE


@pytest.mark.parametrize(
    "session_id",
    [
        pytest.param("s" + _SURROGATE, id="d800"),
        pytest.param("s" + chr(0xD83D) + chr(0xDE00), id="pair_halves"),
    ],
)
async def test_session_id_holding_a_surrogate_still_raises_raw(
    store_and_raw: tuple[RedisStateStore, Any], session_id: str
) -> None:
    """A ``session_id`` holding a surrogate code point still raises ``UnicodeEncodeError``, not ``StateStoreError``, from ``append`` and ``get_messages`` (BR-025).

    Pins a documented gap, not a decision: BR-025 escapes no ``session_id``.
    An id is part of every key, and its escape would be the same key as the
    six characters typed, so two sessions would become one (the battery's
    X1, which escapes the id in ``append``, turns this test red). Nothing is
    stored. If this test changes, the CHANGELOG's BR-025 *Not covered* item
    changes with it.
    """
    store, raw = store_and_raw

    with pytest.raises(UnicodeEncodeError) as appended:
        await store.append(session_id, ChatMessage(role="user", content="hello"))
    with pytest.raises(UnicodeEncodeError) as read:
        await store.get_messages(session_id)

    assert not isinstance(appended.value, StateStoreError)
    assert not isinstance(read.value, StateStoreError)
    assert await raw.keys("*") == []
    assert await store.get_messages("s" + _ESCAPE) == []


def _assigned(**fields: object) -> ChatMessage:
    """A validated message whose fields are then replaced by assignment, which skips validation."""
    message = ChatMessage(role="assistant", content="x", name="n", tool_call_id="t")
    for key, value in fields.items():
        setattr(message, key, value)
    return message


@pytest.mark.parametrize(
    ("message", "read_back"),
    [
        pytest.param(_assigned(content=123), None, id="content_int"),
        pytest.param(_assigned(content=None), None, id="content_none"),
        pytest.param(
            ChatMessage.model_construct(role="assistant", content=b"by"), "by", id="content_bytes"
        ),
        pytest.param(_assigned(name=5), None, id="name_int"),
        pytest.param(_assigned(tool_call_id=7), None, id="tool_call_id_int"),
    ],
)
async def test_append_of_an_unvalidated_non_str_field_is_stored_as_before(
    store_and_raw: tuple[RedisStateStore, Any], message: ChatMessage, read_back: str | None
) -> None:
    """A message built or changed without validation, holding something other than a ``str`` in a text field, is stored as before BR-025.

    The entry is the message's own ``model_dump_json()``. Reading it back
    raises pydantic's ``ValidationError`` unless the value still validates
    as a ``str`` (bytes do), as before: the store does not wrap that
    corruption signal. BR-025's escape passes such a value through:
    ``escape_surrogates`` would raise ``AttributeError`` on it out of
    ``append`` (BR-025 evidence, round 2, M2). The ``cd4b811`` ``src``
    behaved as this test asserts. pydantic's serializer warnings for the
    unexpected types are silenced here.
    """
    store, raw = store_and_raw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        await store.append("s1", message)
        expected = message.model_dump_json().encode("utf-8")

    assert await _raw_entries(store, raw) == [expected]
    if read_back is None:
        with pytest.raises(ValidationError):
            await store.get_messages("s1")
    else:
        got = await store.get_messages("s1")
        assert got[0].content == read_back
