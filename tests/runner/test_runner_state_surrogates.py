"""``AgentRunner`` persisting text that holds a surrogate code point, through each state store (BR-025).

Before BR-025, when ``AgentRunner`` persisted a final answer holding a
surrogate code point (U+D800-U+DFFF), ``SqlStateStore`` (SQLite through
aiosqlite) raised ``UnicodeEncodeError`` and ``RedisStateStore`` (fakeredis)
``PydanticSerializationError`` out of ``AgentRunner.run()`` after the
``FinalEvent``, not as ``StateStoreError``. The answer was not stored, the
session kept the user turn without it, and the next run sent two user
messages in a row (BR-025 evidence, P1, on the ``cd4b811`` ``src`` and on
1.10.1). ``MemoryStateStore`` stored the answer. Since BR-025 the two
durable stores write each surrogate code point in ``content``, ``name`` and
``tool_call_id`` as its six-character ``\\udXXX`` escape, and reading the
message back returns that escaped text, which cannot be told apart from the
same six characters typed; ``MemoryStateStore`` is unchanged and keeps the
code point.

Every test drives a real ``AgentRunner`` over a real ``AgentLoop`` and the
real ``OpenAICompatibleClient`` (``max_retries=0``) on pytest-httpx, with a
real store: ``MemoryStateStore``, ``SqlStateStore`` on in-memory aiosqlite
(``StaticPool``) or ``RedisStateStore`` on fakeredis (its own
``FakeServer``). Bodies holding a surrogate code point are sent as
``content=`` bytes from ``json.dumps``, whose default ``ensure_ascii=True``
writes the JSON escape, which the ``openai`` client decodes into the code
point. Never ``add_response(json=...)``: httpx 0.28.1 encodes a ``json=``
body as strict UTF-8, so the mock response itself would fail to build.
Expected escapes are written as literals.

Release dependence (as for BR-024's tests): what the stores keep does not
depend on ``openai``; the requests do. The shipped client's serialiser
writes an assistant message's content with the same escape (BR-024), and
``openai`` 2.43.0 and 2.54.0 then encode the body as strict UTF-8, which is
why a user message holding a surrogate code point still raises at the
request (RS4). On a release whose encoder escaped the body itself RS4 would
not raise there (by reading; not run).

No new code path here logs message text, so these tests use pytest's
default output capture, which writes with ``errors="replace"`` and so could
not show a log line raising. BR-025 probes P1 and P5 ran runs of the same
shapes, over ``httpx.MockTransport``, under structlog's default
configuration with stdout a strict UTF-8 file, and no log line raised.

What these do NOT pin: Postgres, a real Redis server, ``tool_calls``
(``AgentRunner`` persists none; see the BR-025 sections of
``tests/state/test_sql.py`` and ``tests/state/test_redis.py``).
"""

from __future__ import annotations

import json
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import fakeredis
import pytest
from pytest_httpx import HTTPXMock
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from fifty_agent_sdk import (
    AgentEvent,
    AgentLoop,
    AgentRunner,
    ErrorEvent,
    FinalEvent,
    MemoryStateStore,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    StateStore,
    ToolMode,
)
from fifty_agent_sdk.state.redis import RedisStateStore
from fifty_agent_sdk.state.sql import SqlStateStore, sql_metadata

_ENDPOINT = "https://example.com/v1/chat/completions"
_SURROGATE = chr(0xD800)
_ESCAPE = "\\ud800"  # the six characters written for chr(0xD800)
_STORES = ["memory", "sql", "redis"]
_JSON_FINAL = (
    '{{"thought": "done", "action": "final", "tool_name": null, "tool_args": null, "answer": "{}"}}'
)


def _completion(content: str) -> bytes:
    """A chat completion body; ``json.dumps`` writes a surrogate code point as the JSON escape."""
    return json.dumps(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _respond(httpx_mock: HTTPXMock, content: str) -> None:
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINT,
        content=_completion(content),
        headers={"content-type": "application/json"},
    )


def _runner(
    store: StateStore, tool_mode: ToolMode = ToolMode.NATIVE, system_prompt: str | None = None
) -> AgentRunner:
    client = OpenAICompatibleClient(
        api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
    )
    loop = AgentLoop(
        llm=client,
        registry=Registry(),
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=tool_mode,
    )
    return AgentRunner(loop=loop, state=store, system_prompt=system_prompt)


@asynccontextmanager
async def _open_store(kind: str) -> AsyncIterator[StateStore]:
    """A fresh store of ``kind``: memory, SQL on in-memory aiosqlite, or Redis on its own ``FakeServer``."""
    if kind == "memory":
        yield MemoryStateStore()
        return
    if kind == "sql":
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(sql_metadata.create_all)
        try:
            yield SqlStateStore(engine)
        finally:
            await engine.dispose()
        return
    store = RedisStateStore("redis://localhost:6379/0")
    store._client = fakeredis.aioredis.FakeRedis(
        server=fakeredis.FakeServer(), decode_responses=True
    )
    try:
        yield store
    finally:
        await store.aclose()


def _stored_text(kind: str, text: str) -> str:
    """What ``kind`` returns for ``text`` holding U+D800: the code point (memory) or the escape."""
    return text if kind == "memory" else text.replace(_SURROGATE, _ESCAPE)


async def _stored(store: StateStore) -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in await store.get_messages("s1")]


def _ends_on(events: list[AgentEvent], text: str) -> None:
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].text == text
    assert not any(isinstance(e, ErrorEvent) for e in events)


# --- RS1: the final answer is stored ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["native", "json_mode"])
@pytest.mark.parametrize("kind", _STORES)
async def test_runner_stores_a_final_answer_holding_a_surrogate(
    httpx_mock: HTTPXMock, kind: str, mode: str
) -> None:
    """A final answer holding U+D800 is stored; the run ends on its ``FinalEvent`` with no exception (BR-025, AC-1/AC-4).

    The events keep the provider's text. ``SqlStateStore`` and
    ``RedisStateStore`` store the answer with ``\\ud800``; ``MemoryStateStore``
    keeps the code point. In JSON mode the Runner stores the raw completion,
    the envelope, so the escape sits inside its ``answer`` string. On the
    ``cd4b811`` ``src`` the sql and redis rows raised ``UnicodeEncodeError``
    and ``PydanticSerializationError`` out of ``run()`` after the
    ``FinalEvent`` and stored only the user turn (BR-025 evidence, P1).
    """
    if mode == "native":
        completion, answer = "final " + _SURROGATE, "final " + _SURROGATE
        runner_mode = ToolMode.NATIVE
    else:
        completion, answer = _JSON_FINAL.format("answer " + _SURROGATE), "answer " + _SURROGATE
        runner_mode = ToolMode.JSON
    _respond(httpx_mock, completion)

    async with _open_store(kind) as store:
        events = [e async for e in _runner(store, runner_mode).run("s1", "first question")]

        _ends_on(events, answer)
        final = events[-1]
        assert isinstance(final, FinalEvent)
        assert final.raw_completion == completion
        assert await _stored(store) == [
            ("user", "first question"),
            ("assistant", _stored_text(kind, completion)),
        ]


# --- RS2: the next run sends one user turn, then the stored answer --------------------------------


@pytest.mark.parametrize("kind", _STORES)
async def test_next_run_sends_one_user_turn_then_the_stored_answer(
    httpx_mock: HTTPXMock, kind: str
) -> None:
    """After a final answer holding U+D800, the next run sends user, answer, user, and the history alternates (BR-025, AC-2).

    The replayed answer goes out with ``\\ud800`` from every store: SQL and
    Redis return the escape, and the shipped client escapes the code point
    ``MemoryStateStore`` returns (BR-024). On the ``cd4b811`` ``src`` the sql
    and redis rows sent two user messages in a row (BR-025 evidence, P1).
    Release dependence: see the module docstring.
    """
    _respond(httpx_mock, "final " + _SURROGATE)
    _respond(httpx_mock, "done")

    async with _open_store(kind) as store:
        runner = _runner(store)
        first = [e async for e in runner.run("s1", "first question")]
        second = [e async for e in runner.run("s1", "second question")]

        _ends_on(first, "final " + _SURROGATE)
        _ends_on(second, "done")
        requests = httpx_mock.get_requests()
        assert len(requests) == 2
        assert json.loads(requests[1].content)["messages"][1:] == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "final " + _ESCAPE},
            {"role": "user", "content": "second question"},
        ]
        assert await _stored(store) == [
            ("user", "first question"),
            ("assistant", _stored_text(kind, "final " + _SURROGATE)),
            ("user", "second question"),
            ("assistant", "done"),
        ]


# --- RS3: the replayed request does not depend on the store ----------------------------------------


async def test_replayed_request_is_the_same_bytes_for_every_store(httpx_mock: HTTPXMock) -> None:
    """The second run's request body is the same bytes whichever store holds the history (BR-025, AC-4).

    ``MemoryStateStore`` returns the code point and the shipped client
    escapes it; SQL and Redis return the escape, which the client sends as
    it is. BR-025 probe P1 measured the same for its four rows. Release
    dependence: see the module docstring.
    """
    bodies: dict[str, bytes] = {}
    for kind in _STORES:
        _respond(httpx_mock, "final " + _SURROGATE)
        _respond(httpx_mock, "done")
        async with _open_store(kind) as store:
            runner = _runner(store)
            [e async for e in runner.run("s1", "first question")]
            [e async for e in runner.run("s1", "second question")]
        bodies[kind] = httpx_mock.get_requests()[-1].content

    assert len(httpx_mock.get_requests()) == 6
    assert json.loads(bodies["memory"])["messages"][2] == {
        "role": "assistant",
        "content": "final " + _ESCAPE,
    }
    assert bodies["sql"] == bodies["memory"]
    assert bodies["redis"] == bodies["memory"]


# --- RS4, RS5: host text (decision D4) -------------------------------------------------------------


def _assert_raised_by_the_client(exc: UnicodeEncodeError) -> None:
    assert exc.encoding == "utf-8"
    frames = [frame.filename for frame in traceback.extract_tb(exc.__traceback__)]
    assert any(name.endswith("openai_compat.py") for name in frames)  # the client, not a store


@pytest.mark.parametrize("kind", _STORES)
async def test_user_message_holding_a_surrogate_is_stored_escaped_and_the_client_raises(
    httpx_mock: HTTPXMock, kind: str
) -> None:
    """A user message holding U+D800 is stored, then the shipped client raises ``UnicodeEncodeError`` with nothing sent; on SQL and Redis the next run sends two user turns, with Memory it raises again (BR-025 decision D4).

    Pins BR-025's rule (the escape follows the field, for every role) and
    BR-024's host-text residual (user text is sent as it is). The Runner
    stores the user turn before the loop runs, and the loop sends the
    caller's message, which still holds the code point. SQL and Redis store
    it with ``\\ud800``, so the next run's request carries two user
    messages in a row (the stored, escaped turn and the new one) and the run
    completes. ``MemoryStateStore`` keeps the code point, so its next run
    raises the same way, again with nothing sent. On the ``cd4b811`` ``src``
    the sql and redis rows raised from the store and stored nothing (BR-025
    evidence, P5). Release dependence: see the module docstring. If this
    test changes, the CHANGELOG's BR-025 item changes with it.
    """
    async with _open_store(kind) as store:
        runner = _runner(store)
        with pytest.raises(UnicodeEncodeError) as first:
            [e async for e in runner.run("s1", "first " + _SURROGATE)]

        _assert_raised_by_the_client(first.value)
        assert httpx_mock.get_requests() == []
        assert await _stored(store) == [("user", _stored_text(kind, "first " + _SURROGATE))]

        if kind == "memory":
            with pytest.raises(UnicodeEncodeError) as second_raise:
                [e async for e in runner.run("s1", "second")]
            _assert_raised_by_the_client(second_raise.value)
            assert httpx_mock.get_requests() == []
            return

        _respond(httpx_mock, "done")
        second = [e async for e in runner.run("s1", "second")]

        _ends_on(second, "done")
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        assert json.loads(requests[0].content)["messages"][1:] == [
            {"role": "user", "content": "first " + _ESCAPE},
            {"role": "user", "content": "second"},
        ]


@pytest.mark.parametrize("kind", _STORES)
async def test_system_prompt_holding_a_surrogate_is_stored_escaped_and_the_client_raises(
    httpx_mock: HTTPXMock, kind: str
) -> None:
    """A ``system_prompt`` holding U+D800 is stored on the first turn, then the client raises; on SQL and Redis the next run sends both system messages and two user turns, with Memory it raises again (BR-025 decision D4).

    The Runner stores the ``system_prompt`` as a system message, then the
    user turn, and the loop sends the caller's text, which still holds the
    code point. SQL and Redis store the system message with ``\\ud800``, so
    the next run's request carries the loop's system message, the stored
    system message (escaped), the first user turn and the new one, and
    completes. ``MemoryStateStore`` keeps the code point, so its next run
    raises again, with nothing sent. On the ``cd4b811`` ``src`` the sql and
    redis rows raised from the store and stored nothing (BR-025 evidence,
    P5). Release dependence: see the module docstring.
    """
    async with _open_store(kind) as store:
        runner = _runner(store, system_prompt="sys " + _SURROGATE)
        with pytest.raises(UnicodeEncodeError) as first:
            [e async for e in runner.run("s1", "first")]

        _assert_raised_by_the_client(first.value)
        assert httpx_mock.get_requests() == []
        assert await _stored(store) == [
            ("system", _stored_text(kind, "sys " + _SURROGATE)),
            ("user", "first"),
        ]

        if kind == "memory":
            with pytest.raises(UnicodeEncodeError) as second_raise:
                [e async for e in runner.run("s1", "second")]
            _assert_raised_by_the_client(second_raise.value)
            assert httpx_mock.get_requests() == []
            return

        _respond(httpx_mock, "done")
        second = [e async for e in runner.run("s1", "second")]

        _ends_on(second, "done")
        messages = json.loads(httpx_mock.get_requests()[0].content)["messages"]
        assert messages[0]["role"] == "system"  # the loop's own system prompt
        assert messages[1:] == [
            {"role": "system", "content": "sys " + _ESCAPE},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
