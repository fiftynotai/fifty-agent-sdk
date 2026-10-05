"""Model-written text holding a surrogate code point, re-sent through the real client (BR-024).

Before BR-024 a surrogate code point (U+D800-U+DFFF) in text the model wrote
made the request that re-sent it raise ``UnicodeEncodeError`` out of
``AgentLoop.run()``, with no ``ErrorEvent`` or ``FinalEvent``: a JSON-mode or
PROSE completion echoed back on a tool step, a parser retry or a require-tool
re-ask (in JSON mode also streamed), a native tool call's name or content, a
tool name ``JsonModeParser`` decoded from the escape's six characters, and an
assistant message passed to ``run()``, such as a final answer the Runner
stored (BR-024 evidence, P4, on the BR-023 tree ``6d64a22``, ``openai`` 2.43.0
on CPython 3.14.3 and 2.54.0 on CPython 3.11.15, 3.13.2 and 3.14.3, with
structlog's output silenced). Since BR-024 the shipped client writes each one
as its six-character ``\\udXXX`` escape in the request, and the run
continues.

A tool name holding a surrogate code point had a second raw exit. Under
structlog's default configuration (the SDK never configures structlog) the
loop's ``tool_invoked`` debug line printed the model's tool name as it was
(structlog 26.1.0's ``ConsoleRenderer`` uses ``repr`` only for a string
holding a space, tab, ``=``, a quote or a line break; by reading
``structlog/dev.py:923-926``, and measured for each of those characters),
and with stdout a strict UTF-8 stream (a file or a TTY under a UTF-8
locale) that ``print`` raised ``UnicodeEncodeError`` before the second
request (BR-024 evidence, round 2). Since BR-024 that line writes the name
escaped too. B13 and B14 pin it on their own strict stream; under pytest's
default output capture, the other tests here log to its capture stream,
which writes with ``errors="replace"``, so they cannot see a log line
raise.

Every test drives a real ``AgentLoop`` (or ``AgentRunner``) over the real
``OpenAICompatibleClient`` (``max_retries=0``) on pytest-httpx, because
``FakeLLMClient`` never encodes a request. Bodies holding a surrogate code
point are sent as ``content=`` bytes from ``json.dumps``, whose default
``ensure_ascii=True`` writes the JSON escape, which the ``openai`` client
decodes into the code point; one row splices the UTF-8 encoded surrogate
bytes ``ED A0 80`` into the body instead. Never ``add_response(json=...)``:
httpx 0.28.1 encodes a ``json=`` body as strict UTF-8
(``httpx/_content.py:176-179``, by reading), so the mock response itself
would fail to build. Expected escapes are written as literals.

Release dependence (as for BR-022's W3): on ``openai`` 2.43.0 and 2.54.0 the
client encodes the request body as strict UTF-8, which is why these runs
raised before BR-024. On a release whose encoder escaped the body itself a
run would not raise, but the exact-value assertions would still see a
missing escape (by reading; not run);
``tests/llm/test_openai_compat_surrogate_replay.py`` (B1) pins the serialiser
on any release.

What these do NOT pin: HTTP bytes; a custom ``LLMClient`` (it receives the
text as before); user and system messages, pinned here as a raw raise (a
documented gap, B9); log output under any structlog configuration other
than the default one B13 and B14 use.
"""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from pytest_httpx import HTTPXMock

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    ActionEvent,
    AgentEvent,
    AgentLoop,
    AgentRunner,
    ChatMessage,
    ErrorEvent,
    FinalEvent,
    JsonModeParser,
    MemoryStateStore,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ThoughtEvent,
    ToolFailedEvent,
    ToolMode,
)
from tests.loop.conftest import FakeTool

_ENDPOINT = "https://example.com/v1/chat/completions"
_SURROGATE = chr(0xD800)
_ESCAPE = "\\ud800"  # the six characters the client writes for chr(0xD800)
_NOT_FOUND = "ToolNotFound: tool 'find\\ud800' is not registered."
_MARK = "XMARKX"

# JSON-mode and PROSE completions, with one slot for the surrogate code point or its escape.
_JSON_TOOL = (
    '{{"thought": "think {}", "action": "tool", "tool_name": "lookup", '
    '"tool_args": {{"q": "x"}}, "answer": null}}'
)
_JSON_FINAL_TEMPLATE = (
    '{{"thought": "done", "action": "final", "tool_name": null, "tool_args": null, '
    '"answer": "answer {}"}}'
)
_PROSE_TOOL = 'Thought: think {}\nAction: lookup\nAction Input: {{"q": "x"}}'
_JSON_FINAL = (
    '{"thought": "done", "action": "final", "tool_name": null, "tool_args": null, "answer": "done"}'
)
_PROSE_FINAL = "Thought: done\nFinal Answer: done"


# --- Bodies --------------------------------------------------------------------------------


def _completion(
    content: str | None, tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": message,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _call(name: str, call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"q": "x"}'},
    }


def _escaped_body(body: dict[str, Any]) -> bytes:
    """The body as JSON with the stdlib default, so a surrogate code point is the JSON escape."""
    data = json.dumps(body).encode()
    assert b"\\ud800" in data
    return data


def _raw_surrogate_body(body: dict[str, Any]) -> bytes:
    """The body with the UTF-8 encoded surrogate bytes ``ED A0 80`` spliced in at the marker."""
    data = json.dumps(body).encode()
    assert data.count(_MARK.encode()) == 1
    return data.replace(_MARK.encode(), b"\xed\xa0\x80")


def _sse(content: str) -> bytes:
    """A ``text/event-stream`` body that streams ``content`` and then ``stop``."""
    chunks = [
        {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None},
        {"index": 0, "delta": {}, "finish_reason": "stop"},
    ]
    events = [
        {
            "id": "cmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [choice],
        }
        for choice in chunks
    ]
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def _respond(httpx_mock: HTTPXMock, body: bytes, content_type: str = "application/json") -> None:
    httpx_mock.add_response(
        method="POST", url=_ENDPOINT, content=body, headers={"content-type": content_type}
    )


# --- Loops ---------------------------------------------------------------------------------


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
    )


def _loop(
    tool_mode: ToolMode | None,
    *,
    stream: bool = False,
    **safety: Any,
) -> AgentLoop:
    """A loop over the real client; ``tool_mode=None`` is the legacy ``JsonModeParser`` path."""
    registry = Registry()
    registry.register(FakeTool("lookup"))
    kwargs: dict[str, Any] = {}
    if tool_mode is None:
        kwargs["parser"] = JsonModeParser()
        kwargs["output_format"] = JSON_MODE_OUTPUT_FORMAT
    else:
        kwargs["tool_mode"] = tool_mode
    return AgentLoop(
        llm=_client(),
        registry=registry,
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(**safety),
        model="test-model",
        stream=stream,
        **kwargs,
    )


async def _run(loop: AgentLoop, messages: list[ChatMessage] | None = None) -> list[AgentEvent]:
    sent = messages if messages is not None else [ChatMessage(role="user", content="q")]
    return [event async for event in loop.run(sent)]


def _sent_messages(httpx_mock: HTTPXMock, index: int) -> list[dict[str, Any]]:
    body = json.loads(httpx_mock.get_requests()[index].content)
    messages: list[dict[str, Any]] = body["messages"]
    return messages


def _ends_on(events: list[AgentEvent], text: str) -> None:
    assert isinstance(events[-1], FinalEvent)
    assert events[-1].text == text
    assert not any(isinstance(e, ErrorEvent) for e in events)


# --- B4: a completion echoed back ------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        "json_tool_turn",
        "json_tool_turn_raw_bytes",
        "json_parser_retry",
        "json_require_tool",
        "prose_tool_turn",
    ],
)
async def test_model_text_holding_a_surrogate_is_echoed_escaped(
    httpx_mock: HTTPXMock, row: str
) -> None:
    """A completion holding U+D800 that the loop echoes back is sent with ``\\ud800`` and the run ends on the model's answer (BR-024).

    Rows: a JSON-mode tool turn (the body carries the JSON escape, or the
    UTF-8 encoded surrogate bytes ``ED A0 80``), a JSON-mode completion the
    parser rejects (the parser-retry echo), a JSON-mode final answer the
    require-tool guard sends back (``require_tool_before_final=True``), and
    a PROSE tool turn. Each takes two requests; the echo is the third
    message of the second (after the system prompt and the user message).
    On the ``6d64a22`` ``src`` each raised ``UnicodeEncodeError`` from the
    ``openai`` client's body encode at the second request (BR-024 evidence,
    P4). The events keep the model's text: where the echoed completion
    carries the thought (the two JSON tool rows and the PROSE row), the
    first ``ThoughtEvent`` holds U+D800. Release dependence: see the module
    docstring.
    """
    tool_mode = ToolMode.PROSE if row == "prose_tool_turn" else ToolMode.JSON
    safety: dict[str, Any] = {}
    final = _PROSE_FINAL if row == "prose_tool_turn" else _JSON_FINAL
    if row == "json_tool_turn":
        completion, echoed = _JSON_TOOL.format(_SURROGATE), _JSON_TOOL.format(_ESCAPE)
        first = _escaped_body(_completion(completion))
    elif row == "json_tool_turn_raw_bytes":
        completion, echoed = _JSON_TOOL.format(_SURROGATE), _JSON_TOOL.format(_ESCAPE)
        first = _raw_surrogate_body(_completion(_JSON_TOOL.format(_MARK)))
    elif row == "json_parser_retry":
        completion, echoed = "not json " + _SURROGATE, "not json " + _ESCAPE
        first = _escaped_body(_completion(completion))
    elif row == "json_require_tool":
        completion = _JSON_FINAL_TEMPLATE.format(_SURROGATE)
        echoed = _JSON_FINAL_TEMPLATE.format(_ESCAPE)
        first = _escaped_body(_completion(completion))
        safety["require_tool_before_final"] = True
    else:
        completion, echoed = _PROSE_TOOL.format(_SURROGATE), _PROSE_TOOL.format(_ESCAPE)
        first = _escaped_body(_completion(completion))
    _respond(httpx_mock, first)
    _respond(httpx_mock, json.dumps(_completion(final)).encode())

    events = await _run(_loop(tool_mode, **safety))

    assert len(httpx_mock.get_requests()) == 2
    assert _sent_messages(httpx_mock, 1)[2] == {"role": "assistant", "content": echoed}
    _ends_on(events, "done")
    if row in {"json_tool_turn", "json_tool_turn_raw_bytes", "prose_tool_turn"}:
        thought = next(e for e in events if isinstance(e, ThoughtEvent))
        assert thought.text == "think " + _SURROGATE


# --- B5: a native tool call's name (and content) ----------------------------------------------


@pytest.mark.parametrize("row", ["single", "batch", "legacy_native_flag", "content_too"])
async def test_native_tool_name_holding_a_surrogate_is_replayed_escaped(
    httpx_mock: HTTPXMock, row: str
) -> None:
    """A native tool call named ``find`` + U+D800 is replayed with ``find\\ud800`` in the assistant turn and in the tool reply's ``name``; the events keep the model's name (BR-024).

    Rows: ``ToolMode.NATIVE`` with one call, a batch whose second call holds
    the code point, the legacy path with
    ``SafetyConfig(native_tools_enabled=True)``, and a native turn whose
    content holds one too. The tool is not registered, so its reply is the
    ``ToolNotFound`` text, which the loop escapes (BR-022). The ids still
    pair. On the ``6d64a22`` ``src``, with structlog silenced, each raised
    ``UnicodeEncodeError`` at the second request (BR-024 evidence, P4); under
    structlog's default configuration and a strict stdout each raised
    earlier, from the ``tool_invoked`` debug line, which
    ``test_tool_invoked_log_line_writes_the_tool_name_escaped`` (B13) pins
    (round 2). This test logs to pytest's capture, so it pins the request
    only. Release dependence: see the module docstring.
    """
    name = "find" + _SURROGATE
    calls = [_call("lookup", "call_1"), _call(name, "call_2")] if row == "batch" else [_call(name)]
    content = "thinking " + _SURROGATE if row == "content_too" else None
    _respond(httpx_mock, _escaped_body(_completion(content, calls)))
    final = _JSON_FINAL if row == "legacy_native_flag" else "done"
    _respond(httpx_mock, json.dumps(_completion(final)).encode())
    if row == "legacy_native_flag":
        loop = _loop(None, native_tools_enabled=True)
    else:
        loop = _loop(ToolMode.NATIVE)

    events = await _run(loop)

    assert len(httpx_mock.get_requests()) == 2
    sent = _sent_messages(httpx_mock, 1)
    turn, replies = sent[2], sent[3:]
    expected_names = ["lookup", "find" + _ESCAPE] if row == "batch" else ["find" + _ESCAPE]
    assert turn["role"] == "assistant"
    assert turn["content"] == ("thinking " + _ESCAPE if row == "content_too" else "")
    assert [entry["function"]["name"] for entry in turn["tool_calls"]] == expected_names
    assert [reply["name"] for reply in replies] == expected_names
    assert [reply["role"] for reply in replies] == ["tool"] * len(expected_names)
    assert replies[-1]["content"] == _NOT_FOUND
    assert [reply["tool_call_id"] for reply in replies] == [e["id"] for e in turn["tool_calls"]]
    actions = [e.tool_name for e in events if isinstance(e, ActionEvent)]
    assert actions == (["lookup", name] if row == "batch" else [name])
    failed = next(e for e in events if isinstance(e, ToolFailedEvent))
    assert failed.tool_name == name
    _ends_on(events, "done")


# --- B6: a tool name JsonModeParser decoded from the escape's six characters ------------------


async def test_parsed_tool_name_holding_a_surrogate_is_replayed_escaped(
    httpx_mock: HTTPXMock,
) -> None:
    """On the legacy path, a JSON-mode ``tool_name`` written as ``find\\ud800`` is decoded by ``JsonModeParser`` into ``find`` + U+D800, and the tool reply's ``name`` is sent as ``find\\ud800`` (BR-024).

    No ``tool_mode`` and the default ``tool_message_role`` (``"tool"``), so
    the reply carries the decoded name in its ``name`` field. The completion
    itself holds only ASCII (precondition), so it is echoed unchanged, and an
    escape where the provider's text is decoded could not see this name: it
    becomes a surrogate code point only when the parser decodes it. On the
    ``6d64a22`` ``src``, with structlog silenced, this run raised
    ``UnicodeEncodeError`` at the second request (BR-024 evidence, P4 row
    c4); under structlog's default configuration and a strict stdout it
    raised earlier, from the ``tool_invoked`` debug line, which B13's
    ``legacy_json_parser`` row pins (round 2). This test logs to pytest's
    capture, so it pins the request only. Release dependence: see the module
    docstring.
    """
    completion = (
        '{"thought": "calling", "action": "tool", "tool_name": "find\\ud800", '
        '"tool_args": {"q": "x"}, "answer": null}'
    )
    completion.encode("utf-8")  # precondition: the six ASCII characters, not the code point
    assert _SURROGATE not in completion
    _respond(httpx_mock, json.dumps(_completion(completion)).encode())
    _respond(httpx_mock, json.dumps(_completion(_JSON_FINAL)).encode())

    events = await _run(_loop(None))

    assert len(httpx_mock.get_requests()) == 2
    sent = _sent_messages(httpx_mock, 1)
    assert sent[2] == {"role": "assistant", "content": completion}
    assert sent[3]["role"] == "tool"
    assert sent[3]["name"] == "find" + _ESCAPE
    assert sent[3]["content"] == _NOT_FOUND
    action = next(e for e in events if isinstance(e, ActionEvent))
    assert action.tool_name == "find" + _SURROGATE
    _ends_on(events, "done")


# --- B7: streamed --------------------------------------------------------------------------------


async def test_streamed_completion_holding_a_surrogate_is_echoed_escaped(
    httpx_mock: HTTPXMock,
) -> None:
    """With ``stream=True``, a JSON-mode tool turn streamed with U+D800 (the JSON escape in the event data) is echoed with ``\\ud800`` in the second streamed request (BR-024).

    The request body is encoded by the ``create()`` call that opens each
    stream. On the ``6d64a22`` ``src`` this run raised
    ``UnicodeEncodeError`` from that call at the second request (BR-024
    evidence, P4). UTF-8 encoded surrogate bytes in a stream are another
    route: the stream decoder rejects them, which ends the run with
    ``UndecodableProviderBody`` (BR-023; BR-024 evidence, P4 control).
    Release dependence: see the module docstring.
    """
    for content in (_JSON_TOOL.format(_SURROGATE), _JSON_FINAL):
        _respond(httpx_mock, _sse(content), "text/event-stream")
    assert b"\\ud800" in _sse(_JSON_TOOL.format(_SURROGATE))

    events = await _run(_loop(ToolMode.JSON, stream=True))

    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    body = json.loads(requests[1].content)
    assert body["stream"] is True
    assert body["messages"][2] == {"role": "assistant", "content": _JSON_TOOL.format(_ESCAPE)}
    _ends_on(events, "done")


# --- B8: an assistant message the caller passes in -----------------------------------------------


async def test_host_supplied_assistant_message_is_sent_escaped(httpx_mock: HTTPXMock) -> None:
    """An assistant message passed to ``run()`` holding U+D800 is sent with ``\\ud800``; the caller's message objects are not changed (BR-024).

    The escape follows the role, not where the text came from. On the
    ``6d64a22`` ``src`` this run raised ``UnicodeEncodeError`` before the
    first request was sent (BR-024 evidence, P4). Release dependence: see
    the module docstring.
    """
    _respond(httpx_mock, json.dumps(_completion("done")).encode())
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="a" + _SURROGATE + "b"),
        ChatMessage(role="user", content="next"),
    ]

    events = await _run(_loop(ToolMode.NATIVE), list(history))

    assert len(httpx_mock.get_requests()) == 1
    sent = _sent_messages(httpx_mock, 0)
    assert sent[2] == {"role": "assistant", "content": "a" + _ESCAPE + "b"}
    assert history[1].content == "a" + _SURROGATE + "b"
    _ends_on(events, "done")


# --- B9: user and system messages (the documented residual) --------------------------------------


@pytest.mark.parametrize("role", ["user", "system"])
async def test_host_user_and_system_messages_holding_a_surrogate_still_raise_raw(
    httpx_mock: HTTPXMock, role: str
) -> None:
    """A user or system message passed to ``run()`` holding U+D800 still raises ``UnicodeEncodeError`` raw, with nothing sent (BR-024 residual).

    Pins a documented gap, not a decision: BR-024 escapes only the fields
    the model writes, and these are host text. The exception comes from the
    ``openai`` client's strict UTF-8 body encode (``openai`` 2.43.0 and
    2.54.0; BR-024 evidence, P4 rows c1). If this test changes, the
    CHANGELOG's BR-024 *Not covered* item must change with it.
    """
    if role == "user":
        messages = [ChatMessage(role="user", content="a" + _SURROGATE + "b")]
    else:
        messages = [
            ChatMessage(role="system", content="s" + _SURROGATE),
            ChatMessage(role="user", content="q"),
        ]

    with pytest.raises(UnicodeEncodeError) as exc:
        await _run(_loop(ToolMode.NATIVE), messages)

    assert exc.value.encoding == "utf-8"
    assert httpx_mock.get_requests() == []


# --- B11: a final answer that is not re-sent -----------------------------------------------------


async def test_native_final_answer_keeps_the_provider_text(httpx_mock: HTTPXMock) -> None:
    """A NATIVE final answer holding U+D800 reaches ``FinalEvent`` as the provider wrote it (BR-024, AC-3).

    Control: no further request carries it, so this run completed before
    BR-024 too (BR-024 evidence, P4). An escape where the provider's text is
    decoded would show here as the six characters.
    """
    _respond(httpx_mock, _escaped_body(_completion("final " + _SURROGATE)))

    events = await _run(_loop(ToolMode.NATIVE))

    assert len(httpx_mock.get_requests()) == 1
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.text == "final " + _SURROGATE
    assert final.raw_completion == "final " + _SURROGATE


# --- B12: the Runner replays a stored final answer ------------------------------------------------


async def test_runner_replays_a_persisted_completion_escaped(httpx_mock: HTTPXMock) -> None:
    """A NATIVE final answer holding U+D800 is stored as it was, and the next turn sends it with ``\\ud800`` (BR-024).

    ``AgentRunner`` with ``MemoryStateStore``. Run 1 ends on the answer and
    the store keeps the provider's text. Run 2 replays the history, whose
    assistant message holds the code point. On the ``6d64a22`` ``src`` run 2
    raised ``UnicodeEncodeError`` before its first request was sent (BR-024
    evidence, P4). Release dependence: see the module docstring.
    """
    _respond(httpx_mock, _escaped_body(_completion("final " + _SURROGATE)))
    _respond(httpx_mock, json.dumps(_completion("done")).encode())
    store = MemoryStateStore()
    runner = AgentRunner(loop=_loop(ToolMode.NATIVE), state=store)

    first = [event async for event in runner.run("s1", "first question")]

    assert isinstance(first[-1], FinalEvent) and first[-1].text == "final " + _SURROGATE
    stored = await store.get_messages("s1")
    assert [(m.role, m.content) for m in stored] == [
        ("user", "first question"),
        ("assistant", "final " + _SURROGATE),
    ]

    second = [event async for event in runner.run("s1", "second question")]

    assert len(httpx_mock.get_requests()) == 2
    replayed = _sent_messages(httpx_mock, 1)
    assert replayed[1:] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "final " + _ESCAPE},
        {"role": "user", "content": "second question"},
    ]
    _ends_on(second, "done")


# --- B13/B14: the loop's tool_invoked log line under structlog's default configuration ------


@pytest.fixture
def strict_log_stream() -> Iterator[io.TextIOWrapper]:
    """structlog's default configuration, printing to a strict UTF-8 stream instead of stdout.

    The SDK never configures structlog, and its default ``PrintLogger``
    prints to ``sys.stdout``. Under pytest that is the capture stream, which
    writes with ``errors="replace"`` (BR-024 evidence, round 2), so a log
    line holding a surrogate code point passes there and fails on a real
    stdout. This fixture keeps every other default (processors, wrapper
    class, context class, no logger cache) and gives ``PrintLoggerFactory``
    a ``TextIOWrapper`` over ``BytesIO`` with ``errors="strict"``, as
    ``sys.stdout`` is under a UTF-8 locale (measured: a file and a TTY). It
    restores the previous configuration values afterwards (when structlog
    was unconfigured, the processor list comes back as an equal new list).
    """
    was_configured = structlog.is_configured()
    previous = structlog.get_config()
    structlog.reset_defaults()
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict", write_through=True)
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=stream))
    try:
        yield stream
    finally:
        structlog.reset_defaults()
        if was_configured:
            structlog.configure(**previous)


def _logged_tool_names(stream: io.TextIOWrapper) -> list[str]:
    """The ``name`` value of every ``tool_invoked`` line, ANSI colour codes removed."""
    assert isinstance(stream.buffer, io.BytesIO)
    text = re.sub(r"\x1b\[[0-9;]*m", "", stream.buffer.getvalue().decode("utf-8"))
    lines = [line for line in text.splitlines() if " tool_invoked " in line]
    names = []
    for line in lines:
        match = re.search(r" name=(\S+)", line)
        assert match is not None, line
        names.append(match.group(1))
    return names


@pytest.mark.parametrize("row", ["single", "batch", "legacy_json_parser"])
async def test_tool_invoked_log_line_writes_the_tool_name_escaped(
    httpx_mock: HTTPXMock, strict_log_stream: io.TextIOWrapper, row: str
) -> None:
    """Under structlog's default configuration, the ``tool_invoked`` debug line writes a model tool name holding U+D800 as ``find\\ud800``, and the run ends on the model's answer (BR-024, AC-6).

    Rows: a native call (the loop's single-call ``tool_invoked`` line), a
    native batch whose second call holds the code point (the batch line),
    and the legacy ``JsonModeParser`` path, where the parser decodes the
    escape's six characters into the code point (the single-call line). The
    default ``ConsoleRenderer`` writes a string value without a space, tab,
    ``=``, quote or line break as it is (by reading
    ``structlog/dev.py:923-926``), so on the ``6d64a22`` ``src`` each run
    raised ``UnicodeEncodeError`` from structlog's ``print`` before the
    second request, with stdout a file or a TTY under a UTF-8 locale (BR-024
    evidence, round 2). The name is now
    written as it goes out on the wire. This test writes to its own strict
    stream (``strict_log_stream``), so pytest's output capture cannot hide
    the raise. Release dependence: see the module docstring.
    """
    name = "find" + _SURROGATE
    if row == "legacy_json_parser":
        completion = (
            '{"thought": "calling", "action": "tool", "tool_name": "find\\ud800", '
            '"tool_args": {"q": "x"}, "answer": null}'
        )
        _respond(httpx_mock, json.dumps(_completion(completion)).encode())
        _respond(httpx_mock, json.dumps(_completion(_JSON_FINAL)).encode())
        loop = _loop(None)
    else:
        calls = (
            [_call("lookup", "call_1"), _call(name, "call_2")] if row == "batch" else [_call(name)]
        )
        _respond(httpx_mock, _escaped_body(_completion(None, calls)))
        _respond(httpx_mock, json.dumps(_completion("done")).encode())
        loop = _loop(ToolMode.NATIVE)

    events = await _run(loop)

    expected = ["lookup", "find" + _ESCAPE] if row == "batch" else ["find" + _ESCAPE]
    assert _logged_tool_names(strict_log_stream) == expected
    assert len(httpx_mock.get_requests()) == 2
    action = [e.tool_name for e in events if isinstance(e, ActionEvent)]
    assert action[-1] == name  # the event keeps the model's name
    _ends_on(events, "done")


@pytest.mark.parametrize("row", ["batch", "single"])
async def test_tool_invoked_log_line_keeps_other_names_as_they_are(
    httpx_mock: HTTPXMock, strict_log_stream: io.TextIOWrapper, row: str
) -> None:
    """Control: a tool name without a surrogate code point, ASCII or Arabic, is logged as the model wrote it (BR-024, AC-6).

    Rows: a native batch calling ``lookup`` (registered) and ``بحث`` (not
    registered, so a ``ToolNotFound`` observation), which reaches the batch
    ``tool_invoked`` line, and a native call to ``بحث`` alone, which reaches
    the single-call line. Under structlog's default configuration each line
    carries the name unchanged: the escape touches only surrogate code
    points.
    """
    calls = (
        [_call("lookup", "call_1"), _call("بحث", "call_2")] if row == "batch" else [_call("بحث")]
    )
    _respond(httpx_mock, json.dumps(_completion(None, calls)).encode())
    _respond(httpx_mock, json.dumps(_completion("done")).encode())

    events = await _run(_loop(ToolMode.NATIVE))

    expected = ["lookup", "بحث"] if row == "batch" else ["بحث"]
    assert _logged_tool_names(strict_log_stream) == expected
    _ends_on(events, "done")
