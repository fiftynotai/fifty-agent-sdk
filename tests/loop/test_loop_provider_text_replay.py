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
raise. Since BR-026 the line carries the name under ``tool_name``
(``name`` before, a key stdlib's ``LogRecord`` reserves); B15 pins the
escaped value on the stdlib record under structlog's
``render_to_log_kwargs`` recipe.

Since BR-028 the line also writes each control character (U+0000-U+001F,
U+007F-U+009F) in the name as a backslash, ``u`` and four lowercase hex
digits. Before, structlog's default configuration wrote such a name as it
was: measured with structlog 26.1.0 and 24.1.0 on CPython 3.11.15, 3.13.2
and 3.14.3, with stdout a file and a terminal (BR-028 evidence, P1b), apart
from tab, LF and CR on 26.1.0 (P1; B16 has the LF row), which its renderer
writes with ``repr``; 24.1.0 wrote LF as a line break. On an ASCII stream a
name holding a C1 control character, and none of the seven characters below
on 26.1.0, made the line raise ``UnicodeEncodeError`` (P3). B16 pins the
escape for ESC sequences, U+009B, DEL, NUL, VT and LF on both lines, for the same
payloads but VT on the legacy parser path (``JsonModeParser`` strips the
name), and with a colour renderer; B17 is its control (names without
a control character, U+202E, NBSP, U+200B and U+2028 among them, logged
exactly as written); B18 pins the ASCII stream (two C1 names complete, ESC
is the control, an Arabic name still raises, and row ``e``, a tab and
U+202E, pins the raise BR-028 adds on structlog 26.1.0); B19 pins the escaped value on
the stdlib record under ``render_to_log_kwargs`` and
``wrap_for_formatter``. Their payloads hold none of the seven characters
for which 26.1.0's renderer switches to ``repr`` (space, tab, ``=``, the
two quotes, CR, LF), apart from the LF row and B18's row ``e``, whose tab is
the point of that row, because a value written with
``repr`` hides a missing escape. Their parser, ``_tool_invoked_lines``,
splits on ``"\\n"`` only and removes only SGR colour codes:
``splitlines()`` would split on VT, FF, NEL and U+2028, and a wider ANSI
strip would remove a raw ``ESC [2J``; B13 and B14 keep their own parser.

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
documented gap, B9); log output under structlog configurations other
than the default one B13, B14, B16-B18 use (B16 also with
``ConsoleRenderer(colors=True)``), and the ``render_to_log_kwargs`` and
``wrap_for_formatter`` stdlib recipes B15 and B19 use; and streams other
than strict UTF-8 and ASCII.
"""

from __future__ import annotations

import io
import json
import logging
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
from tests.stdlib_routing import record_field, records_for, stdlib_routed_structlog

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
def strict_log_stream(request: pytest.FixtureRequest) -> Iterator[io.TextIOWrapper]:
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

    An indirect parameter names another encoding (BR-028's B18 uses
    ``"ascii"``, as a stdout under ``PYTHONIOENCODING=ascii`` is); without
    one the stream is UTF-8, as B13 and B14 use it.
    """
    encoding = getattr(request, "param", "utf-8")
    was_configured = structlog.is_configured()
    previous = structlog.get_config()
    structlog.reset_defaults()
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict", write_through=True)
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=stream))
    try:
        yield stream
    finally:
        structlog.reset_defaults()
        if was_configured:
            structlog.configure(**previous)


def _logged_tool_names(stream: io.TextIOWrapper) -> list[str]:
    """The ``tool_name`` value (``name`` before BR-026) of every ``tool_invoked`` line, ANSI colour codes removed.

    Also asserts that no ``tool_invoked`` line still writes a ``name`` key.
    """
    assert isinstance(stream.buffer, io.BytesIO)
    text = re.sub(r"\x1b\[[0-9;]*m", "", stream.buffer.getvalue().decode("utf-8"))
    lines = [line for line in text.splitlines() if " tool_invoked " in line]
    names = []
    for line in lines:
        assert " name=" not in line, line
        match = re.search(r" tool_name=(\S+)", line)
        assert match is not None, line
        names.append(match.group(1))
    return names


# The ``tool_invoked`` names B13 and B15 expect per row: the escape where the model's name held U+D800.
_ESCAPED_NAMES = {
    "single": ["find" + _ESCAPE],
    "batch": ["lookup", "find" + _ESCAPE],
    "legacy_json_parser": ["find" + _ESCAPE],
}


def _queue_surrogate_name_row(httpx_mock: HTTPXMock, row: str) -> AgentLoop:
    """Queue ``row``'s two responses (a tool call to ``find`` plus U+D800, then ``done``) and build its loop."""
    if row == "legacy_json_parser":
        completion = (
            '{"thought": "calling", "action": "tool", "tool_name": "find\\ud800", '
            '"tool_args": {"q": "x"}, "answer": null}'
        )
        _respond(httpx_mock, json.dumps(_completion(completion)).encode())
        _respond(httpx_mock, json.dumps(_completion(_JSON_FINAL)).encode())
        return _loop(None)
    name = "find" + _SURROGATE
    calls = [_call("lookup", "call_1"), _call(name, "call_2")] if row == "batch" else [_call(name)]
    _respond(httpx_mock, _escaped_body(_completion(None, calls)))
    _respond(httpx_mock, json.dumps(_completion("done")).encode())
    return _loop(ToolMode.NATIVE)


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
    written as it goes out on the wire, as the line's ``tool_name`` value
    (``name`` before BR-026; the helper also asserts no line writes
    ``name``). This test writes to its own strict
    stream (``strict_log_stream``), so pytest's output capture cannot hide
    the raise. Release dependence: see the module docstring.
    """
    loop = _queue_surrogate_name_row(httpx_mock, row)

    events = await _run(loop)

    assert _logged_tool_names(strict_log_stream) == _ESCAPED_NAMES[row]
    assert len(httpx_mock.get_requests()) == 2
    action = [e.tool_name for e in events if isinstance(e, ActionEvent)]
    assert action[-1] == "find" + _SURROGATE  # the event keeps the model's name
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
    carries the name unchanged under ``tool_name``: the escape touches only
    surrogate code points and, since BR-028, control characters, and these
    names hold neither.
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


@pytest.mark.parametrize("row", ["single", "batch", "legacy_json_parser"])
async def test_tool_invoked_record_carries_the_escaped_tool_name_under_stdlib_routing(
    httpx_mock: HTTPXMock, row: str
) -> None:
    """B15: under structlog's ``render_to_log_kwargs`` recipe at DEBUG, the ``tool_invoked`` record's ``tool_name`` attribute holds the escape, and the run ends on ``done`` (BR-024 AC-6, BR-026 AC-3).

    B13's rows and responses, through the real client. Before BR-026 each
    run raised ``KeyError`` for ``'name'`` at the line instead (BR-026
    evidence, P2). The record's ``name`` is the logger's name.
    """
    loop = _queue_surrogate_name_row(httpx_mock, row)

    with stdlib_routed_structlog("render_to_log_kwargs", logging.DEBUG) as records:
        events = await _run(loop)

    invoked = records_for(records, "tool_invoked")
    assert [record.__dict__["tool_name"] for record in invoked] == _ESCAPED_NAMES[row]
    assert all(record.name == "fifty_agent_sdk.loop" for record in invoked)
    assert len(httpx_mock.get_requests()) == 2
    action = [e.tool_name for e in events if isinstance(e, ActionEvent)]
    assert action[-1] == "find" + _SURROGATE
    _ends_on(events, "done")


# --- B16-B19: control characters in the tool_invoked line (BR-028) --------------------------

# Each payload holds none of the seven characters for which structlog 26.1.0's ConsoleRenderer
# switches to ``repr`` (space, tab, ``=``, ``"``, ``'``, CR, LF), apart from ``lf``: a value it
# writes with ``repr`` would hide a missing escape. Each maps to the model's name and the value
# the line must carry, a backslash, ``u`` and four lowercase hex digits per control character.
_CONTROL_NAMES: dict[str, tuple[str, str]] = {
    "csi": ("find\x1b[2J", "find\\u001b[2J"),
    "osc": ("find\x1b]0;x\x07", "find\\u001b]0;x\\u0007"),
    "c1_csi": ("find\x9b2J", "find\\u009b2J"),
    "del": ("find\x7f", "find\\u007f"),
    "nul": ("find\x00", "find\\u0000"),
    "vt": ("find\x0b", "find\\u000b"),
    "lf": ("find\nx", "find\\u000ax"),
}

_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _is_control(ch: str) -> bool:
    return ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F


def _tool_invoked_lines(stream: io.TextIOWrapper) -> list[str]:
    """The ``tool_invoked`` lines, split on ``"\\n"`` only and with only SGR colour codes removed.

    Not ``splitlines()``, which also splits on VT, FF, the C1 NEL, U+2028
    and others, and not a wider ANSI strip, which would remove a raw
    ``ESC [2J`` too: either could hide a control character the line wrote.
    """
    assert isinstance(stream.buffer, io.BytesIO)
    text = _SGR.sub("", stream.buffer.getvalue().decode("utf-8"))
    return [line for line in text.split("\n") if " tool_invoked " in line]


def _tool_name_value(line: str) -> str:
    """The rest of the line after `` tool_name=``: the renderer sorts keys, so ``tool_name`` is last."""
    _, separator, value = line.partition(" tool_name=")
    assert separator, line
    return value


def _queue_named_call(httpx_mock: HTTPXMock, row: str, name: str) -> AgentLoop:
    """Queue ``row``'s two responses (a call to ``name``, then ``done``) and build its loop.

    ``legacy_json_parser``: a JSON-mode completion whose ``tool_name`` is
    ``json.dumps``' escape of ``name``, which ``JsonModeParser`` decodes.
    """
    if row == "legacy_json_parser":
        envelope = {
            "thought": "calling",
            "action": "tool",
            "tool_name": name,
            "tool_args": {"q": "x"},
            "answer": None,
        }
        completion = json.dumps(envelope)
        # Precondition: the completion carries the control characters as JSON escapes.
        assert completion.isascii() and not any(_is_control(ch) for ch in completion)
        _respond(httpx_mock, json.dumps(_completion(completion)).encode())
        _respond(httpx_mock, json.dumps(_completion(_JSON_FINAL)).encode())
        return _loop(None)
    calls = [_call("lookup", "call_1"), _call(name, "call_2")] if row == "batch" else [_call(name)]
    _respond(httpx_mock, json.dumps(_completion(None, calls)).encode())
    _respond(httpx_mock, json.dumps(_completion("done")).encode())
    return _loop(ToolMode.NATIVE)


def _wire_name(httpx_mock: HTTPXMock, row: str) -> str:
    """The model's name as request 2 sends it: the last tool call's name, or the tool reply's name."""
    messages = _sent_messages(httpx_mock, 1)
    if row == "legacy_json_parser":
        return str([m for m in messages if m["role"] == "tool"][-1]["name"])
    calls = [tc for m in messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
    return str(calls[-1]["function"]["name"])


_B16_ROWS = [
    *[
        pytest.param(row, payload, "default", id=f"{row}-{payload}")
        for row in ("single", "batch", "legacy_json_parser")
        for payload in _CONTROL_NAMES
        # JsonModeParser strips the tool name (`str.strip`), which removes a trailing VT.
        if not (row == "legacy_json_parser" and payload == "vt")
    ],
    pytest.param("single", "csi", "colors", id="single-csi-colors"),
]


@pytest.mark.parametrize(("row", "payload", "renderer"), _B16_ROWS)
async def test_tool_invoked_writes_control_characters_escaped(
    httpx_mock: HTTPXMock,
    strict_log_stream: io.TextIOWrapper,
    row: str,
    payload: str,
    renderer: str,
) -> None:
    """B16: a model tool name holding a control character is written to the ``tool_invoked`` line escaped, and the events and the request keep the model's name (BR-028).

    Rows: a native call (the single-call line), a native batch whose second
    call holds the payload (the batch line) and the legacy ``JsonModeParser``
    path (the single-call line; no ``vt`` row, because the parser strips the
    name). Payloads: ESC ``[2J``, an OSC title sequence ending in BEL, the C1
    CSI U+009B, DEL, NUL, VT and LF. The ``colors`` row replaces the default
    chain's renderer with ``ConsoleRenderer(colors=True)``, as a host whose
    stdout was a terminal when structlog was imported has it.

    Before BR-028, structlog's default configuration wrote the name as it
    was: measured with stdout a file and a terminal, structlog 26.1.0 and
    24.1.0, CPython 3.11.15, 3.13.2 and 3.14.3 (BR-028 evidence, P1b), apart
    from LF, which 26.1.0 writes with ``repr`` and 24.1.0 as a line break.
    """
    name, expected = _CONTROL_NAMES[payload]
    loop = _queue_named_call(httpx_mock, row, name)
    if renderer == "colors":
        processors = list(structlog.get_config()["processors"])
        structlog.configure(
            processors=[*processors[:-1], structlog.dev.ConsoleRenderer(colors=True)]
        )

    events = await _run(loop)

    lines = _tool_invoked_lines(strict_log_stream)
    assert len(lines) == (2 if row == "batch" else 1)
    if row == "batch":
        assert _tool_name_value(lines[0]) == "lookup"
    assert _tool_name_value(lines[-1]) == expected
    assert not any(_is_control(ch) for line in lines for ch in line)
    if renderer == "colors":
        # Precondition: the renderer wrote colour codes, which the parser removed.
        assert b"\x1b[0m" in strict_log_stream.buffer.getvalue()  # type: ignore[attr-defined]
    assert len(httpx_mock.get_requests()) == 2
    action = [e.tool_name for e in events if isinstance(e, ActionEvent)]
    assert action[-1] == name
    assert _wire_name(httpx_mock, row) == name
    _ends_on(events, "done")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("lookup", id="ascii"),
        pytest.param("بحث", id="arabic"),
        pytest.param("find\\u001b", id="typed_escape"),
        pytest.param("find‮x", id="rlo_u202e"),
        pytest.param("find ", id="nbsp"),
        pytest.param("find​", id="zwsp"),
        pytest.param("find x", id="line_separator"),
    ],
)
@pytest.mark.parametrize("row", ["single", "batch"])
async def test_tool_invoked_keeps_names_without_control_characters(
    httpx_mock: HTTPXMock, strict_log_stream: io.TextIOWrapper, row: str, name: str
) -> None:
    """B17, control: a tool name without a control character is logged exactly as the model wrote it (BR-028).

    The rows hold none of the seven characters 26.1.0's renderer writes with
    ``repr``. Among them: the six characters of the ESC escape typed (the
    escape cannot be told apart from them), U+202E, which BR-028 leaves as
    it is (decision D1), NBSP, a zero-width space and U+2028, which a
    ``splitlines()`` parser would split on.
    """
    loop = _queue_named_call(httpx_mock, row, name)

    events = await _run(loop)

    lines = _tool_invoked_lines(strict_log_stream)
    assert len(lines) == (2 if row == "batch" else 1)
    assert _tool_name_value(lines[-1]) == name
    assert len(httpx_mock.get_requests()) == 2
    _ends_on(events, "done")


@pytest.mark.parametrize("strict_log_stream", ["ascii"], indirect=True)
@pytest.mark.parametrize(
    ("row", "name", "expected"),
    [
        pytest.param("a", "find\x85", "find\\u0085", id="a-nel"),
        pytest.param("b", "find\x9b2J", "find\\u009b2J", id="b-c1_csi"),
        pytest.param("c", "find\x1b[2J", None, id="c-csi_control"),
        pytest.param("d", "بحث", None, id="d-arabic_residual"),
        pytest.param("e", "find\t\u202ex", None, id="e-tab_rlo_residual"),
    ],
)
async def test_tool_invoked_with_a_c1_name_completes_on_a_non_utf8_stream(
    httpx_mock: HTTPXMock,
    strict_log_stream: io.TextIOWrapper,
    row: str,
    name: str,
    expected: str | None,
) -> None:
    """B18: on an ASCII stream, a native call whose name holds a C1 control character completes, with the name escaped (BR-028).

    Before BR-028, rows ``a`` and ``b`` raised ``UnicodeEncodeError`` from
    the ``tool_invoked`` line out of ``AgentLoop.run()`` after one request,
    with no ``FinalEvent`` (BR-028 evidence, P3: ASCII and cp1252 streams,
    row ``a`` also with a real stdout under ``PYTHONIOENCODING=ascii``). Row
    ``c`` (ESC, which ASCII can encode) is the control and completes before
    and after. Row ``d`` is the residual: an Arabic name is not a control
    character, so the line still raises on such a stream. Row ``e`` is the
    raise BR-028 adds on structlog 26.1.0: ``find`` + TAB + U+202E + ``x``.
    Before, 26.1.0's renderer wrote that name with ``repr`` because of the
    tab, which escaped U+202E, and the run completed; the escape now rewrites
    the tab, so the name is written as it is and U+202E, which ASCII cannot
    encode, makes the line raise. With 24.1.0 the line raised before and
    after (BR-028 evidence, round 2).
    """
    if row in ("d", "e"):
        _respond(httpx_mock, json.dumps(_completion(None, [_call(name)])).encode())
        with pytest.raises(UnicodeEncodeError):
            await _run(_loop(ToolMode.NATIVE))
        assert len(httpx_mock.get_requests()) == 1
        return
    loop = _queue_named_call(httpx_mock, "single", name)

    events = await _run(loop)

    assert len(httpx_mock.get_requests()) == 2
    _ends_on(events, "done")
    if expected is not None:
        (line,) = _tool_invoked_lines(strict_log_stream)
        assert _tool_name_value(line) == expected


@pytest.mark.parametrize("recipe", ["render_to_log_kwargs", "wrap_for_formatter"])
async def test_tool_invoked_record_carries_control_escapes_under_stdlib_routing(
    httpx_mock: HTTPXMock, recipe: str
) -> None:
    """B19: under ``render_to_log_kwargs`` and ``wrap_for_formatter`` at DEBUG, the ``tool_invoked`` record carries the escaped name (BR-028).

    Under ``wrap_for_formatter`` the event dict is the record's ``msg``, and
    the formatter's ``ConsoleRenderer`` output carries the escape too.
    """
    name, expected = _CONTROL_NAMES["csi"]
    loop = _queue_named_call(httpx_mock, "single", name)

    with stdlib_routed_structlog(recipe, logging.DEBUG) as records:
        events = await _run(loop)

    (record,) = records_for(records, "tool_invoked")
    assert record_field(record, "tool_name") == expected
    if recipe == "wrap_for_formatter":
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False),
            ]
        )
        rendered = formatter.format(record)
        assert rendered.endswith(" tool_name=" + expected)
    assert len(httpx_mock.get_requests()) == 2
    _ends_on(events, "done")
