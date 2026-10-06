"""Model-written text holding a surrogate code point, on the request wire (BR-024).

``OpenAICompatibleClient._serialize_message`` writes each surrogate code point
(U+D800-U+DFFF) as its six-character ``\\udXXX`` escape in three fields: an
assistant message's ``content`` (the native envelope and the
``model_dump`` branch), each ``tool_calls[].function.name`` of an assistant
message, and a ``"tool"`` message's ``name``. The rule follows the role. Every
other field, and text without a surrogate code point, is serialised as
before.

What these pin: the exact escaped value of each field, with every other key
and value equal to the pre-BR-024 form (B1); that user and system text, a
``name`` on a user or assistant message and a ``"tool"`` message's content
keep the code point (B2, a documented gap); that a message without a
surrogate code point serialises to a dict equal to the pre-BR-024 form,
with the same key order and, for the three escaped fields, the same string
objects (B3); and that the client hands the provider's text back unescaped
(B10, through the real ``openai`` client on pytest-httpx).

What these do NOT pin: HTTP bytes (BR-024 evidence, P5b, compared them
outside the suite); a run through ``AgentLoop`` (in
``tests/loop/test_loop_provider_text_replay.py``); how another ``openai``
release encodes the body.

Bodies holding a surrogate code point are sent as ``content=`` bytes from
``json.dumps`` (its default ``ensure_ascii=True`` writes the JSON escape),
never with ``add_response(json=...)``: httpx 0.28.1 encodes a ``json=`` body
as strict UTF-8 (``httpx/_content.py:176-179``, by reading), so the mock
response itself would fail to build.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk.llm.openai_compat import OpenAICompatibleClient
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest, ToolCall

BASE_URL = "https://example.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"
MODEL = "gpt-4o"
_SURROGATE = chr(0xD800)

_serialize = OpenAICompatibleClient._serialize_message


def _base_form(msg: ChatMessage) -> dict[str, Any]:
    """The pre-BR-024 serialised form: the native envelope, or ``model_dump`` with no escape."""
    if msg.role == "assistant" and msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id if tc.id is not None else msg.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.args, ensure_ascii=False),
                    },
                }
                for tc in msg.tool_calls
            ],
        }
    return msg.model_dump(exclude_none=True)


def _encodes(serialised: dict[str, Any]) -> None:
    json.dumps(serialised, ensure_ascii=False).encode("utf-8")  # must not raise


# --- B1: the three model-written fields --------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param(
            ChatMessage(role="assistant", content="a" + _SURROGATE + "b"),
            {"role": "assistant", "content": "a\\ud800b"},
            id="assistant_plain",
        ),
        pytest.param(
            ChatMessage(
                role="assistant",
                content="think " + _SURROGATE,
                tool_call_id="c1",
                tool_calls=[ToolCall(name="lookup", args={"q": "x"})],
            ),
            {
                "role": "assistant",
                "content": "think \\ud800",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "x"}'},
                    }
                ],
            },
            id="assistant_native_content",
        ),
        pytest.param(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(name="find" + _SURROGATE, args={}, id="c1"),
                    ToolCall(name="get" + chr(0xDC80), args={"k": 1}, id="c2"),
                ],
            ),
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "find\\ud800", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "get\\udc80", "arguments": '{"k": 1}'},
                    },
                ],
            },
            id="tool_calls_names",
        ),
        pytest.param(
            ChatMessage(role="tool", content="ok", tool_call_id="c1", name="find" + _SURROGATE),
            {"role": "tool", "content": "ok", "name": "find\\ud800", "tool_call_id": "c1"},
            id="tool_name",
        ),
    ],
)
def test_serialize_message_escapes_model_written_fields(
    message: ChatMessage, expected: dict[str, Any]
) -> None:
    """Each surrogate code point in an assistant ``content``, a tool call's ``name`` or a ``"tool"`` ``name`` is sent as its ``\\udXXX`` escape; nothing else changes (BR-024).

    The expected dicts are literals, keys in the order the serialiser writes
    them, so the escaped value, every other key and value, and the key order
    are all pinned. Before BR-024 each of these dicts held the code point
    itself, and the ``openai`` client's strict UTF-8 encode raised
    ``UnicodeEncodeError`` (BR-024 evidence, P4). This pins the serialiser
    on any ``openai`` release.
    """
    serialised = _serialize(message)

    assert serialised == expected
    assert list(serialised) == list(expected)
    _encodes(serialised)


# --- B2: host-role text is sent as it is (the documented residual) ----------------------


@pytest.mark.parametrize(
    ("message", "field"),
    [
        pytest.param(
            ChatMessage(role="user", content="a" + _SURROGATE + "b"), "content", id="user_content"
        ),
        pytest.param(
            ChatMessage(role="system", content="s" + _SURROGATE), "content", id="system_content"
        ),
        pytest.param(
            ChatMessage(role="user", content="q", name="n" + _SURROGATE), "name", id="user_name"
        ),
        pytest.param(
            ChatMessage(role="tool", content="t" + _SURROGATE, tool_call_id="c1", name="lookup"),
            "content",
            id="host_tool_content",
        ),
        pytest.param(
            ChatMessage(role="assistant", content="ok", name="n" + _SURROGATE),
            "name",
            id="assistant_name",
        ),
    ],
)
def test_serialize_message_sends_host_role_text_as_it_is(message: ChatMessage, field: str) -> None:
    """A user or system message, a ``name`` on a user or assistant message, and a ``"tool"`` message's content keep the code point (BR-024 residual).

    Pins a documented gap, not a decision. BR-024 escapes only the fields the
    model writes (an assistant message's content, a tool call's name, a tool
    reply's name). These fields are host text, so the shipped client still
    raises ``UnicodeEncodeError`` with nothing sent when one holds a
    surrogate code point: measured for user and system content, a
    ``"tool"`` message's content and a ``name`` on a user, assistant or
    system message (BR-024 evidence, P4 rows c1 and c1b, and round 2).
    ``test_request_encoding_errors_stay_raw_through_the_real_client``
    and
    ``test_host_user_and_system_messages_holding_a_surrogate_still_raise_raw``
    pin the raise. A ``"tool"`` message the loop builds is already escaped
    by the loop (BR-022); only one passed to ``run()`` reaches the client
    with the code point. If this test changes, the CHANGELOG's BR-024 *Not
    covered* item must change with it.
    """
    serialised = _serialize(message)

    assert serialised == message.model_dump(exclude_none=True)
    assert _SURROGATE in serialised[field]


# --- B3: text without a surrogate code point is unchanged --------------------------------

_TEXTS = [
    pytest.param("فاطمة الزهراء", id="arabic"),
    pytest.param("astral 😀 emoji", id="astral_emoji"),
    pytest.param("del \x7f here", id="u007f"),
    pytest.param("the six characters \\ud800 typed literally", id="literal_escape_text"),
]


def _shapes(text: str) -> dict[str, ChatMessage]:
    return {
        "system": ChatMessage(role="system", content=text),
        "user": ChatMessage(role="user", content=text),
        "user_named": ChatMessage(role="user", content=text, name="u" + text),
        "assistant": ChatMessage(role="assistant", content=text),
        "assistant_named": ChatMessage(role="assistant", content=text, name="helper"),
        "assistant_empty_tool_calls": ChatMessage(role="assistant", content=text, tool_calls=[]),
        "tool_named": ChatMessage(role="tool", content=text, tool_call_id="c1", name="look" + text),
        "tool_unnamed": ChatMessage(role="tool", content=text, tool_call_id="c1"),
        "native_envelope": ChatMessage(
            role="assistant",
            content=text,
            tool_calls=[
                ToolCall(name="a" + text, args={"q": text}, id="c1"),
                ToolCall(name="b" + text, args={}, id="c2"),
            ],
        ),
    }


_SHAPE_IDS = list(_shapes("x"))


@pytest.mark.parametrize("shape", _SHAPE_IDS)
@pytest.mark.parametrize("text", _TEXTS)
def test_serialize_message_without_surrogates_is_unchanged(text: str, shape: str) -> None:
    """A message with no surrogate code point serialises to the pre-BR-024 dict, keys in the same order, the escaped fields the same string objects (BR-024, AC-2).

    Covers every role, a ``name`` on user, assistant and tool messages, an
    assistant message with ``tool_calls=[]`` (the ``model_dump`` branch,
    where ``content`` is not the last key) and a two-call native envelope,
    each with Arabic text, an astral emoji, U+007F and the six characters of
    an escape typed literally. The order-preserving ``json.dumps`` comparison
    is the key-order pin, which the goldens cannot give (their fixtures are
    canonical JSON; BR-024 evidence, P5). Under an escape of every
    non-ASCII character (BR-024 evidence, mutant MS6), the Arabic and emoji
    rows of the five shapes whose text reaches the three fields
    (``assistant``, ``assistant_named``, ``assistant_empty_tool_calls``,
    ``tool_named``, ``native_envelope``) turn red on the values; the U+007F
    and typed-escape rows of those shapes keep equal values, because their
    text is ASCII, and turn red only on the ``is`` assertions, because that
    escape builds a new string. The other four shapes stay green: their text
    is not in the three fields.
    """
    message = _shapes(text)[shape]

    serialised = _serialize(message)

    base = _base_form(message)
    assert serialised == base
    assert json.dumps(serialised, ensure_ascii=False) == json.dumps(base, ensure_ascii=False)
    if message.role == "assistant":
        assert serialised["content"] is message.content
        for entry, tool_call in zip(
            serialised.get("tool_calls") or [], message.tool_calls or [], strict=True
        ):
            assert entry["function"]["name"] is tool_call.name
    if message.role == "tool" and message.name is not None:
        assert serialised["name"] is message.name


# --- B10: the client hands the provider's text back unescaped ----------------------------


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(api_key="test-key", base_url=BASE_URL, timeout=5.0, max_retries=0)


def _request() -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="hi")], model=MODEL)


def _completion(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _sse_with_content(content: str) -> bytes:
    chunk = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


@pytest.mark.parametrize(
    "route", ["complete_json_escape", "complete_raw_bytes", "native_tool_name", "stream_delta"]
)
async def test_client_returns_the_provider_text_unescaped(
    httpx_mock: HTTPXMock, route: str
) -> None:
    """``ChatResponse`` and stream chunks carry the provider's surrogate code point as it was; the escape exists only in requests (BR-024, AC-3).

    Routes: a completion whose ``content`` holds the JSON escape
    ``\\ud800``, the same with the UTF-8 encoded surrogate bytes
    ``ED A0 80`` spliced into the body (the ``openai`` client decodes a body
    with ``surrogatepass``; BR-023 evidence, P1 f), a native tool call whose
    name holds the JSON escape, and a stream delta holding it. How each body
    decodes is the ``openai`` client's behaviour; this test passed on 2.43.0
    and 2.54.0 (BR-024 evidence, the gate).
    """
    if route == "stream_delta":
        body = _sse_with_content("a" + _SURROGATE + "b")
        assert b"\\ud800" in body
        httpx_mock.add_response(
            method="POST", url=ENDPOINT, content=body, headers={"content-type": "text/event-stream"}
        )
        chunks = [chunk async for chunk in _client().stream(_request())]
        assert [c.message.content for c in chunks] == ["a" + _SURROGATE + "b"]
        return
    if route == "native_tool_name":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "find" + _SURROGATE, "arguments": "{}"},
                }
            ],
        }
        body = json.dumps(_completion(message)).encode()
    elif route == "complete_json_escape":
        body = json.dumps(
            _completion({"role": "assistant", "content": "a" + _SURROGATE + "b"})
        ).encode()
    else:
        marked = json.dumps(_completion({"role": "assistant", "content": "aXMARKXb"})).encode()
        assert marked.count(b"XMARKX") == 1
        body = marked.replace(b"XMARKX", b"\xed\xa0\x80")
    if route != "complete_raw_bytes":
        assert b"\\ud800" in body
    httpx_mock.add_response(
        method="POST", url=ENDPOINT, content=body, headers={"content-type": "application/json"}
    )

    response = await _client().complete(_request())

    if route == "native_tool_name":
        assert response.message.tool_calls is not None
        assert [tc.name for tc in response.message.tool_calls] == ["find" + _SURROGATE]
    else:
        assert response.message.content == "a" + _SURROGATE + "b"
