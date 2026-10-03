"""Provider bodies the client cannot decode, and wrong-typed fields (BR-023).

``complete()`` asks the ``openai`` client for the raw response
(``with_raw_response``) and decodes it in a separate step. A failure of that
step (a body labelled JSON that is not valid UTF-8, holds an integer literal over
the interpreter's int-to-str digit limit, or nests deeper than ``json.loads``
decodes) is an ``LLMError`` typed ``UndecodableProviderBody`` quoting the body.
``stream()`` types the same three failures of the ``openai`` event iterator
alike. A 200 whose fields have the wrong JSON type is ``MalformedResponse`` with
a fixed message. The request step keeps only the ``openai`` arms, so a request
that fails to encode still raises raw.

Every test drives the real ``openai`` client through ``pytest-httpx``
(``max_retries=0``, no network), with bodies sent as bytes (one well-formed
completion in the header test is sent with ``json=``). The digit limit
comes from ``sys.get_int_max_str_digits()``; depths are measured in each test's
own frame, with inline loops (BR-019's lesson), and no figure is asserted.

Release dependence: which exception the ``openai`` client raises for each body,
and at which step, is the installed release's behaviour. It was measured on
2.43.0 (CPython 3.14.3) and 2.54.0 (CPython 3.11.15 and 3.13.2), BR-023
evidence P1 and P2. The ``__cause__`` assertions would turn red on a release
that surfaced a body another way. The release dependence of the raw-encoding
pins (P7b, c1) is the same as BR-022's.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.openai_compat import OpenAICompatibleClient
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest

BASE_URL = "https://example.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"
MODEL = "gpt-4o"

# The fixed message prefixes, written out here so a change to them is a
# deliberate, visible change to this contract.
_UNDECODABLE_PREFIX = "Provider response body could not be decoded: "
_UNDECODABLE_STREAM_PREFIX = "Provider stream could not be decoded: "
_NON_JSON_PREFIX = "Provider response body could not be read as a JSON object: "
_WRONG_TYPE_MESSAGE = "Malformed provider response: a field has the wrong type or value ({})."
_TRUNCATED = "…[truncated]"

_JSON_LABELS = ["application/json", "application/problem+json", "text/json"]
_SENTINEL = "SENTINEL-br023"
_DIGIT_LIMIT = sys.get_int_max_str_digits()
_NEEDS_DIGIT_LIMIT = pytest.mark.skipif(
    _DIGIT_LIMIT == 0, reason="the int-to-str digit limit is disabled (0) in this interpreter"
)


# --- Helpers -------------------------------------------------------------------------


def _client(*, api_key: str = "test-key") -> OpenAICompatibleClient:
    return OpenAICompatibleClient(api_key=api_key, base_url=BASE_URL, timeout=5.0, max_retries=0)


def _request(content: str = "hi") -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=content)], model=MODEL)


def _completion(**message: Any) -> dict[str, Any]:  # noqa: ANN401 - free-form test payload
    msg: dict[str, Any] = {"role": "assistant", "content": "hello"}
    msg.update(message)
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _with_extra_key(raw_json_value: str) -> bytes:
    """A valid completion with one more key, ``"x"``, holding ``raw_json_value`` as written."""
    text = json.dumps(_completion())
    return (text[:-1] + ', "x": ' + raw_json_value + "}").encode()


def _non_utf8_completion() -> bytes:
    """A valid completion whose ``content`` holds the byte 0xE9, which is not valid UTF-8 there."""
    body = (
        json.dumps(_completion(content=f"{_SENTINEL} cafX")).encode().replace(b"cafX", b"caf\xe9")
    )
    with pytest.raises(UnicodeDecodeError):
        body.decode("utf-8")
    return body


def _respond(httpx_mock: HTTPXMock, body: bytes, content_type: str) -> None:
    httpx_mock.add_response(
        method="POST", url=ENDPOINT, content=body, headers={"content-type": content_type}
    )


async def _complete_error(client: OpenAICompatibleClient) -> LLMError:
    with pytest.raises(LLMError) as exc:
        await client.complete(_request())
    return exc.value


async def _stream_error(client: OpenAICompatibleClient) -> LLMError:
    with pytest.raises(LLMError) as exc:
        async for _ in client.stream(_request()):
            pass
    return exc.value


def _expected_message(prefix: str, text: str) -> str:
    stripped = text.strip()
    return prefix + stripped[:500] + (_TRUNCATED if len(stripped) > 500 else "")


# --- complete(): bodies the decode step cannot decode ---------------------------------


@pytest.mark.parametrize("content_type", _JSON_LABELS)
async def test_complete_non_utf8_json_body_is_undecodable_provider_body(
    httpx_mock: HTTPXMock, content_type: str
) -> None:
    """A JSON-labelled 200 that is not valid UTF-8 is UndecodableProviderBody quoting the body (BR-023 AC-2).

    The message quotes the body's bytes read as UTF-8, in which the invalid
    byte is U+FFFD; ``body_length`` counts that text's characters. Before
    BR-023 the ``UnicodeDecodeError`` escaped ``complete()`` raw (evidence
    P1 b).
    """
    body = _non_utf8_completion()
    text = body.decode("utf-8", "replace")
    _respond(httpx_mock, body, content_type)

    err = await _complete_error(_client())

    assert isinstance(err.__cause__, UnicodeDecodeError)
    assert err.message == _UNDECODABLE_PREFIX + text.strip()
    assert "�" in err.message
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "body_length": len(text),
        "decode_error": "UnicodeDecodeError",
    }


async def test_complete_non_utf8_text_plain_body_stays_non_json_provider_body(
    httpx_mock: HTTPXMock,
) -> None:
    """The same bytes labelled text/plain stay NonJsonProviderBody: openai returns them as text (BR-023 control).

    The ``openai`` client decodes a body whose Content-Type does not end in
    ``json`` only inside its own ``try``, and returns the text when that
    fails (evidence P1 b), so the adapter gets a ``str`` (BR-021's route).
    """
    body = _non_utf8_completion()
    _respond(httpx_mock, body, "text/plain")

    err = await _complete_error(_client())

    assert err.__cause__ is None
    assert err.message == _NON_JSON_PREFIX + body.decode("utf-8", "replace").strip()
    assert err.context["type"] == "NonJsonProviderBody"


async def test_complete_undecodable_body_naming_an_overflow_is_classified(
    httpx_mock: HTTPXMock,
) -> None:
    """An undecodable body whose text names an overflow is ContextLengthExceeded (BR-023, BR-021's rule)."""
    body = b'{"error": {"message": "context_length_exceeded caf\xe9"}}'
    _respond(httpx_mock, body, "application/json")

    err = await _complete_error(_client())

    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "body_length": len(body.decode("utf-8", "replace")),
        "decode_error": "UnicodeDecodeError",
        "classified_from": "UndecodableProviderBody",
    }


@_NEEDS_DIGIT_LIMIT
@pytest.mark.parametrize("content_type", _JSON_LABELS)
async def test_complete_integer_over_the_digit_limit_is_undecodable_provider_body(
    httpx_mock: HTTPXMock, content_type: str
) -> None:
    """An integer literal one digit over the interpreter's limit is UndecodableProviderBody (BR-023 AC-1).

    The cause is the bare ``ValueError`` the C scanner raises; before BR-023
    it escaped ``complete()`` raw (evidence P1 a). The body is longer than
    500 characters, so the message is cut and marked.
    """
    body = _with_extra_key("1" * (_DIGIT_LIMIT + 1))
    text = body.decode()
    _respond(httpx_mock, body, content_type)

    err = await _complete_error(_client())

    assert type(err.__cause__) is ValueError
    assert err.message == _expected_message(_UNDECODABLE_PREFIX, text)
    assert err.message.endswith(_TRUNCATED)
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "body_length": len(text),
        "decode_error": "ValueError",
    }


@_NEEDS_DIGIT_LIMIT
async def test_complete_integer_at_the_digit_limit_still_completes(httpx_mock: HTTPXMock) -> None:
    """An integer literal of exactly the limit's digits decodes, and the completion maps (BR-023 control)."""
    _respond(httpx_mock, _with_extra_key("1" * _DIGIT_LIMIT), "application/json")

    response = await _client().complete(_request())

    assert response.message.content == "hello"
    assert response.finish_reason == "stop"


async def test_complete_body_nested_past_the_decoder_is_undecodable_provider_body(
    httpx_mock: HTTPXMock,
) -> None:
    """A body nested twice as deep as json.loads decodes in this frame is UndecodableProviderBody (BR-023 AC-7).

    ``decode_max`` is the deepest array chain ``json.loads`` decodes from
    this test's frame (doubling from 128, then a binary search, inline).
    ``complete()`` decodes deeper in the stack than this frame, so twice that
    depth fails there too. Before BR-023 the ``RecursionError`` escaped
    ``complete()`` raw (evidence P1 d).
    """
    lo, hi = 1, 128
    while True:
        try:
            json.loads("[" * hi + "]" * hi)
        except RecursionError:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            json.loads("[" * mid + "]" * mid)
            lo = mid
        except RecursionError:
            hi = mid
    depth = 2 * lo
    body = "[" * depth + "]" * depth
    with pytest.raises(RecursionError):
        json.loads(body)
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert isinstance(err.__cause__, RecursionError)
    assert err.message == _UNDECODABLE_PREFIX + "[" * 500 + _TRUNCATED
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "body_length": 2 * depth,
        "decode_error": "RecursionError",
    }


async def test_complete_deep_but_decodable_extra_field_still_completes(
    httpx_mock: HTTPXMock,
) -> None:
    """Half the depth json.loads decodes in this frame, in an extra key, still completes (BR-023 control).

    BR-023 adds no body pre-scan, so a body is not refused for its depth
    alone: this one nests half as deep as ``json.loads`` decodes here, in
    a key the adapter does not read, and maps as before. ``decode_max`` is
    measured inline in this frame, as in the test above.
    """
    lo, hi = 1, 128
    while True:
        try:
            json.loads("[" * hi + "]" * hi)
        except RecursionError:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            json.loads("[" * mid + "]" * mid)
            lo = mid
        except RecursionError:
            hi = mid
    depth = lo // 2
    assert depth > 64
    _respond(httpx_mock, _with_extra_key("[" * depth + "]" * depth), "application/json")

    response = await _client().complete(_request())

    assert response.message.content == "hello"


# --- stream(): the openai event iterator cannot decode an event -----------------------


@pytest.mark.parametrize(
    ("row", "content_type", "decode_error"),
    [
        pytest.param("digit_limit", "text/event-stream", "ValueError", id="digit_limit_data"),
        pytest.param("non_utf8", "text/event-stream", "UnicodeDecodeError", id="non_utf8_line"),
        pytest.param("deep", "text/event-stream", "RecursionError", id="deep_data"),
        pytest.param("non_utf8_body", "application/json", "UnicodeDecodeError", id="buffered_json"),
    ],
)
async def test_stream_events_the_client_cannot_decode_are_undecodable(
    httpx_mock: HTTPXMock, row: str, content_type: str, decode_error: str
) -> None:
    """A stream the openai iterator cannot decode is UndecodableProviderBody, phase stream (BR-023 AC-1, AC-2, AC-7).

    The message is the prefix plus ``str()`` of the decode error, which
    holds no provider text, so there is no ``body_length`` and no
    classification. Before BR-023 the type was the exception's class name
    and the message ``str(e)`` (evidence P1 e). The ``deep`` row's depth is
    measured in this test's frame; the ``buffered_json`` row is read ahead
    (not ``text/event-stream``) and then fails in the SSE line decoder.
    """
    if row == "digit_limit":
        if _DIGIT_LIMIT == 0:
            pytest.skip("the int-to-str digit limit is disabled (0) in this interpreter")
        body = b'data: {"x": ' + b"1" * (_DIGIT_LIMIT + 1) + b"}\n\n"
    elif row == "non_utf8":
        body = b'data: {"x": "caf\xe9"}\n\n'
    elif row == "non_utf8_body":
        body = b'{"x": "caf\xe9"}'
    else:
        lo, hi = 1, 128
        while True:
            try:
                json.loads("[" * hi + "]" * hi)
            except RecursionError:
                break
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try:
                json.loads("[" * mid + "]" * mid)
                lo = mid
            except RecursionError:
                hi = mid
        depth = 2 * lo
        body = b"data: " + b"[" * depth + b"]" * depth + b"\n\n"
    _respond(httpx_mock, body, content_type)

    err = await _stream_error(_client())

    assert type(err.__cause__).__name__ == decode_error
    assert err.message == _UNDECODABLE_STREAM_PREFIX + str(err.__cause__)
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "decode_error": decode_error,
        "phase": "stream",
    }


async def test_chunk_mapping_value_error_keeps_the_generic_wrap(httpx_mock: HTTPXMock) -> None:
    """A ValueError from mapping a decoded chunk keeps the generic wrap, not UndecodableProviderBody (BR-023 control).

    ``usage.prompt_tokens: "abc"`` decodes; ``int("abc")`` in the chunk
    mapping raises after the iterator step (evidence P1 e), so the type is
    the class name and the message ``str(e)``, unprefixed, as before BR-023.
    """
    chunk = {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": "abc", "completion_tokens": 1, "total_tokens": 2},
    }
    _respond(
        httpx_mock, f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream"
    )

    err = await _stream_error(_client())

    assert type(err.__cause__) is ValueError
    assert err.message == str(err.__cause__)
    assert err.context == {"model": MODEL, "type": "ValueError", "phase": "stream"}


# --- A charset label does not change the quote (round 2) ------------------------------
#
# httpx's ``Response.text`` decodes with the charset the Content-Type names, and
# for some labels that decode raises (``utf-16`` or ``utf-32`` on these UTF-8
# bodies; ``hex``, ``base64``, ``rot13`` are not text encodings). The adapter
# quotes ``content.decode("utf-8", "replace")`` instead, whatever the label.

_CHARSETS = ["latin-1", "utf-16", "utf-32", "hex", "base64", "rot13"]


@pytest.mark.parametrize("charset", _CHARSETS)
@pytest.mark.parametrize(
    ("kind", "decode_error"),
    [("non_utf8", "UnicodeDecodeError"), ("digit_limit", "ValueError")],
)
async def test_complete_undecodable_body_under_a_charset_label_is_quoted_as_utf8(
    httpx_mock: HTTPXMock, kind: str, decode_error: str, charset: str
) -> None:
    """An undecodable JSON-labelled body is UndecodableProviderBody quoting its UTF-8 reading, whatever charset is named (BR-023 round 2).

    The ``openai`` client decodes a JSON-labelled body from its bytes
    (``json.loads``), so the label's charset does not change the decode
    error. In round 1 the quote came from ``Response.text``: under
    ``utf-16``, ``utf-32``, ``hex``, ``base64`` and ``rot13`` that read raised
    (raw ``UnicodeDecodeError``, ``AssertionError`` or ``TypeError`` out of
    ``complete()``), and under ``latin-1`` it quoted the byte 0xE9 as "é",
    where every other row shows U+FFFD (evidence, round 2, probe A).
    """
    if kind == "non_utf8":
        body = _non_utf8_completion()
    else:
        if _DIGIT_LIMIT == 0:
            pytest.skip("the int-to-str digit limit is disabled (0) in this interpreter")
        body = _with_extra_key("1" * (_DIGIT_LIMIT + 1))
    text = body.decode("utf-8", "replace")
    _respond(httpx_mock, body, f"application/json; charset={charset}")

    err = await _complete_error(_client())

    assert type(err.__cause__).__name__ == decode_error
    assert err.message == _expected_message(_UNDECODABLE_PREFIX, text)
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "body_length": len(text),
        "decode_error": decode_error,
    }


@pytest.mark.parametrize("charset", ["utf-16", "utf-32"])
async def test_complete_text_body_the_client_cannot_decode_under_its_charset_is_undecodable(
    httpx_mock: HTTPXMock, charset: str
) -> None:
    """A text/plain body the openai client cannot decode with the charset it names is UndecodableProviderBody (BR-023 round 2).

    For a body not labelled JSON the ``openai`` client returns
    ``Response.text``, which decodes with the named charset; this UTF-8 text
    has no byte-order mark, and the incremental UTF-16/UTF-32 decoders
    ``httpx`` uses raise for a BOM-less stream even with
    ``errors="replace"``, the setting ``httpx`` 0.28.1 passes (a one-shot
    ``bytes.decode`` does not raise; BR-023 evidence, round 3). The
    exception is a
    ``UnicodeDecodeError`` on CPython 3.13.2 and 3.14.3 and a bare
    ``UnicodeError`` on 3.11.15 (evidence, round 2), so the assertions take
    ``decode_error`` from the cause. Before BR-023 it escaped ``complete()``
    raw, and in round 1 the arm's own ``.text`` read raised it again.
    """
    body = b"upstream service unavailable"
    _respond(httpx_mock, body, f"text/plain; charset={charset}")

    err = await _complete_error(_client())

    assert isinstance(err.__cause__, UnicodeError)
    assert err.message == _UNDECODABLE_PREFIX + body.decode()
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "body_length": len(body),
        "decode_error": type(err.__cause__).__name__,
    }


@pytest.mark.parametrize("charset", ["hex", "base64", "rot13"])
async def test_complete_text_body_under_a_non_text_charset_still_raises_raw(
    httpx_mock: HTTPXMock, charset: str
) -> None:
    """A text/plain body labelled with a codec that is not a text encoding still raises raw (BR-023 residual).

    Pins a documented gap, not a decision. The ``openai`` client's own
    ``Response.text`` read inside ``parse()`` raises ``AssertionError``
    (``hex``, ``base64``) or ``TypeError`` (``rot13``), neither a
    ``ValueError``, so the decode-step arm does not take it; the same body
    raised the same way before BR-023 (evidence, round 2: identical on the
    9ce1ec8 ``src``, ``openai`` 2.43.0 and 2.54.0). If this test changes,
    the module docstring and CHANGELOG must change with it.
    """
    _respond(httpx_mock, b"upstream service unavailable", f"text/plain; charset={charset}")

    with pytest.raises((AssertionError, TypeError)):
        await _client().complete(_request())


# --- The request step keeps no ValueError arm ------------------------------------------


@pytest.mark.parametrize("call", ["complete", "stream"])
@pytest.mark.parametrize(
    ("api_key", "content", "encoding"),
    [
        pytest.param("مفتاح", "hi", "ascii", id="non_ascii_api_key"),
        pytest.param("test-key", "a\ud800b", "utf-8", id="surrogate_in_user_message"),
    ],
)
async def test_request_encoding_errors_stay_raw_through_the_real_client(
    httpx_mock: HTTPXMock, call: str, api_key: str, content: str, encoding: str
) -> None:
    """A request that cannot be encoded raises UnicodeEncodeError raw, with nothing sent (BR-023 D8).

    The decode arm (``ValueError``, ``RecursionError``) wraps only the step
    that decodes the provider's body. A non-ASCII API key (httpx's ``ascii``
    header encode, BR-022 probe P7b) and a surrogate code point in a user
    message (the ``openai`` client's strict UTF-8 body encode, BR-022 P1 c1)
    raise in the request step, which keeps only the ``openai`` arms. The
    surrogate row pins a documented gap (host text), not a decision.
    Measured on ``openai`` 2.43.0 and 2.54.0 (BR-023 evidence P2 iii).
    """
    client = _client(api_key=api_key)

    with pytest.raises(UnicodeEncodeError) as exc:
        if call == "complete":
            await client.complete(_request(content))
        else:
            async for _ in client.stream(_request(content)):
                pass

    assert exc.value.encoding == encoding
    assert httpx_mock.get_requests() == []


async def test_complete_sends_the_raw_response_header_and_an_unchanged_body(
    httpx_mock: HTTPXMock,
) -> None:
    """complete() sends X-Stainless-Raw-Response: true; the body is _build_body's; stream() sends no such header (BR-023 D1).

    The header is how ``with_raw_response`` tells the ``openai`` client not
    to decode the body itself. It is the one request difference BR-023's
    probe P2 found (on 2.43.0 and 2.54.0).
    """
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_completion())
    sse = (
        "data: "
        + json.dumps(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )
    _respond(httpx_mock, sse.encode(), "text/event-stream")
    client = _client()

    await client.complete(_request())
    _ = [chunk async for chunk in client.stream(_request())]

    sent_complete, sent_stream = httpx_mock.get_requests()
    assert sent_complete.headers["x-stainless-raw-response"] == "true"
    assert json.loads(sent_complete.content) == client._build_body(
        _request(), model=MODEL, stream=False
    )
    assert "x-stainless-raw-response" not in sent_stream.headers


# --- complete(): fields of the wrong type (AC-8) --------------------------------------


def _tool_call_with_null_name() -> dict[str, Any]:
    payload = _completion(
        content=None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": None, "arguments": "{}"}}
        ],
    )
    payload["choices"][0]["finish_reason"] = "tool_calls"
    return payload


def _with_usage(prompt_tokens: object) -> dict[str, Any]:
    payload = _completion()
    payload["usage"]["prompt_tokens"] = prompt_tokens
    return payload


@pytest.mark.parametrize(
    ("body", "cause"),
    [
        pytest.param(json.dumps(_completion(content=5)), "ValidationError", id="content_int"),
        pytest.param(
            json.dumps(_completion(content=[_SENTINEL])), "ValidationError", id="content_list"
        ),
        pytest.param(
            json.dumps(_tool_call_with_null_name()), "ValidationError", id="tool_name_null"
        ),
        pytest.param(json.dumps(_with_usage(-1)), "ValidationError", id="usage_negative"),
        pytest.param(json.dumps(_with_usage(_SENTINEL)), "ValueError", id="usage_not_a_number"),
        pytest.param(
            json.dumps(_with_usage(7)).replace('"prompt_tokens": 7', '"prompt_tokens": 1e400'),
            "OverflowError",
            id="usage_infinite",
        ),
        pytest.param(
            json.dumps({**_completion(), "choices": {_SENTINEL: {"index": 0}}}),
            "KeyError",
            id="choices_object",
        ),
    ],
)
async def test_complete_wrong_typed_fields_are_malformed_response(
    httpx_mock: HTTPXMock, body: str, cause: str
) -> None:
    """A 200 completion with a field of the wrong type or value is MalformedResponse, fixed message (BR-023 AC-8).

    Before BR-023 each raised raw out of ``complete()`` and ``run()``: a
    pydantic ``ValidationError`` building ``ChatMessage``, ``ToolCall`` or
    ``Usage``, ``int()``'s ``ValueError``, ``OverflowError`` for
    ``int(inf)`` (evidence P7), or ``KeyError`` for ``choices`` sent as an
    object, which ``choices[0]`` indexes by key (round 2). The message names
    only the exception class: the ``content_list``, ``usage_not_a_number``
    and ``choices_object`` rows put a sentinel in the wrong-typed value, and
    ``str()`` of each error would quote it.
    """
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert type(err.__cause__).__name__ == cause
    assert err.message == _WRONG_TYPE_MESSAGE.format(cause)
    assert _SENTINEL not in err.message
    assert err.context == {"model": MODEL, "type": "MalformedResponse"}


@pytest.mark.parametrize(
    ("body", "cause"),
    [
        pytest.param(
            json.dumps({**_completion(), "choices": "abc"}), "AttributeError", id="choices_string"
        ),
        pytest.param(json.dumps({**_completion(), "choices": 5}), "TypeError", id="choices_number"),
        pytest.param(
            json.dumps(_completion()).replace('"finish_reason": "stop"', '"finish_reason": [1]'),
            "TypeError",
            id="finish_reason_list",
        ),
        pytest.param(json.dumps(_with_usage([1])), "TypeError", id="usage_count_list"),
        pytest.param(json.dumps(_completion(tool_calls=5)), "TypeError", id="tool_calls_number"),
    ],
)
async def test_complete_attribute_and_type_errors_keep_the_earlier_malformed_message(
    httpx_mock: HTTPXMock, body: str, cause: str
) -> None:
    """Fields failing with AttributeError, IndexError or TypeError keep 'Malformed provider response: {e}' (BR-023 control).

    BR-023's fixed message is for ``ValueError``, ``OverflowError`` and
    ``LookupError`` other than ``IndexError``; the older arm, which quotes
    ``str(e)``, runs first and is unchanged. These shapes were already
    ``MalformedResponse`` before BR-023 (evidence, round 2, probe C).
    """
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert type(err.__cause__).__name__ == cause
    assert err.message == f"Malformed provider response: {err.__cause__}"
    assert err.context == {"model": MODEL, "type": "MalformedResponse"}
