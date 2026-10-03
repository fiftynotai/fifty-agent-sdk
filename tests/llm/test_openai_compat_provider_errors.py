"""Provider bodies and context-window errors in ``OpenAICompatibleClient`` (BR-021).

An OpenAI-compatible endpoint can answer an error with HTTP 200 and a body that
is not a chat completion. These tests drive the real ``openai`` client through
``pytest-httpx`` (``max_retries=0``, no network) and pin how each such body
surfaces from ``complete()`` and ``stream()``: as an ``LLMError`` whose message
quotes the body (cut to 500 characters) and whose ``context["type"]`` is
``NonJsonProviderBody`` or ``NonStreamProviderBody``, re-typed
``ContextLengthExceeded`` when the provider text names a context-window
overflow.

Release dependence: how a body reaches the adapter (a ``str``, a raw
``json.JSONDecodeError``, a stream with no events) is the installed ``openai``
release's behaviour. It was measured on 2.43.0, the dev venv's release; CI
installs whatever release the declared range resolves to. The BR-021 suite
also passed with 2.54.0 (CPython 3.11.15 and 3.13.2), and BR-021's evidence
records probes of the declared floor, 1.30.0. The adapter's checks do not
depend on the release. Tests that assert the route (``__cause__``) would turn
red on a release that surfaced the body another way; the others assert only
the resulting error.

Since BR-023 a non-streamed 200 JSON error envelope (an object without
``choices`` and with a truthy ``error`` member) is ``ErrorEnvelopeProviderBody``
carrying its text, classified like its streamed twin; the bodies the client
cannot decode are in ``test_openai_compat_body_decoding.py``. Two bodies the
client still does not catch are pinned as documented residuals, not endorsed:
a no-choices JSON object without an ``error`` member stays
``MalformedResponse`` with its text lost, and a body without SSE fields
labelled ``text/event-stream`` yields nothing. Not pinned: other providers'
phrasings of an overflow (they stay unclassified).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.openai_compat import (
    _CONTEXT_LENGTH_MARKERS,
    OpenAICompatibleClient,
    _is_context_length_exceeded,
)
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest

BASE_URL = "https://example.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"
MODEL = "gpt-4o"

# The fixed message prefixes, written out here so a change to them is a
# deliberate, visible change to this contract.
_BODY_PREFIX = "Provider response body could not be read as a JSON object: "
_EVENT_PREFIX = "Provider stream event is not a JSON object: "
_STREAM_PREFIX = (
    "Provider stream produced no chunk and its Content-Type is not text/event-stream. Body: "
)
_ENVELOPE_PREFIX = "Provider returned an error object instead of a completion: "
_UNDECODABLE_STREAM_PREFIX = "Provider stream could not be decoded: "
_TRUNCATED = "…[truncated]"

# The three markers, written out independently of the module constant so that
# dropping one from the source turns a test red instead of moving the test.
_MARKERS = ("max_prompt_length", "context_length_exceeded", "maximum context length")

_PROMPT_LENGTH = "Prompt length 145048 exceeds max_prompt_length 131072"
_MODEL_MAX_CONTEXT = (
    "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens."
)


# --- Helpers -------------------------------------------------------------------------


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(api_key="test-key", base_url=BASE_URL, timeout=5.0, max_retries=0)


def _request() -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="hi")], model=MODEL)


def _chunk(content: str, finish_reason: str | None) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    }


_VALID_SSE = (
    f"data: {json.dumps(_chunk('Hello', None))}\n\n"
    f"data: {json.dumps(_chunk(' world', 'stop'))}\n\n"
    "data: [DONE]\n\n"
).encode()


def _respond(
    httpx_mock: HTTPXMock,
    body: bytes,
    content_type: str | None,
    *,
    status_code: int = 200,
) -> None:
    headers = {"content-type": content_type} if content_type is not None else None
    httpx_mock.add_response(
        method="POST", url=ENDPOINT, status_code=status_code, content=body, headers=headers
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


# --- complete(): a 200 body the adapter receives as text ------------------------------


async def test_complete_200_text_body_raises_llm_error_carrying_the_body(
    httpx_mock: HTTPXMock,
) -> None:
    """A 200 text/plain body is an LLMError quoting the stripped body, typed NonJsonProviderBody (BR-021).

    ``body_length`` counts the decoded body's characters before stripping,
    trailing newline included.
    Before BR-021 this was ``MalformedResponse`` with the body lost.
    """
    body = "upstream service unavailable, try later\n"
    _respond(httpx_mock, body.encode(), "text/plain")

    err = await _complete_error(_client())

    assert err.message == _BODY_PREFIX + body.strip()
    assert err.context == {"model": MODEL, "type": "NonJsonProviderBody", "body_length": len(body)}


async def test_complete_200_text_body_without_content_type_raises_llm_error_carrying_the_body(
    httpx_mock: HTTPXMock,
) -> None:
    """With no Content-Type header the 200 text body takes the same path as text/plain (BR-021)."""
    body = "upstream service unavailable"
    _respond(httpx_mock, body.encode(), None)

    err = await _complete_error(_client())

    assert err.message == _BODY_PREFIX + body
    assert err.context == {"model": MODEL, "type": "NonJsonProviderBody", "body_length": len(body)}


async def test_complete_200_json_labelled_text_body_raises_llm_error_not_json_decode_error(
    httpx_mock: HTTPXMock,
) -> None:
    """A 200 labelled application/json with a plain-text body is an LLMError, not a raw JSONDecodeError (BR-021).

    On openai 2.43.0 the client's own decode raises ``json.JSONDecodeError``
    out of ``create()``, which used to escape ``complete()`` and ``run()``
    raw. The cause assertion is release-dependent: a release that returned
    the text instead would fail it here rather than pass silently.
    """
    body = "upstream service unavailable"
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert isinstance(err.__cause__, json.JSONDecodeError)
    assert err.message == _BODY_PREFIX + body
    assert err.context == {"model": MODEL, "type": "NonJsonProviderBody", "body_length": len(body)}


async def test_complete_200_json_string_body_is_a_provider_body(httpx_mock: HTTPXMock) -> None:
    """A 200 body that is a bare JSON string reaches the adapter as a str and is NonJsonProviderBody (BR-021).

    ``body_length`` is the decoded string's length, not the body's: the
    quotes are not counted.
    """
    body = json.dumps("upstream busy")
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert err.message == _BODY_PREFIX + "upstream busy"
    assert err.context["type"] == "NonJsonProviderBody"
    assert err.context["body_length"] == len("upstream busy") == len(body) - 2
    assert err.__cause__ is None


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
async def test_complete_200_json_error_envelope_carries_its_text_and_is_classified(
    httpx_mock: HTTPXMock, content_type: str
) -> None:
    """A non-streamed 200 JSON error envelope quotes its body and is classified (BR-023 AC-3).

    Before BR-023 this pinned a BR-021 residual: ``MalformedResponse``
    ("Provider returned no choices."), unclassified, the text lost, under
    both labels. The streamed twin is
    ``test_stream_200_json_error_body_is_classified``.
    """
    body = json.dumps({"error": {"message": _MODEL_MAX_CONTEXT, "code": "context_length_exceeded"}})
    _respond(httpx_mock, body.encode(), content_type)

    err = await _complete_error(_client())

    assert err.__cause__ is None
    assert err.message == _ENVELOPE_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "body_length": len(body),
        "classified_from": "ErrorEnvelopeProviderBody",
    }


async def test_complete_200_error_envelope_without_a_marker_is_unclassified(
    httpx_mock: HTTPXMock,
) -> None:
    """An error envelope that names no overflow is ErrorEnvelopeProviderBody, with no classified_from (BR-023)."""
    body = json.dumps(
        {"error": {"message": "invalid temperature", "type": "invalid_request_error"}}
    )
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _complete_error(_client())

    assert err.message == _ENVELOPE_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "ErrorEnvelopeProviderBody",
        "body_length": len(body),
    }


@pytest.mark.parametrize(
    "envelope",
    [
        pytest.param({"error": {"message": _MODEL_MAX_CONTEXT}}, id="marker_in_message"),
        pytest.param(
            {"error": {"message": "request rejected", "code": "context_length_exceeded"}},
            id="code_only",
        ),
    ],
)
async def test_complete_and_stream_treat_the_same_error_envelope_alike(
    httpx_mock: HTTPXMock, envelope: dict[str, Any]
) -> None:
    """complete() and stream() quote the same text, the same length and the same class for one envelope (BR-023 AC-3).

    Both classify on the whole body text, so in the ``code_only`` row, whose
    message names no overflow, the ``code`` string in the text classifies it
    on both.
    """
    body = json.dumps(envelope)
    _respond(httpx_mock, body.encode(), "application/json")
    _respond(httpx_mock, body.encode(), "application/json")

    complete_err = await _complete_error(_client())
    stream_err = await _stream_error(_client())

    assert complete_err.context["type"] == stream_err.context["type"] == "ContextLengthExceeded"
    assert complete_err.message.removeprefix(_ENVELOPE_PREFIX) == body
    assert stream_err.message.removeprefix(_STREAM_PREFIX) == body
    assert complete_err.context["body_length"] == stream_err.context["body_length"] == len(body)
    assert complete_err.context["classified_from"] == "ErrorEnvelopeProviderBody"
    assert stream_err.context["classified_from"] == "NonStreamProviderBody"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"id": "cmpl-1", "object": "chat.completion", "choices": []}, id="empty_choices"
        ),
        pytest.param({"id": "x"}, id="no_choices_no_error"),
        pytest.param({"error": None}, id="error_null"),
        pytest.param({"error": ""}, id="error_empty_string"),
        pytest.param({"error": {}}, id="error_empty_object"),
        pytest.param({"detail": _MODEL_MAX_CONTEXT}, id="detail_member"),
    ],
)
async def test_complete_no_choices_without_an_error_member_stays_malformed_response(
    httpx_mock: HTTPXMock, payload: dict[str, Any]
) -> None:
    """A no-choices object whose error member is absent or falsy stays MalformedResponse, text lost (BR-023 control).

    The envelope rule needs a truthy ``error`` member, the ``openai``
    client's own rule for a streamed error event. The ``detail_member`` row
    pins a documented gap, not a decision: such a body (a FastAPI-style
    error, for example) is not classified even when it names an overflow.
    If it changes, the module docstring, README and CHANGELOG must change
    with it.
    """
    _respond(httpx_mock, json.dumps(payload).encode(), "application/json")

    err = await _complete_error(_client())

    assert err.message == "Provider returned no choices."
    assert err.context == {"model": MODEL, "type": "MalformedResponse"}


@pytest.mark.parametrize("charset", ["utf-16", "utf-16-le", "utf-32", "hex", "base64", "rot13"])
async def test_complete_error_envelope_under_a_charset_label_is_quoted_as_utf8(
    httpx_mock: HTTPXMock, charset: str
) -> None:
    """A UTF-8 error envelope is quoted and classified whatever charset its Content-Type names (BR-023 round 2).

    The ``openai`` client parses a JSON-labelled body from its bytes, so the
    envelope is found under every label; the adapter quotes those bytes read
    as UTF-8. In round 1 it quoted ``Response.text``, which decodes with the
    named charset: ``hex`` and ``base64`` raised a raw ``AssertionError``
    out of ``complete()`` (``MalformedResponse`` before BR-023), ``utf-16``
    and ``utf-32`` raised in that read and were mislabelled
    ``MalformedResponse``, ``rot13`` was ``MalformedResponse`` from a
    ``TypeError``, and ``utf-16-le`` was quoted unreadably and not
    classified (evidence, round 2, probe A).
    """
    body = json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}})
    _respond(httpx_mock, body.encode(), f"application/json; charset={charset}")

    err = await _complete_error(_client())

    assert err.message == _ENVELOPE_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "body_length": len(body),
        "classified_from": "ErrorEnvelopeProviderBody",
    }


async def test_complete_error_envelope_encoded_in_utf16_is_quoted_as_its_utf8_reading(
    httpx_mock: HTTPXMock,
) -> None:
    """An envelope actually encoded in UTF-16 is quoted as its UTF-8 reading and not classified (BR-023 residual).

    Pins a documented gap, not a decision. ``json.loads`` detects UTF-16 from
    the bytes, so the envelope is found, but the quote reads the bytes as
    UTF-8: each ASCII character is followed by a NUL and the byte-order mark
    becomes two U+FFFD, so no marker matches. Round 1 quoted
    ``Response.text`` and classified this body when the label named
    ``utf-16``; that read was not total (see the test above). If this test
    changes, the module docstring and CHANGELOG must change with it.
    """
    body = json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}}).encode("utf-16")
    _respond(httpx_mock, body, "application/json; charset=utf-16")

    err = await _complete_error(_client())

    text = body.decode("utf-8", "replace")
    assert "\x00" in text
    assert err.message == _ENVELOPE_PREFIX + text.strip()
    assert err.context == {
        "model": MODEL,
        "type": "ErrorEnvelopeProviderBody",
        "body_length": len(text),
    }


async def test_error_member_beside_valid_choices_is_a_normal_completion(
    httpx_mock: HTTPXMock,
) -> None:
    """A completion with choices AND an error member maps as a completion; choices are checked first (BR-023 control)."""
    payload = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "error": {"message": _MODEL_MAX_CONTEXT},
    }
    _respond(httpx_mock, json.dumps(payload).encode(), "application/json")

    response = await _client().complete(_request())

    assert response.message.content == "hello"
    assert response.finish_reason == "stop"


async def test_provider_body_excerpt_is_bounded_and_length_recorded(
    httpx_mock: HTTPXMock,
) -> None:
    """A 2,000-character body is quoted to 500 characters plus the marker; body_length is 2000 (BR-021)."""
    body = "a" * 500 + "B" + "c" * 1499
    assert len(body) == 2000
    _respond(httpx_mock, body.encode(), "text/plain")

    err = await _complete_error(_client())

    assert err.message == _BODY_PREFIX + "a" * 500 + _TRUNCATED
    assert "B" not in err.message
    assert err.context["body_length"] == 2000


@pytest.mark.parametrize(
    ("length", "marked"), [(500, False), (501, True)], ids=["at_bound", "one_over"]
)
async def test_provider_body_marker_appears_only_past_the_bound(
    httpx_mock: HTTPXMock, length: int, marked: bool
) -> None:
    """A body of exactly 500 characters is quoted whole; 501 is cut and marked (BR-021)."""
    body = "z" * length
    _respond(httpx_mock, body.encode(), "text/plain")

    err = await _complete_error(_client())

    expected = _BODY_PREFIX + "z" * 500 + (_TRUNCATED if marked else "")
    assert err.message == expected


# --- Context-window classification ----------------------------------------------------


async def test_context_length_marker_past_the_excerpt_still_classifies(
    httpx_mock: HTTPXMock,
) -> None:
    """A marker at character 900, outside the quoted excerpt, still classifies the error (BR-021)."""
    body = "x" * 900 + " max_prompt_length exceeded"
    _respond(httpx_mock, body.encode(), "text/plain")

    err = await _complete_error(_client())

    assert "max_prompt_length" not in err.message
    assert err.context["type"] == "ContextLengthExceeded"
    assert err.context["classified_from"] == "NonJsonProviderBody"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_PROMPT_LENGTH, id="max_prompt_length"),
        pytest.param("error: context_length_exceeded", id="context_length_exceeded"),
        pytest.param(_MODEL_MAX_CONTEXT, id="maximum_context_length"),
        pytest.param("MAXIMUM CONTEXT LENGTH reached", id="upper_case"),
        pytest.param("Max_Prompt_Length exceeded", id="mixed_case"),
    ],
)
async def test_each_over_length_message_is_classified_context_length_exceeded(
    httpx_mock: HTTPXMock, body: str
) -> None:
    """Each over-length message, holding only its own marker, is ContextLengthExceeded (BR-021 AC-2).

    The body is still quoted in the message, and ``classified_from`` keeps
    the type the error would otherwise have had.
    """
    assert sum(marker in body.casefold() for marker in _MARKERS) == 1
    _respond(httpx_mock, body.encode(), "text/plain")

    err = await _complete_error(_client())

    assert err.message == _BODY_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "body_length": len(body),
        "classified_from": "NonJsonProviderBody",
    }


@pytest.mark.parametrize(
    ("status_code", "body", "content_type", "classified_from"),
    [
        pytest.param(
            400,
            json.dumps(
                {
                    "error": {
                        "message": _MODEL_MAX_CONTEXT,
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                }
            ),
            "application/json",
            "BadRequestError",
            id="400_openai_shaped",
        ),
        pytest.param(400, _PROMPT_LENGTH, "text/plain", "BadRequestError", id="400_plain_text"),
        pytest.param(
            500,
            json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}}),
            "application/json",
            "InternalServerError",
            id="500_json",
        ),
    ],
)
async def test_non_2xx_over_length_errors_are_classified(
    httpx_mock: HTTPXMock,
    status_code: int,
    body: str,
    content_type: str,
    classified_from: str,
) -> None:
    """A 4xx/5xx over-length error is ContextLengthExceeded and keeps its message (BR-021 AC-2)."""
    _respond(httpx_mock, body.encode(), content_type, status_code=status_code)

    err = await _complete_error(_client())

    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "classified_from": classified_from,
    }
    # The message is still the openai exception's own text, unchanged by BR-021.
    assert err.message == str(err.__cause__)


async def test_stream_open_400_over_length_is_classified(httpx_mock: HTTPXMock) -> None:
    """A 400 over-length error raised when stream() opens the request is classified (BR-021)."""
    _respond(httpx_mock, _PROMPT_LENGTH.encode(), "text/plain", status_code=400)

    err = await _stream_error(_client())

    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "classified_from": "BadRequestError",
    }


async def test_mid_stream_error_event_with_context_length_code_is_classified(
    httpx_mock: HTTPXMock,
) -> None:
    """A mid-stream error event whose code is context_length_exceeded is classified (BR-021).

    Its message names no marker, so only the ``code`` rule can classify it.
    """
    event = {"error": {"message": "request rejected", "code": "context_length_exceeded"}}
    _respond(httpx_mock, f"data: {json.dumps(event)}\n\n".encode(), "text/event-stream")

    err = await _stream_error(_client())

    assert err.message == "request rejected"
    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "phase": "stream",
        "classified_from": "APIError",
    }


@pytest.mark.parametrize("call", ["complete", "stream"])
@pytest.mark.parametrize(
    ("response", "expected_type"),
    [
        pytest.param(
            (400, json.dumps({"error": {"message": "invalid temperature"}})),
            "BadRequestError",
            id="400_unrelated",
        ),
        pytest.param(
            (429, json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}})),
            "RateLimitError",
            id="429_with_marker",
        ),
        pytest.param(
            httpx.ConnectError("maximum context length"), "APIConnectionError", id="connect"
        ),
        pytest.param(
            httpx.TimeoutException("maximum context length"), "APITimeoutError", id="timeout"
        ),
    ],
)
async def test_unrelated_errors_keep_their_type(
    httpx_mock: HTTPXMock,
    call: str,
    response: tuple[int, str] | Exception,
    expected_type: str,
) -> None:
    """Unrelated errors, rate limits and transport failures keep exactly their pre-BR-021 context (BR-021).

    A 429 is never classified, even when its text holds a marker: a rate
    limit is not a context-window error. Neither is a connection error or a
    timeout.
    """
    if isinstance(response, Exception):
        httpx_mock.add_exception(response)
    else:
        status_code, body = response
        _respond(httpx_mock, body.encode(), "application/json", status_code=status_code)
    client = _client()

    err = await (_complete_error(client) if call == "complete" else _stream_error(client))

    assert err.context == {"model": MODEL, "type": expected_type}


# --- stream(): a 200 body that is not an event stream ---------------------------------


@pytest.mark.parametrize(
    ("content_type", "recorded"),
    [
        pytest.param("text/plain", "text/plain", id="text_plain"),
        pytest.param("Text/HTML; charset=utf-8", "text/html", id="params_dropped"),
        pytest.param(None, "", id="no_header"),
    ],
)
async def test_stream_200_text_body_raises_llm_error_carrying_the_body(
    httpx_mock: HTTPXMock, content_type: str | None, recorded: str
) -> None:
    """A streaming request answered with a 200 non-SSE body is NonStreamProviderBody, quoting it (BR-021).

    ``content_type`` is the media type only, casefolded, and ``""`` with no
    header. Before BR-021 the stream looked empty and the text was lost.
    """
    body = "upstream service unavailable"
    _respond(httpx_mock, body.encode(), content_type)

    err = await _stream_error(_client())

    assert err.message == _STREAM_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "NonStreamProviderBody",
        "body_length": len(body),
        "phase": "stream",
        "content_type": recorded,
    }


async def test_stream_200_json_error_body_is_classified(httpx_mock: HTTPXMock) -> None:
    """A stream request answered with a 200 JSON error envelope is classified from its body (BR-021)."""
    body = json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}})
    _respond(httpx_mock, body.encode(), "application/json")

    err = await _stream_error(_client())

    assert err.message == _STREAM_PREFIX + body
    assert err.context == {
        "model": MODEL,
        "type": "ContextLengthExceeded",
        "body_length": len(body),
        "phase": "stream",
        "content_type": "application/json",
        "classified_from": "NonStreamProviderBody",
    }


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        pytest.param(_PROMPT_LENGTH, "ContextLengthExceeded", id="over_length"),
        pytest.param("upstream busy", "NonJsonProviderBody", id="plain"),
    ],
)
async def test_stream_event_data_that_is_not_json_carries_the_text(
    httpx_mock: HTTPXMock, data: str, expected_type: str
) -> None:
    """SSE data that is not JSON is NonJsonProviderBody quoting the data, not JSONDecodeError (BR-021).

    The decode failure is the openai SSE decoder's (measured on 2.43.0);
    the cause assertion would fail on a release that reported it otherwise.
    """
    _respond(httpx_mock, f"data: {data}\n\n".encode(), "text/event-stream")

    err = await _stream_error(_client())

    assert isinstance(err.__cause__, json.JSONDecodeError)
    assert err.message == _EVENT_PREFIX + data
    assert err.context["type"] == expected_type
    assert err.context["phase"] == "stream"
    assert err.context["body_length"] == len(data)
    if expected_type == "ContextLengthExceeded":
        assert err.context["classified_from"] == "NonJsonProviderBody"
    else:
        assert "classified_from" not in err.context


async def test_stream_event_that_is_a_bare_json_string_carries_the_text(
    httpx_mock: HTTPXMock,
) -> None:
    """An SSE event whose data is a JSON string reaches _map_chunk as a str: NonJsonProviderBody, not MalformedChunk (BR-021)."""
    _respond(httpx_mock, b'data: "upstream busy"\n\n', "text/event-stream")

    err = await _stream_error(_client())

    assert err.__cause__ is None
    assert err.message == _EVENT_PREFIX + "upstream busy"
    assert err.context == {
        "model": MODEL,
        "type": "NonJsonProviderBody",
        "body_length": len("upstream busy"),
        "phase": "stream",
    }


@pytest.mark.parametrize(
    ("content_type", "expected_reads"),
    [
        pytest.param("text/event-stream", 0, id="event_stream"),
        pytest.param("text/event-stream; charset=utf-8", 0, id="event_stream_params"),
        pytest.param("Text/Event-Stream", 0, id="event_stream_case"),
        pytest.param("text/plain", 1, id="text_plain"),
    ],
)
async def test_only_non_event_stream_responses_are_buffered(
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    expected_reads: int,
) -> None:
    """Only a non-text/event-stream response is read ahead; every one still yields its chunks (BR-021 D8).

    The spy records ``httpx.Response.aread`` calls. The three event-stream
    rows show that, on this path, nothing else in the stack (openai 2.43.0,
    httpx, pytest-httpx) calls it on a 200 stream, so the ``text/plain``
    row's one call is the adapter's.
    """
    reads: list[httpx.Response] = []
    original = httpx.Response.aread

    async def spy(self: httpx.Response) -> bytes:
        reads.append(self)
        return await original(self)

    monkeypatch.setattr(httpx.Response, "aread", spy)
    _respond(httpx_mock, _VALID_SSE, content_type)

    chunks = [chunk async for chunk in _client().stream(_request())]

    assert len(reads) == expected_reads
    assert [c.message.content for c in chunks] == ["Hello", " world"]


async def test_mislabelled_event_stream_still_yields_its_chunks(httpx_mock: HTTPXMock) -> None:
    """Valid SSE labelled application/json is read ahead and still yields every chunk (BR-021 D8).

    The no-header twin is the unchanged
    ``test_build_body_reasoning_model_stream_uses_max_completion_tokens``
    in ``tests/llm/test_openai_compat.py``.
    """
    _respond(httpx_mock, _VALID_SSE, "application/json")

    chunks = [chunk async for chunk in _client().stream(_request())]

    assert "".join(c.message.content for c in chunks) == "Hello world"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"data: [DONE]\n\n", id="done_only"),
        pytest.param(_PROMPT_LENGTH.encode(), id="plain_text_labelled_sse"),
        pytest.param(
            json.dumps({"error": {"message": _MODEL_MAX_CONTEXT}}).encode(),
            id="json_envelope_labelled_sse",
        ),
    ],
)
async def test_empty_event_stream_is_not_a_provider_error(
    httpx_mock: HTTPXMock, body: bytes
) -> None:
    """A text/event-stream response that yields no chunk yields nothing and raises nothing (BR-021 D8).

    The ``done_only`` row pins a decision: an empty event stream keeps the
    loop's parser retry, as it may be transient. The
    ``plain_text_labelled_sse`` and ``json_envelope_labelled_sse`` rows pin a
    documented gap, not a decision: openai 2.43.0's SSE decoder discards a
    body without SSE fields, so the adapter never sees its text. If those
    rows change, the module docstring, README and CHANGELOG must change with
    them.
    """
    _respond(httpx_mock, body, "text/event-stream")

    chunks = [chunk async for chunk in _client().stream(_request())]

    assert chunks == []


@pytest.mark.parametrize(
    ("content_type", "recorded"),
    [
        pytest.param("application/json", "application/json", id="json"),
        pytest.param(None, "", id="no_header"),
    ],
)
async def test_mislabelled_stream_with_no_chunk_names_what_was_detected(
    httpx_mock: HTTPXMock, content_type: str | None, recorded: str
) -> None:
    """An SSE-framed body with only data: [DONE], not labelled text/event-stream, raises NonStreamProviderBody (BR-021 D8).

    The body IS an event stream; it carries no chunk. So the message says
    what was detected (no chunk, and the Content-Type is not
    ``text/event-stream``) rather than that the body is not an event
    stream. Before BR-021 this yielded nothing.
    """
    _respond(httpx_mock, b"data: [DONE]\n\n", content_type)

    err = await _stream_error(_client())

    assert err.message == (
        "Provider stream produced no chunk and its Content-Type is not text/event-stream. "
        "Body: data: [DONE]"
    )
    assert err.context == {
        "model": MODEL,
        "type": "NonStreamProviderBody",
        "body_length": len("data: [DONE]\n\n"),
        "phase": "stream",
        "content_type": recorded,
    }


async def test_stream_content_type_is_cut_to_100_characters(httpx_mock: HTTPXMock) -> None:
    """context["content_type"] keeps the first 100 characters of a longer media type (BR-021)."""
    media_type = "application/" + "x" * 288
    assert len(media_type) == 300
    _respond(httpx_mock, b"not an event stream", media_type)

    err = await _stream_error(_client())

    assert err.context["content_type"] == media_type[:100]


class _ReadFailsAfter(httpx.AsyncByteStream):
    """A response body that yields ``first`` and then raises ``error``."""

    def __init__(self, first: bytes, error: Exception) -> None:
        self._first = first
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        raise self._error


async def test_read_ahead_failure_is_wrapped_with_its_httpx_type(httpx_mock: HTTPXMock) -> None:
    """A read failure while a non-event-stream body is read ahead is LLMError(type=ReadError, phase=stream) (BR-021)."""
    error = httpx.ReadError("connection reset mid-body")
    httpx_mock.add_callback(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ReadFailsAfter(b"partial provider text ", error),
        ),
        method="POST",
        url=ENDPOINT,
    )

    err = await _stream_error(_client())

    assert err.__cause__ is error
    assert err.message == "connection reset mid-body"
    assert err.context == {"model": MODEL, "type": "ReadError", "phase": "stream"}


async def test_mid_stream_read_failure_is_never_classified(httpx_mock: HTTPXMock) -> None:
    """A mid-stream httpx read timeout naming a marker keeps its own type; it is not ContextLengthExceeded (BR-021).

    The generic mid-stream arm wraps transport failures without
    classification; only the ``APIError`` arms classify.
    """
    first_event = f"data: {json.dumps(_chunk('Hello', None))}\n\n".encode()
    error = httpx.ReadTimeout("read timed out: maximum context length")
    httpx_mock.add_callback(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ReadFailsAfter(first_event, error),
        ),
        method="POST",
        url=ENDPOINT,
    )
    chunks: list[str] = []

    with pytest.raises(LLMError) as exc:
        async for chunk in _client().stream(_request()):
            chunks.append(chunk.message.content)

    assert chunks == ["Hello"]
    assert exc.value.__cause__ is error
    assert exc.value.context == {"model": MODEL, "type": "ReadTimeout", "phase": "stream"}


# --- The new except arms stay narrow --------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("not a decode error"), id="value_error"),
        pytest.param(
            UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed"),
            id="unicode_encode_error",
        ),
    ],
)
async def test_json_decode_arm_does_not_wrap_other_value_errors(
    monkeypatch: pytest.MonkeyPatch, error: ValueError
) -> None:
    """complete()'s request step leaves every ValueError raw (BR-021 D9; BR-023 D8).

    A ``UnicodeEncodeError`` (a ``ValueError``) from encoding the request
    would be reported as a provider body by a ``ValueError`` arm. BR-022
    escapes surrogate code points in the tool-result messages the loop
    builds and adds no arm here. Since BR-023 ``complete()`` calls the
    patched ``create`` through ``with_raw_response`` (BR-023 evidence P2 iv),
    and its decode arms wrap only the separate decode step, so the error
    raised here, before any body exists, still escapes raw. The real-client
    pins are in ``test_openai_compat_body_decoding.py``.
    """
    client = _client()

    async def raising_create(**_: Any) -> Any:  # noqa: ANN401 - mirrors the SDK signature
        raise error

    monkeypatch.setattr(client._client.chat.completions, "create", raising_create)

    with pytest.raises(ValueError) as exc:
        await client.complete(_request())

    assert exc.value is error


async def test_mid_stream_iterator_value_error_is_undecodable_not_a_provider_text_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ValueError from the stream iterator is UndecodableProviderBody quoting the error, not a body (BR-023 D3).

    Changed on purpose by BR-023. As BR-021's
    ``test_mid_stream_json_decode_arm_does_not_claim_other_value_errors``
    this pinned the generic wrap (``type="ValueError"``, message
    ``str(e)``). BR-023 types every ``ValueError`` the ``openai`` iterator
    step raises as a decode failure: the ones its probe P1 measured from the
    iterator (``openai`` 2.43.0 and 2.54.0) were. What BR-021 D9 protected
    still holds: the message quotes the exception, not provider text, the
    type is not ``NonJsonProviderBody``, and there is no ``body_length``.
    The fake stream has no ``response`` attribute, so the adapter iterates
    it without reading ahead (the ``getattr`` fallback).
    """
    error = ValueError("not a decode error")

    class _RaisingStream:
        def __aiter__(self) -> AsyncIterator[Any]:
            return self

        async def __anext__(self) -> Any:  # noqa: ANN401 - opaque chunk type
            raise error

    client = _client()

    async def fake_create(**_: Any) -> _RaisingStream:
        return _RaisingStream()

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    err = await _stream_error(client)

    assert err.__cause__ is error
    assert err.message == _UNDECODABLE_STREAM_PREFIX + "not a decode error"
    assert err.context == {
        "model": MODEL,
        "type": "UndecodableProviderBody",
        "decode_error": "ValueError",
        "phase": "stream",
    }


# --- The predicate --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "code", "expected"),
    [
        pytest.param(_PROMPT_LENGTH, None, True, id="max_prompt_length"),
        pytest.param("code=context_length_exceeded", None, True, id="context_length_exceeded"),
        pytest.param(_MODEL_MAX_CONTEXT, None, True, id="maximum_context_length"),
        pytest.param("MAX_PROMPT_LENGTH", None, True, id="upper_max_prompt_length"),
        pytest.param("Context_Length_Exceeded", None, True, id="mixed_context_length_exceeded"),
        pytest.param("the Maximum Context LENGTH", None, True, id="mixed_maximum_context_length"),
        pytest.param("prompt is too long", None, False, id="other_phrasing"),
        pytest.param("", "context_length_exceeded", True, id="code_only"),
        pytest.param("", "CONTEXT_LENGTH_EXCEEDED", True, id="code_case"),
        pytest.param("", "rate_limit_exceeded", False, id="other_code"),
        pytest.param("", 42, False, id="non_str_code"),
        pytest.param("", None, False, id="empty"),
    ],
)
def test_is_context_length_exceeded_predicate(text: str, code: object, expected: bool) -> None:
    """The BR-021 predicate: any of three markers, case-insensitive, or the context_length_exceeded code."""
    assert _is_context_length_exceeded(text, code) is expected


def test_context_length_marker_list_is_the_documented_three() -> None:
    """The shipped marker tuple is exactly the three markers BR-021 documents.

    Adding, dropping or reordering a marker turns this red, so a change to
    the closed list needs a deliberate edit of this test; that edit is the
    reminder that the CHANGELOG must say so (nothing here checks the
    CHANGELOG).
    """
    assert _CONTEXT_LENGTH_MARKERS == _MARKERS
