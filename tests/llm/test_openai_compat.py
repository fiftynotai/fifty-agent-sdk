"""Tests for fifty_agent_sdk.llm.openai_compat.OpenAICompatibleClient.

Uses ``pytest-httpx`` to intercept the HTTP calls the openai Python SDK
makes under the hood. No real network is required.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH
from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.openai_compat import OpenAICompatibleClient
from fifty_agent_sdk.llm.protocol import LLMClient
from fifty_agent_sdk.llm.types import ChatMessage, ChatRequest, ToolCall, ToolChoiceFunction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://example.com/v1"
ENDPOINT = f"{BASE_URL}/chat/completions"


def _canonical_response(
    *,
    content: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    total_tokens: int = 8,
    model: str = "gpt-4o",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(
    *,
    delta_role: str | None = None,
    delta_content: str | None = None,
    finish_reason: str | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if delta_role is not None:
        delta["role"] = delta_role
    if delta_content is not None:
        delta["content"] = delta_content
    return {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _make_client(
    *,
    base_url: str = BASE_URL,
    model: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url=base_url,
        model=model,
        timeout=5.0,
        max_retries=0,
        http_client=http_client,
    )


def _basic_request(*, model: str = "gpt-4o", **overrides: Any) -> ChatRequest:
    fields: dict[str, Any] = {
        "messages": [ChatMessage(role="user", content="hi")],
        "model": model,
    }
    fields.update(overrides)
    return ChatRequest(**fields)


# ---------------------------------------------------------------------------
# Happy path: complete()
# ---------------------------------------------------------------------------


async def test_complete_happy_path_maps_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.message.role == "assistant"
    assert resp.message.content == "hello"
    assert resp.finish_reason == "stop"
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.completion_tokens == 3
    assert resp.usage.total_tokens == 8


async def test_complete_targets_configured_base_url(httpx_mock: HTTPXMock) -> None:
    base = "https://gdc.example.com/v1"
    httpx_mock.add_response(
        method="POST",
        url=f"{base}/chat/completions",
        json=_canonical_response(),
    )
    client = _make_client(base_url=base)
    resp = await client.complete(_basic_request())
    assert resp.message.content == "hello"

    # Confirm the SDK hit the custom base_url, not the default OpenAI URL.
    request = httpx_mock.get_request()
    assert request is not None
    assert str(request.url) == f"{base}/chat/completions"


async def test_complete_sends_provider_agnostic_payload(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            temperature=0.7,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["model"] == "gpt-4o"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 128
    assert body["response_format"] == {"type": "json_object"}
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


async def test_per_request_model_overrides_client_default(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client(model="gpt-3.5-turbo")  # client default
    await client.complete(_basic_request(model="gpt-4o-mini"))  # request override
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["model"] == "gpt-4o-mini"


async def test_client_default_model_used_when_request_omits_it(httpx_mock: HTTPXMock) -> None:
    """A request whose model is falsy is sent with the client's default model.

    BR-023 moved this test onto pytest-httpx. It used to patch
    ``client._client.chat.completions.create`` with a double returning a
    ``ChatCompletion``; ``complete()`` now calls ``with_raw_response.create``
    and decodes the raw response itself, so such a double no longer fits
    (``_client`` is private).
    """
    # The Pydantic model requires `model`, so construct via model_construct
    # to bypass validation with a falsy model.
    req = ChatRequest.model_construct(
        messages=[ChatMessage(role="user", content="hi")],
        model="",  # falsy → triggers default lookup
        temperature=0.0,
        max_tokens=None,
        response_format=None,
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client(model="default-model")

    await client.complete(req)

    raw = httpx_mock.get_request()
    assert raw is not None
    assert json.loads(raw.read())["model"] == "default-model"


async def test_complete_raises_when_no_model_anywhere() -> None:
    client = _make_client(model=None)
    req = ChatRequest.model_construct(
        messages=[ChatMessage(role="user", content="hi")],
        model="",
        temperature=0.0,
        max_tokens=None,
        response_format=None,
    )
    with pytest.raises(LLMError) as exc:
        await client.complete(req)
    assert "No model specified" in exc.value.message


async def test_complete_omits_optional_fields_when_unset(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "max_tokens" not in body
    assert "response_format" not in body


async def test_build_body_sends_default_temperature(httpx_mock: HTTPXMock) -> None:
    """The ``0.0`` default IS sent — the key is present unless explicitly ``None``."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["temperature"] == 0.0


async def test_build_body_omits_temperature_when_none(httpx_mock: HTTPXMock) -> None:
    """``temperature=None`` removes the key entirely (reasoning-model providers)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(temperature=None))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "temperature" not in body


def _sent_body(httpx_mock: HTTPXMock) -> dict[str, Any]:
    raw = httpx_mock.get_request()
    assert raw is not None
    body: dict[str, Any] = json.loads(raw.read())
    return body


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "llama-3.1-70b", "gemini-2.5-pro", "gpt-50"],
)
async def test_build_body_sends_max_tokens_for_legacy_models(
    httpx_mock: HTTPXMock, model: str
) -> None:
    """Models that accept ``max_tokens`` keep the pre-existing wire shape."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model=model, max_tokens=64))
    body = _sent_body(httpx_mock)
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5.1",
        "gpt-5-mini",
        "GPT-5.1",
        "openai/gpt-5.1",
        "o1",
        "o3-mini",
        "o4-mini",
        "ft:o4-mini:org::abc",
    ],
)
async def test_build_body_sends_max_completion_tokens_for_reasoning_models(
    httpx_mock: HTTPXMock, model: str
) -> None:
    """gpt-5.x / o-series reject ``max_tokens`` with HTTP 400; send the successor key."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model=model, max_tokens=64))
    body = _sent_body(httpx_mock)
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body


async def test_build_body_reasoning_model_uses_client_default_model(
    httpx_mock: HTTPXMock,
) -> None:
    """The rule applies to the resolved model, including the client default."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client(model="gpt-5.1")
    req = ChatRequest.model_construct(
        messages=[ChatMessage(role="user", content="hi")],
        model="",
        temperature=0.0,
        max_tokens=32,
        response_format=None,
        tools=None,
        tool_choice=None,
    )
    await client.complete(req)
    body = _sent_body(httpx_mock)
    assert body["model"] == "gpt-5.1"
    assert body["max_completion_tokens"] == 32
    assert "max_tokens" not in body


async def test_build_body_reasoning_model_stream_uses_max_completion_tokens(
    httpx_mock: HTTPXMock,
) -> None:
    sse = _sse(_chunk(delta_content="hi", finish_reason="stop", model="gpt-5.1")) + (
        b"data: [DONE]\n\n"
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, content=sse)
    client = _make_client()
    async for _ in client.stream(_basic_request(model="gpt-5.1", max_tokens=16)):
        pass
    body = _sent_body(httpx_mock)
    assert body["stream"] is True
    assert body["max_completion_tokens"] == 16
    assert "max_tokens" not in body


async def test_build_body_reasoning_model_omits_both_keys_when_unset(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model="gpt-5.1"))
    body = _sent_body(httpx_mock)
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


# ---------------------------------------------------------------------------
# FR-002: reasoning_effort on the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.1",
        "gpt-5",
        "o3-mini",
        "openai/gpt-5.1",
        "gpt-4o",
        "llama-3.1-70b",
        "my-azure-deployment",
    ],
)
async def test_build_body_sends_reasoning_effort_for_any_model(
    httpx_mock: HTTPXMock, model: str
) -> None:
    """A set ``reasoning_effort`` reaches the HTTP body for every model name (FR-002 AC-2, D3).

    The non-reasoning names (``gpt-4o``, ``llama-3.1-70b``, an Azure-style
    deployment) pin the decision that there is no model-name filter: the value
    is sent and a provider that rejects it answers with an error.
    """
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model=model, reasoning_effort="medium"))
    body = _sent_body(httpx_mock)
    assert body["reasoning_effort"] == "medium"
    assert "extra_body" not in body


@pytest.mark.parametrize("model", ["gpt-5.1", "o3-mini", "gpt-4o", "llama-3.1-70b"])
async def test_build_body_omits_reasoning_effort_when_unset(
    httpx_mock: HTTPXMock, model: str
) -> None:
    """Unset ``reasoning_effort`` adds no key to the HTTP body or the built dict (FR-002 AC-1)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    request = _basic_request(model=model)
    await client.complete(request)
    body = _sent_body(httpx_mock)
    assert "reasoning_effort" not in body
    assert "extra_body" not in client._build_body(request, model=model, stream=False)


async def test_build_body_sends_string_none_reasoning_effort(httpx_mock: HTTPXMock) -> None:
    """The STRING ``"none"`` is a provider level and is sent; Python ``None`` is not (FR-002 D4)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model="gpt-5.1", reasoning_effort="none"))
    body = _sent_body(httpx_mock)
    assert body["reasoning_effort"] == "none"


@pytest.mark.parametrize("level", ["xhigh", "ultra"])
async def test_build_body_passes_unknown_lowercase_reasoning_effort_verbatim(
    httpx_mock: HTTPXMock, level: str
) -> None:
    """A lowercase level the SDK does not know is sent unchanged; the provider decides (FR-002 D4)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(model="gpt-5.1", reasoning_effort=level))
    body = _sent_body(httpx_mock)
    assert body["reasoning_effort"] == level


async def test_stream_sends_reasoning_effort(httpx_mock: HTTPXMock) -> None:
    """The streaming path sends ``reasoning_effort`` too (FR-002 AC-2)."""
    sse = _sse(_chunk(finish_reason="stop")) + b"data: [DONE]\n\n"
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=sse,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    async for _ in client.stream(_basic_request(model="gpt-5.1", reasoning_effort="low")):
        pass
    body = _sent_body(httpx_mock)
    assert body["stream"] is True
    assert body["reasoning_effort"] == "low"
    assert "extra_body" not in body


async def test_build_body_reasoning_effort_coexists_with_max_completion_tokens_and_tools(
    httpx_mock: HTTPXMock,
) -> None:
    """``reasoning_effort`` does not disturb BR-018's key choice or the tools block (FR-002)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    await client.complete(
        _basic_request(model="gpt-5.1", max_tokens=32, tools=tools, reasoning_effort="high")
    )
    body = _sent_body(httpx_mock)
    assert body["reasoning_effort"] == "high"
    assert body["max_completion_tokens"] == 32
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert "max_tokens" not in body


async def test_build_body_reasoning_effort_via_extra_body() -> None:
    """The built kwargs carry ``reasoning_effort`` under ``extra_body``, not top level (FR-002 D5).

    Deliberate: the typed ``create(reasoning_effort=)`` kwarg does not exist on
    every ``openai`` release in the declared range (floor 1.30.0), where it
    would raise an unwrapped ``TypeError``. ``extra_body`` works across the
    range. Switching to the typed kwarg is a dependency-floor decision, not a
    cleanup.
    """
    client = _make_client()
    try:
        request = _basic_request(model="gpt-5.1", reasoning_effort="low")
        built = client._build_body(request, model="gpt-5.1", stream=False)
    finally:
        await client.aclose()
    assert built == {
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.0,
        "extra_body": {"reasoning_effort": "low"},
    }
    assert "reasoning_effort" not in built


@pytest.mark.parametrize(
    "param,model",
    [
        # Forcing the successor key for a model the rule cannot recognize
        # (an Azure-style deployment name).
        ("max_completion_tokens", "my-reasoning-deployment"),
        # Forcing the legacy key for a gateway that only understands it.
        ("max_tokens", "gpt-5.1"),
    ],
)
async def test_max_tokens_param_option_overrides_model_rule(
    httpx_mock: HTTPXMock, param: str, model: str
) -> None:
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = OpenAICompatibleClient(
        api_key="test-key",
        base_url=BASE_URL,
        max_retries=0,
        max_tokens_param=param,  # type: ignore[arg-type]
    )
    await client.complete(_basic_request(model=model, max_tokens=64))
    body = _sent_body(httpx_mock)
    other = "max_tokens" if param == "max_completion_tokens" else "max_completion_tokens"
    assert body[param] == 64
    assert other not in body


def test_max_tokens_param_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="max_tokens_param"):
        OpenAICompatibleClient(api_key="test-key", max_tokens_param="max_output_tokens")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "upstream,expected",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
        ("function_call", "tool_calls"),  # legacy → tool_calls
    ],
)
async def test_complete_normalizes_finish_reason(
    httpx_mock: HTTPXMock, upstream: str, expected: str
) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        json=_canonical_response(finish_reason=upstream),
    )
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.finish_reason == expected


async def test_complete_handles_null_content(httpx_mock: HTTPXMock) -> None:
    """Some providers send ``content: null`` when only tool_calls are emitted."""
    payload = _canonical_response(finish_reason="tool_calls")
    payload["choices"][0]["message"]["content"] = None
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.message.content == ""
    assert resp.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Native tool_calls mapping (BR-007)
# ---------------------------------------------------------------------------


async def test_complete_maps_native_tool_calls(httpx_mock: HTTPXMock) -> None:
    """OpenAI tool_calls (JSON-string arguments) map to SDK ToolCall list."""
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls is not None
    assert len(resp.message.tool_calls) == 1
    mapped = resp.message.tool_calls[0]
    assert isinstance(mapped, ToolCall)
    assert mapped.name == "search"
    # The JSON-string `arguments` is parsed into a dict.
    assert mapped.args == {"q": "x"}


async def test_complete_maps_native_tool_calls_empty_arguments(httpx_mock: HTTPXMock) -> None:
    """An empty-arguments tool call maps to an empty args dict."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "ping", "arguments": ""},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())

    assert resp.message.tool_calls is not None
    assert resp.message.tool_calls[0].name == "ping"
    assert resp.message.tool_calls[0].args == {}


async def test_complete_malformed_arguments_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """Non-JSON `arguments` raises LLMError(MalformedResponse) with the call id."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "not json"},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_1"
    assert exc.value.context["model"] == "gpt-4o"
    assert exc.value.context["arguments_excerpt"] == "not json"
    assert exc.value.__cause__ is not None


async def test_complete_oversized_integer_arguments_raise_llm_error(
    httpx_mock: HTTPXMock,
) -> None:
    """BR-013 contains bare ValueError from native tool-call argument decoding."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    arguments = f'{{"n":{digits}}}'
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_big",
                "type": "function",
                "function": {"name": "calculate", "arguments": arguments},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert str(exc.value) == "provider tool_call arguments is not valid JSON"
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_big"
    assert len(str(exc.value.context["arguments_excerpt"])) <= 200
    assert type(exc.value.__cause__) is ValueError


async def test_complete_non_object_arguments_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """A non-object JSON `arguments` (e.g. a bare array) is rejected."""
    payload = _canonical_response(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "search", "arguments": "[1, 2, 3]"},
            }
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_3"


# ---------------------------------------------------------------------------
# Tool-call argument nesting limit (BR-019)
# ---------------------------------------------------------------------------

_DEPTH_MESSAGE = f"provider tool_call arguments nest deeper than {MAX_TOOL_ARGS_DEPTH} levels"


def _nested_arguments(depth: int) -> str:
    """Arguments text nested ``depth`` levels: ``{"b":"[","a":[[...0...]]}`` (BR-019).

    The ``"["`` string gives the text one more ``[`` than levels, so the depth
    check reads past its ``str.count`` shortcut at the limit too.
    """
    return '{"b":"[","a":' + "[" * (depth - 1) + "0" + "]" * (depth - 1) + "}"


def _tool_call_entry(call_id: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "search", "arguments": arguments},
    }


def _decode_maximum_here() -> int:
    """Deepest ``_nested_arguments`` text ``json.loads`` decodes, measured from this frame.

    Doubling from 128, then a binary search. Each probe text is built fresh
    and only decoded; nothing deep reaches an ``assert``.
    """
    lo, hi = 1, 128
    while True:
        try:
            json.loads(_nested_arguments(hi))
        except RecursionError:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            json.loads(_nested_arguments(mid))
            lo = mid
        except RecursionError:
            hi = mid
    return lo


async def test_complete_accepts_tool_call_arguments_nested_at_the_limit(
    httpx_mock: HTTPXMock,
) -> None:
    """Arguments nested exactly the limit (64 levels) decode as before (BR-019)."""
    arguments = _nested_arguments(MAX_TOOL_ARGS_DEPTH)
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_tool_call_entry("call_at_limit", arguments)],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    resp = await _make_client().complete(_basic_request())

    assert resp.finish_reason == "tool_calls"
    assert resp.message.tool_calls is not None
    assert resp.message.tool_calls[0].args == json.loads(arguments)


async def test_complete_refuses_tool_call_arguments_one_past_the_limit(
    httpx_mock: HTTPXMock,
) -> None:
    """One level past the limit raises a fixed-message MalformedResponse LLMError (BR-019)."""
    arguments = _nested_arguments(MAX_TOOL_ARGS_DEPTH + 1)
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_tool_call_entry("call_deep", arguments)],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    with pytest.raises(LLMError) as exc:
        await _make_client().complete(_basic_request())

    assert str(exc.value) == _DEPTH_MESSAGE
    for fragment in ("[", "{", '"a"', '"b"'):
        assert fragment not in str(exc.value)
    assert exc.value.context == {
        "model": "gpt-4o",
        "type": "MalformedResponse",
        "tool_call_id": "call_deep",
        "arguments_excerpt": arguments[:200],
        "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
    }
    assert exc.value.__cause__ is None


async def test_complete_refuses_arguments_deeper_than_the_decoder_reaches(
    httpx_mock: HTTPXMock,
) -> None:
    """Arguments nested past json.loads's own limit raise the depth LLMError, not RecursionError (BR-019).

    The depth is twice the decode maximum measured in this process. Before
    BR-019 the decoder raised RecursionError there, which the ``ValueError``
    arm does not catch, so it escaped ``complete()`` raw (BR-019 evidence,
    P1(c), on CPython 3.11.15, 3.13.2 and 3.14.3).
    """
    arguments = _nested_arguments(2 * _decode_maximum_here())
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_tool_call_entry("call_very_deep", arguments)],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    with pytest.raises(LLMError) as exc:
        await _make_client().complete(_basic_request())

    assert str(exc.value) == _DEPTH_MESSAGE
    assert exc.value.context["type"] == "MalformedResponse"
    assert exc.value.context["tool_call_id"] == "call_very_deep"
    assert exc.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert exc.value.context["arguments_excerpt"] == arguments[:200]


async def test_one_too_deep_entry_refuses_the_whole_batch(httpx_mock: HTTPXMock) -> None:
    """A too-deep second entry refuses the whole response, naming that entry (BR-019)."""
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            _tool_call_entry("call_ok", '{"q": "x"}'),
            _tool_call_entry("call_deep", _nested_arguments(MAX_TOOL_ARGS_DEPTH + 1)),
        ],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    with pytest.raises(LLMError) as exc:
        await _make_client().complete(_basic_request())

    assert str(exc.value) == _DEPTH_MESSAGE
    assert exc.value.context["tool_call_id"] == "call_deep"
    assert exc.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH


async def test_brackets_inside_argument_strings_are_not_nesting(httpx_mock: HTTPXMock) -> None:
    """A string holding 200 brackets is one level of arguments and decodes intact (BR-019)."""
    arguments = json.dumps({"q": "[" * 200 + "{" * 200})
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_tool_call_entry("call_brackets", arguments)],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    resp = await _make_client().complete(_basic_request())

    assert resp.message.tool_calls is not None
    assert resp.message.tool_calls[0].args == {"q": "[" * 200 + "{" * 200}


@pytest.mark.parametrize(
    "arguments",
    ["[" * (MAX_TOOL_ARGS_DEPTH + 1) + "not json", "x" + "[" * 100],
    ids=["deep_then_invalid", "invalid_at_the_first_character"],
)
async def test_depth_is_checked_before_json_validity(httpx_mock: HTTPXMock, arguments: str) -> None:
    """Invalid text whose brackets open past the limit gets the depth error, not the invalid-JSON one (BR-019).

    ``x`` followed by 100 ``[`` fails ``json.loads`` at its first character
    and nests nothing; before BR-019 it raised ``provider tool_call arguments
    is not valid JSON``. So ``max_tool_args_depth`` marks the check's
    refusal, not a value proved to nest deeply.
    """
    with pytest.raises(ValueError):
        json.loads(arguments)
    payload = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[_tool_call_entry("call_both", arguments)],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    with pytest.raises(LLMError) as exc:
        await _make_client().complete(_basic_request())

    assert str(exc.value) == _DEPTH_MESSAGE
    assert exc.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert exc.value.__cause__ is None


async def test_complete_no_tool_calls_field_is_none(httpx_mock: HTTPXMock) -> None:
    """The default response (no tool_calls) leaves `tool_calls` as None (additive)."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.message.tool_calls is None
    # Existing assertions on the additive path are unchanged.
    assert resp.message.content == "hello"
    assert resp.finish_reason == "stop"


async def test_complete_native_tool_call_excluded_from_request_body(
    httpx_mock: HTTPXMock,
) -> None:
    """A None tool_calls on request messages emits no `tool_calls` key on the wire."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    for msg in body["messages"]:
        assert "tool_calls" not in msg


# ---------------------------------------------------------------------------
# BR-008: tools/tool_choice declaration + assistant tool_calls wire envelope
# ---------------------------------------------------------------------------


async def test_build_body_no_tools_when_request_tools_unset(httpx_mock: HTTPXMock) -> None:
    """Flag-OFF proof at the adapter level: no tools/tool_choice on the wire."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request())
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert "tools" not in body
    assert "tool_choice" not in body


async def test_build_body_declares_tools_when_set(httpx_mock: HTTPXMock) -> None:
    """A set `request.tools` is emitted verbatim with `tool_choice="auto"` default."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(tools=tools))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


async def test_build_body_tool_choice_override(httpx_mock: HTTPXMock) -> None:
    """A caller-supplied `tool_choice` passes through instead of defaulting to auto."""
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            tools=[{"type": "function", "function": {"name": "x"}}], tool_choice="required"
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tool_choice"] == "required"


async def test_build_body_tool_choice_dict_form_on_wire(httpx_mock: HTTPXMock) -> None:
    """The specific-tool `tool_choice` object is emitted verbatim on the wire."""
    choice: ToolChoiceFunction = {"type": "function", "function": {"name": "search"}}
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(
        _basic_request(
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice=choice,
        )
    )
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assert body["tool_choice"] == {"type": "function", "function": {"name": "search"}}


async def test_assistant_tool_calls_envelope_on_wire(httpx_mock: HTTPXMock) -> None:
    """An assistant turn carrying tool_calls serializes to the OpenAI envelope shape.

    The `tool_calls[].id` is the message's `tool_call_id`, `type` is
    "function", and `function.arguments` is a JSON STRING (not an object) that
    round-trips back to the args dict.
    """
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="pairing-id-123",
        tool_calls=[ToolCall(name="search", args={"q": "x"})],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(messages=[assistant_msg]))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assistant_wire = body["messages"][0]
    assert assistant_wire["role"] == "assistant"
    assert assistant_wire["content"] == ""
    assert len(assistant_wire["tool_calls"]) == 1
    tc = assistant_wire["tool_calls"][0]
    assert tc["id"] == "pairing-id-123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    # arguments MUST be a JSON string, not an object.
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"q": "x"}


async def test_assistant_tool_calls_arguments_keep_non_ascii_literal(
    httpx_mock: HTTPXMock,
) -> None:
    """A replayed native turn's ``arguments`` string keeps Arabic literal, still decodes to the args, and keeps its ``id`` (BR-020)."""
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="pairing-id-123",
        tool_calls=[ToolCall(name="search", args={"q": "فاطمة"})],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(messages=[assistant_msg]))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    tc = body["messages"][0]["tool_calls"][0]
    assert tc["id"] == "pairing-id-123"
    arguments = tc["function"]["arguments"]
    assert isinstance(arguments, str)
    assert arguments == '{"q": "فاطمة"}'
    assert json.loads(arguments) == {"q": "فاطمة"}


async def test_assistant_tool_calls_arguments_with_lone_surrogate_still_send(
    httpx_mock: HTTPXMock,
) -> None:
    """``arguments`` holding a surrogate code point gets the escaped form, so ``complete()`` sends the request and returns (BR-020).

    Its first leg runs through the ``openai`` release's own body encoder: on
    the dev environment's 2.43.0, which encodes the body as strict UTF-8
    without escaping, a literal surrogate made ``complete()`` raise
    ``UnicodeEncodeError`` before sending.
    The exact-``arguments`` assertion requires the escaped text on any
    release; that leg was not run against a release that escapes the body.
    ``tests/test_model_json.py`` and ``tests/loop/test_loop_non_ascii.py`` pin
    the fallback whatever the ``openai`` release.
    """
    surrogate = chr(0xD800)
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="pairing-id-123",
        tool_calls=[ToolCall(name="search", args={"q": surrogate})],
    )
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    resp = await client.complete(_basic_request(messages=[assistant_msg]))
    assert resp.message.content == "hello"
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    arguments = body["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert arguments == '{"q": "\\ud800"}'
    assert json.loads(arguments) == {"q": surrogate}


async def test_id_pairing_assistant_id_matches_tool_reply(httpx_mock: HTTPXMock) -> None:
    """The assistant tool_calls id and the tool reply tool_call_id pair on the wire."""
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id="X",
        tool_calls=[ToolCall(name="search", args={"q": "x"})],
    )
    tool_reply = ChatMessage(role="tool", content="result", name="search", tool_call_id="X")
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=_canonical_response())
    client = _make_client()
    await client.complete(_basic_request(messages=[assistant_msg, tool_reply]))
    raw = httpx_mock.get_request()
    assert raw is not None
    body = json.loads(raw.read())
    assistant_wire = body["messages"][0]
    tool_wire = body["messages"][1]
    assistant_tc_id = assistant_wire["tool_calls"][0]["id"]
    assert assistant_tc_id == "X"
    assert tool_wire["role"] == "tool"
    assert tool_wire["tool_call_id"] == "X"
    # The pairing key matches across both messages.
    assert assistant_tc_id == tool_wire["tool_call_id"]


async def test_two_turn_native_no_400(httpx_mock: HTTPXMock) -> None:
    """End-to-end: a two-turn native round-trip produces a well-formed wire envelope.

    Turn 1 returns a native tool_call; turn 2 (after the assistant turn + tool
    reply are appended) returns a final answer. The request the mock RECEIVES
    on turn 2 MUST carry the OpenAI assistant envelope (tool_calls[].id
    present) AND a paired role="tool" reply whose tool_call_id matches — the
    shape a strict OpenAI endpoint requires to NOT return 400. This asserts
    the request envelope SHAPE, not merely that no exception was raised.
    """
    tool_call_id = "call_abc"
    turn1 = _canonical_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "x"}'},
            }
        ],
    )
    turn2 = _canonical_response(content="the answer", finish_reason="stop")
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=turn1)
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=turn2)
    client = _make_client()

    # Turn 1: plain request; provider returns a native tool_call.
    resp1 = await client.complete(_basic_request(messages=[ChatMessage(role="user", content="q")]))
    assert resp1.finish_reason == "tool_calls"
    assert resp1.message.tool_calls is not None
    # Loop would now synthesize the assistant turn (carrying the pairing id)
    # and the tool reply from the response.
    assistant_msg = ChatMessage(
        role="assistant",
        content="",
        tool_call_id=tool_call_id,
        tool_calls=resp1.message.tool_calls,
    )
    tool_reply = ChatMessage(
        role="tool", content="search results", name="search", tool_call_id=tool_call_id
    )
    # Turn 2: replay with the assistant + tool turns in history.
    resp2 = await client.complete(
        _basic_request(messages=[ChatMessage(role="user", content="q"), assistant_msg, tool_reply])
    )
    assert resp2.finish_reason == "stop"

    # Assert the SECOND request's envelope shape (the turn that would 400 on a
    # strict endpoint if the pairing were broken).
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    body = json.loads(requests[1].read())
    assistant_wire = next(
        m for m in body["messages"] if m["role"] == "assistant" and "tool_calls" in m
    )
    tool_wire = next(m for m in body["messages"] if m["role"] == "tool")
    assert assistant_wire["tool_calls"][0]["id"] == tool_call_id
    assert assistant_wire["tool_calls"][0]["type"] == "function"
    # arguments is a JSON STRING on the wire.
    assert isinstance(assistant_wire["tool_calls"][0]["function"]["arguments"], str)
    assert json.loads(assistant_wire["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert tool_wire["tool_call_id"] == tool_call_id
    # THE pairing invariant: both sides carry the same id.
    assert assistant_wire["tool_calls"][0]["id"] == tool_wire["tool_call_id"]


# ---------------------------------------------------------------------------
# Error paths: complete()
# ---------------------------------------------------------------------------


async def test_connection_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("nope"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "APIConnectionError"
    assert exc.value.context["model"] == "gpt-4o"
    assert exc.value.__cause__ is not None


async def test_timeout_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.TimeoutException("slow"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "APITimeoutError"


async def test_rate_limit_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        status_code=429,
        json={"error": {"message": "rate"}},
    )
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert exc.value.context["type"] == "RateLimitError"


async def test_server_error_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        status_code=500,
        json={"error": {"message": "boom"}},
    )
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    # InternalServerError is an APIError subclass in the openai SDK.
    assert exc.value.context["type"] == "InternalServerError"


async def test_empty_choices_maps_to_llm_error(httpx_mock: HTTPXMock) -> None:
    payload = _canonical_response()
    payload["choices"] = []
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        await client.complete(_basic_request())
    assert "no choices" in exc.value.message.lower()
    assert exc.value.context["type"] == "MalformedResponse"


async def test_missing_usage_field_is_filled_with_zeros(httpx_mock: HTTPXMock) -> None:
    payload = _canonical_response()
    payload["usage"] = None
    httpx_mock.add_response(method="POST", url=ENDPOINT, json=payload)
    client = _make_client()
    resp = await client.complete(_basic_request())
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0
    assert resp.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def test_stream_happy_path(httpx_mock: HTTPXMock) -> None:
    body = b"".join(
        [
            _sse(_chunk(delta_role="assistant", delta_content="")),
            _sse(_chunk(delta_content="Hel")),
            _sse(_chunk(delta_content="lo")),
            _sse(_chunk(delta_content=" world")),
            _sse(_chunk(finish_reason="stop")),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    chunks = []
    async for chunk in client.stream(_basic_request()):
        chunks.append(chunk)

    # Each chunk holds the delta only.
    contents = [c.message.content for c in chunks]
    accum = "".join(contents)
    assert accum == "Hello world"
    # Intermediate chunks emit "in_progress"; only the terminal chunk
    # carries the real terminal reason from upstream.
    for intermediate in chunks[:-1]:
        assert intermediate.finish_reason == "in_progress"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].message.content == ""


async def test_stream_intermediate_chunks_use_in_progress(httpx_mock: HTTPXMock) -> None:
    """A naive consumer that breaks on `finish_reason == "stop"` must not exit early."""
    body = b"".join(
        [
            _sse(_chunk(delta_content="a")),
            _sse(_chunk(delta_content="b")),
            _sse(_chunk(delta_content="c")),
            _sse(_chunk(finish_reason="stop")),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    seen = []
    async for chunk in client.stream(_basic_request()):
        seen.append(chunk.finish_reason)
    # Every chunk before the last reports "in_progress"; only the last
    # carries the real "stop".
    assert seen[:-1] == ["in_progress"] * (len(seen) - 1)
    assert seen[-1] == "stop"


@pytest.mark.parametrize(
    "upstream_terminal,expected_terminal",
    [
        ("stop", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("content_filter", "content_filter"),
    ],
)
async def test_stream_terminal_chunk_preserves_upstream_reason(
    httpx_mock: HTTPXMock,
    upstream_terminal: str,
    expected_terminal: str,
) -> None:
    body = b"".join(
        [
            _sse(_chunk(delta_content="hi")),
            _sse(_chunk(finish_reason=upstream_terminal)),
            b"data: [DONE]\n\n",
        ]
    )
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    chunks = []
    async for chunk in client.stream(_basic_request()):
        chunks.append(chunk)
    assert chunks[-1].finish_reason == expected_terminal


async def test_stream_request_body_marks_stream_true(httpx_mock: HTTPXMock) -> None:
    body = _sse(_chunk(finish_reason="stop")) + b"data: [DONE]\n\n"
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    async for _ in client.stream(_basic_request()):
        pass
    raw = httpx_mock.get_request()
    assert raw is not None
    payload = json.loads(raw.read())
    assert payload["stream"] is True


async def test_stream_connection_error_during_open(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("nope"))
    client = _make_client()
    with pytest.raises(LLMError) as exc:
        async for _ in client.stream(_basic_request()):
            pass
    assert exc.value.context["type"] == "APIConnectionError"


async def test_stream_malformed_chunk_raises_llm_error(httpx_mock: HTTPXMock) -> None:
    """A chunk with a non-dict ``delta`` triggers the defensive mapper."""
    bad = {
        "id": "cmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": "not-an-object", "finish_reason": None}],
    }
    body = _sse(bad) + b"data: [DONE]\n\n"
    httpx_mock.add_response(
        method="POST",
        url=ENDPOINT,
        content=body,
        headers={"content-type": "text/event-stream"},
    )
    client = _make_client()
    # The openai SDK validates chunks against its own model and may raise
    # before our mapper sees them. Either path must surface as LLMError.
    with pytest.raises(LLMError):
        async for _ in client.stream(_basic_request()):
            pass


# ---------------------------------------------------------------------------
# Lifecycle: aclose() and the async context manager
# ---------------------------------------------------------------------------


async def test_aclose_closes_owned_client() -> None:
    """An owned client (no injected http_client) is closed by aclose()."""
    client = _make_client()
    await client.aclose()
    assert client._client.is_closed()


async def test_aclose_is_idempotent() -> None:
    """A second ``aclose()`` is a no-op and MUST NOT raise."""
    client = _make_client()
    await client.aclose()
    await client.aclose()
    assert client._client.is_closed()


async def test_aclose_does_not_close_injected_http_client() -> None:
    """An injected ``http_client`` stays open after aclose() — the caller owns it."""
    injected = httpx.AsyncClient()
    client = _make_client(http_client=injected)
    await client.aclose()
    assert not injected.is_closed
    await injected.aclose()


async def test_async_context_manager_closes_owned_client_on_exit() -> None:
    """``async with`` returns the client itself and aclose()s it on exit."""
    async with _make_client() as client:
        assert isinstance(client, OpenAICompatibleClient)
        assert not client._client.is_closed()
    assert client._client.is_closed()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_client_satisfies_llmclient_protocol_at_runtime() -> None:
    client = OpenAICompatibleClient(api_key="x", base_url=BASE_URL, model="gpt-4o", max_retries=0)
    assert isinstance(client, LLMClient)
