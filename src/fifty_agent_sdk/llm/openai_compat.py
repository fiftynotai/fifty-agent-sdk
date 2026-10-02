"""OpenAI-compatible :class:`LLMClient` adapter.

Backed by the official ``openai`` Python SDK pointed at any OpenAI-compatible
``/v1/chat/completions`` endpoint — OpenAI itself, Google Distributed Cloud
(GDC), local OSS servers (vLLM, Ollama via the openai-compat layer, etc.).
The provider differences are absorbed by ``base_url``.

All provider SDK exceptions are wrapped into :class:`fifty_agent_sdk.errors.LLMError`
at the public method boundary, in line with the
:class:`fifty_agent_sdk.llm.protocol.LLMClient` contract.

Provider bodies and context-window errors (BR-021)
    An OpenAI-compatible endpoint can answer an error with HTTP 200 and a
    body that is not a chat completion, for example plain text. Measured on
    ``openai`` 2.43.0, such a body reaches the adapter as a ``str``, as a
    :class:`json.JSONDecodeError` (a body labelled JSON whose text is not
    JSON), or on a streaming request as a stream with no events. The adapter
    raises :class:`~fifty_agent_sdk.errors.LLMError` for each, with one of
    these ``context["type"]`` values:

    * ``"NonJsonProviderBody"``: the adapter got text where it expected a
      JSON object, as the response body or as the data of a stream event:
      text that is not JSON, a bare JSON string, or (a non-streaming body
      whose Content-Type is not JSON) JSON the client could not decode.
    * ``"NonStreamProviderBody"``: a streaming request got a 200 whose
      media type is not ``text/event-stream``, or that has no
      ``Content-Type``, and that produced no chunk. The type names what was
      detected, not the body's format: an SSE-framed body holding only
      ``data: [DONE]`` gets it too. ``context["content_type"]`` holds the
      media type (``""`` when the header is missing), cut to 100 characters.
    * ``"ContextLengthExceeded"``: see below.

    The message is a fixed prefix plus the first 500 characters of the
    stripped provider text, with ``…[truncated]`` appended when it was cut.
    That text is the decoded body, the value of a body that is a bare JSON
    string, or the data of the one stream event that failed. ``context``
    holds ``model``, ``type``, ``body_length`` (the length in characters of
    that text before stripping) and, on a stream, ``phase="stream"`` (plus
    ``content_type`` for ``NonStreamProviderBody``, the provider's media type
    cut to 100 characters). It holds none of the body's text.

    Not every such body becomes one of these errors (measured on 2.43.0):

    * A non-streaming 200 whose body is a JSON object without ``choices``,
      such as an error envelope ``{"error": {...}}``, stays
      ``"MalformedResponse"`` (``Provider returned no choices.``) and its
      text is not kept, labelled ``application/json`` or ``text/plain``. On a
      streaming request the same body is a ``NonStreamProviderBody`` error
      carrying its text (``ContextLengthExceeded`` when that text holds a
      marker), unless it is labelled ``text/event-stream``; see the next
      item.
    * A streaming 200 labelled ``text/event-stream`` whose body holds no SSE
      field (plain text, or a JSON error envelope) yields no chunk and no
      error: the ``openai`` SSE decoder drops it before the adapter sees it.
    * A non-streaming 200 with a JSON Content-Type still escapes
      ``complete()`` raw, as before BR-021, when its body is not valid UTF-8
      (``UnicodeDecodeError``; measured labelled ``application/json``,
      ``application/problem+json`` and ``text/json``) or holds an integer
      literal over the interpreter's digit limit (``ValueError``; measured
      labelled ``application/json``). The ``openai`` 2.43.0 source treats a
      Content-Type as JSON when the part before ``;`` ends in ``json``.
    * A 200 body, or a stream event's data, that is JSON but neither an
      object nor a string (an array, a number, ``null``, ``true`` or
      ``false``) still raises ``"MalformedResponse"`` or
      ``"MalformedChunk"`` without its text, as before BR-021 (measured
      labelled ``application/json``).

    The provider text is diagnostic text, not end-user text. It
    reaches ``LLMError.message``, and from there
    :attr:`fifty_agent_sdk.streaming.ErrorEvent.message`, the Runner's
    ``error`` audit payload (``error_message``) and the ``on_error`` hook,
    the same places the provider's 4xx/5xx text already reaches (that text
    is not cut to a bound). :class:`fifty_agent_sdk.loop.AgentLoop` ends
    the run with :attr:`fifty_agent_sdk.safety.SafetyConfig.
    error_fallback_message`, never with this text.

    Context-window overflow is classified on these paths only: the
    ``NonJsonProviderBody`` and ``NonStreamProviderBody`` errors above, and
    any ``openai.APIError`` other than ``APITimeoutError``,
    ``APIConnectionError`` or ``RateLimitError`` (``BadRequestError``,
    ``InternalServerError``, a mid-stream error event, ...), whether
    ``complete()`` gets it, ``stream()`` gets it when opening the request,
    or ``stream()`` gets it while iterating. It is never applied to those
    three when ``complete()`` gets them or ``stream()`` opens the request.
    While iterating, ``stream()`` has no ``RateLimitError`` arm: ``openai``
    2.43.0 raises only the base ``APIError`` for a mid-stream error event,
    so no rate limit was measured there. It is never applied to any other
    failure while a stream is read (an ``httpx`` read error, for example),
    or to the bodies the list above leaves unhandled. When the full text
    contains ``max_prompt_length``, ``context_length_exceeded`` or
    ``maximum context length`` (compared case-insensitively), or the
    error's ``code`` is ``context_length_exceeded``, ``context["type"]`` is
    ``"ContextLengthExceeded"`` and ``context["classified_from"]`` holds the
    type it would otherwise have had. The marker list is closed: other
    phrasings of the same failure stay unclassified, and an unrelated error
    whose text quotes a marker is misclassified.

Tool-call arguments (BR-019)
    Before decoding the ``arguments`` string of a native tool call, the
    adapter checks how deeply it nests, one level per JSON object or array
    (``{"q": [1]}`` is 2 levels; brackets inside strings do not count).
    Arguments nested deeper than 64 levels
    (:data:`fifty_agent_sdk._json_depth.MAX_TOOL_ARGS_DEPTH`) raise
    :class:`~fifty_agent_sdk.errors.LLMError` with ``context["type"] ==
    "MalformedResponse"`` and the fixed message ``provider tool_call
    arguments nest deeper than 64 levels``. ``context`` holds ``model``,
    ``type``, ``tool_call_id``, ``arguments_excerpt`` (the first 200
    characters of the arguments, as for invalid JSON) and
    ``max_tool_args_depth`` (``64``), the key that tells this refusal apart
    from the other ``MalformedResponse`` errors. The check reads brackets,
    not JSON: invalid text whose brackets open more than 64 levels (``x``
    followed by 100 ``[``, for example) gets this error too, where earlier
    releases raised ``provider tool_call arguments is not valid JSON``, so
    the key means the check refused the text, not that the text decodes to
    a deeply nested value. The whole response is
    refused: ``complete()`` returns no :class:`ChatResponse`, so one such
    entry in a multi-call response refuses every call in it. Before BR-019
    such arguments were decoded wherever ``json.loads`` could decode them,
    and re-encoding them for the next request could raise a raw
    :class:`RecursionError`; past ``json.loads``'s own limit, the decode
    itself raised it, out of ``complete()``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Literal, get_args

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds
from fifty_agent_sdk._model_json import dumps_for_model
from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    ToolCall,
    Usage,
)

# OpenAI returns ``"function_call"`` as a legacy finish reason that maps to
# our protocol's ``"tool_calls"``. Anything outside this map is treated as an
# error condition (see :func:`_normalize_finish_reason`).
_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
    "error": "error",
}

_MAX_TOOL_CALL_ARG_EXCERPT: int = 200
"""Maximum length of ``arguments_excerpt`` in the malformed-arguments context."""

_MAX_PROVIDER_BODY_EXCERPT: int = 500
"""Maximum characters of a provider body quoted in an ``LLMError`` message (BR-021)."""

_PROVIDER_BODY_TRUNCATION_MARKER: str = "…[truncated]"
"""Appended to a provider-body excerpt that was cut (BR-021).

The same text as the Runner's result-summary marker. It is defined here
because ``llm/`` must not import ``runner.py``.
"""

_CONTEXT_LENGTH_MARKERS: tuple[str, ...] = (
    "max_prompt_length",
    "context_length_exceeded",
    "maximum context length",
)
"""Lower-case substrings that classify a provider error as a context-window overflow.

BR-021. Matched case-insensitively against the full provider text. The list
is closed: other phrasings stay unclassified, and a provider error that
merely quotes one of these strings is misclassified. Adding a marker moves
those errors to a new ``context["type"]``, so it is a CHANGELOG item.
"""

_CONTEXT_LENGTH_CODE: str = "context_length_exceeded"
"""Provider error ``code`` that classifies an error as a context-window overflow (BR-021)."""

_EVENT_STREAM_MEDIA_TYPE: str = "text/event-stream"
"""The one media type a streamed response is iterated without reading it first (BR-021)."""

_MAX_CONTENT_TYPE_LENGTH: int = 100
"""Maximum length of the media type recorded as ``context["content_type"]`` (BR-021)."""

# Fixed message prefixes of the three provider-body errors (BR-021). The
# excerpt of the provider's own text follows the prefix.
_NON_JSON_BODY_PREFIX: str = "Provider response body could not be read as a JSON object: "
_NON_JSON_EVENT_PREFIX: str = "Provider stream event is not a JSON object: "
# The stream prefix states what was detected (no chunk, and a media type that
# is not text/event-stream), not that the body is not SSE: an SSE-framed body
# holding only `data: [DONE]` also ends here.
_NON_STREAM_BODY_PREFIX: str = (
    "Provider stream produced no chunk and its Content-Type is not text/event-stream. Body: "
)

_MaxTokensParam = Literal["max_tokens", "max_completion_tokens"]
"""Request-body key that carries :attr:`ChatRequest.max_tokens` on the wire."""

# OpenAI's reasoning-model families (o1 / o3 / o4 and gpt-5.x) reject
# ``max_tokens`` with HTTP 400 ("Unsupported parameter ... Use
# 'max_completion_tokens' instead"), including when served through
# OpenAI-compatible gateways. Matches the bare family name or a family name
# followed by a non-alphanumeric suffix (``gpt-5.1``, ``o3-mini``), optionally
# behind a routing/fine-tune prefix (``openai/gpt-5``, ``ft:o4-mini:...``).
_MAX_COMPLETION_TOKENS_MODEL_RE = re.compile(
    r"^(?:.*[/:])?(?:gpt-5|o[134])(?![0-9a-z])",
    re.IGNORECASE,
)


def _is_context_length_exceeded(text: str, code: object = None) -> bool:
    """Return whether a provider error reports a context-window overflow (BR-021).

    True when ``text`` contains any of :data:`_CONTEXT_LENGTH_MARKERS`
    (compared after :meth:`str.casefold`), or when ``code`` is a ``str``
    equal to :data:`_CONTEXT_LENGTH_CODE` after casefolding. The ``code``
    rule covers a provider error whose message carries no marker. A
    non-``str`` ``code`` is ignored.

    Args:
        text: The full provider text: a body, or ``str()`` of an ``openai``
            exception. Never an excerpt, so a marker past the excerpt bound
            still counts.
        code: The ``code`` attribute of an ``openai.APIError``, if any.

    Returns:
        ``True`` when the error should be typed ``"ContextLengthExceeded"``.
    """
    folded = text.casefold()
    if any(marker in folded for marker in _CONTEXT_LENGTH_MARKERS):
        return True
    return isinstance(code, str) and code.casefold() == _CONTEXT_LENGTH_CODE


def _classify(context: dict[str, Any], text: str, code: object = None) -> dict[str, Any]:
    """Re-type an error context as ``"ContextLengthExceeded"`` when the text says so (BR-021).

    Returns ``context`` itself, unchanged, when
    :func:`_is_context_length_exceeded` is false, so an unclassified error
    gains no key. Otherwise returns a new dict with ``type`` set to
    ``"ContextLengthExceeded"`` and ``classified_from`` set to the type it
    replaced (for example ``"BadRequestError"`` or ``"NonJsonProviderBody"``).

    Args:
        context: The error context, which must hold a ``"type"`` key.
        text: The full provider text the classification reads.
        code: The provider error ``code``, if any.

    Returns:
        The context to attach to the :class:`LLMError`.
    """
    if not _is_context_length_exceeded(text, code):
        return context
    return {**context, "type": "ContextLengthExceeded", "classified_from": context["type"]}


def _provider_body_error(
    text: str,
    *,
    model: str,
    type_: str,
    prefix: str,
    extra: dict[str, Any] | None = None,
) -> LLMError:
    """Build the :class:`LLMError` for a provider body the adapter cannot map (BR-021).

    The message is ``prefix`` plus the stripped ``text`` cut to
    :data:`_MAX_PROVIDER_BODY_EXCERPT` characters, with
    :data:`_PROVIDER_BODY_TRUNCATION_MARKER` appended when it was cut. The
    context holds ``model``, ``type``, ``body_length`` (``len(text)``,
    before stripping) and ``extra``, and is passed through :func:`_classify`
    on the WHOLE ``text``. The context holds none of ``text``.

    The error is returned, not raised, so each caller chooses its cause
    (``raise ... from e`` after a decode failure; no cause on a ``str`` body).

    Args:
        text: The provider text: the decoded body, the value of a body that
            is a bare JSON string, or the data of one stream event.
        model: Model identifier for the context.
        type_: The context type before classification.
        prefix: The fixed message prefix naming the failure.
        extra: Further context keys that carry no body text (``phase``,
            ``content_type``).

    Returns:
        The error to raise.
    """
    stripped = text.strip()
    excerpt = stripped[:_MAX_PROVIDER_BODY_EXCERPT]
    if len(stripped) > _MAX_PROVIDER_BODY_EXCERPT:
        excerpt += _PROVIDER_BODY_TRUNCATION_MARKER
    context: dict[str, Any] = {"model": model, "type": type_, "body_length": len(text)}
    if extra:
        context.update(extra)
    return LLMError(prefix + excerpt, context=_classify(context, text))


def _media_type(content_type: str | None) -> str:
    """Return the media type of a ``Content-Type`` header value, casefolded (BR-021).

    The parameters after ``;`` are dropped, so ``"text/event-stream;
    charset=utf-8"`` gives ``"text/event-stream"``. A missing header gives
    ``""``.
    """
    if content_type is None:
        return ""
    return content_type.split(";", 1)[0].strip().casefold()


class OpenAICompatibleClient:
    """:class:`LLMClient` implementation backed by the ``openai`` Python SDK.

    Works against any OpenAI-compatible ``/v1/chat/completions`` endpoint.
    Provider variation is absorbed by ``base_url`` — the same client class
    drives OpenAI itself, GDC, and local OSS servers.

    Async context management
        The client is an async context manager: ``async with
        OpenAICompatibleClient(...) as client:`` closes the underlying
        connection pool on exit via :meth:`aclose`. Without the context
        manager, call :meth:`aclose` explicitly — an unclosed client leaks
        its ``httpx.AsyncClient`` pool until GC.

    Args:
        api_key: API key passed to the upstream provider. Required even for
            local servers that ignore it; pass any non-empty string.
        base_url: Override the default OpenAI base URL. Use for GDC or a
            self-hosted endpoint. ``None`` means use the SDK default.
        model: Default model identifier used when a :class:`ChatRequest` does
            not set one. ``None`` means callers MUST set ``request.model``.
        timeout: Per-request timeout in seconds. Defaults to ``60.0``.
        max_retries: SDK-level retry count for transient failures (429, 5xx,
            connection errors). Defaults to ``2``. Set to ``0`` to make
            errors surface immediately, which is what tests want.
        http_client: Optional pre-configured ``httpx.AsyncClient``. Useful for
            tests that need to inject a mock transport. When omitted, the
            SDK builds its own client, OWNS it, and closes it on
            :meth:`aclose`. An INJECTED client is NOT closed by
            :meth:`aclose` — its lifecycle belongs to the caller. This
            owned-vs-injected discipline mirrors
            :class:`fifty_agent_sdk.mcp.client.MCPClient`.
        max_tokens_param: Which request-body key carries
            :attr:`ChatRequest.max_tokens`. ``None`` (the default) picks per
            request from the model name: ``"max_completion_tokens"`` for
            OpenAI reasoning-model families (``o1`` / ``o3`` / ``o4`` /
            ``gpt-5*``, which reject ``max_tokens``), ``"max_tokens"`` for
            everything else. Set it explicitly when the model name does not
            reveal the family — e.g. an Azure deployment name — or to force
            ``"max_tokens"`` for a server that does not understand
            ``max_completion_tokens``. Has no effect when ``max_tokens`` is
            unset. On reasoning models the cap includes reasoning tokens, so
            a small cap can exhaust the budget before any visible output.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        max_tokens_param: _MaxTokensParam | None = None,
    ) -> None:
        if max_tokens_param is not None and max_tokens_param not in get_args(_MaxTokensParam):
            raise ValueError(
                f"max_tokens_param must be one of {get_args(_MaxTokensParam)} or None; "
                f"got {max_tokens_param!r}"
            )
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url is not None:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = AsyncOpenAI(**kwargs)
        self._default_model = model
        self._max_tokens_param = max_tokens_param
        # Ownership discipline mirrors MCPClient: an injected http_client is
        # the caller's to close; only the client built here is closed by
        # aclose(). (`AsyncOpenAI.close()` closes the underlying httpx client
        # unconditionally, so it must only be called on the owned path.)
        self._owns_client = http_client is None
        self._closed = False

    async def aclose(self) -> None:
        """Close the underlying ``openai`` client and its connection pool.

        Only an OWNED client is closed: when ``http_client`` was injected at
        construction, its lifecycle belongs to the caller and ``aclose()``
        leaves it open. Idempotent: a second ``aclose()`` is a no-op and does
        NOT raise. The client MUST NOT be used after ``aclose()`` returns.
        """
        if self._closed:
            return
        if self._owns_client:
            await self._client.close()
        self._closed = True

    async def __aenter__(self) -> OpenAICompatibleClient:
        """Enter the async context manager, returning ``self``."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the async context manager, closing via :meth:`aclose`."""
        await self.aclose()

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a single non-streaming completion.

        Args:
            request: The chat completion request.

        Returns:
            The mapped :class:`ChatResponse`.

        Raises:
            fifty_agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call (network, timeout, rate-limit, malformed
                envelope, missing fields). A 200 whose body the adapter
                receives as text (text that is not JSON, or a bare JSON
                string; see the module docstring) is ``context["type"] ==
                "NonJsonProviderBody"``, with the body's start in the message
                (BR-021). A context-window overflow is
                ``"ContextLengthExceeded"`` (see the module docstring). Tool
                call ``arguments`` nested deeper than 64 levels are
                ``"MalformedResponse"`` with ``context["max_tool_args_depth"]``
                (BR-019; see the module docstring).
        """
        model = request.model or self._default_model
        if not model:
            raise LLMError(
                "No model specified: pass `model` to ChatRequest or set a default on the client.",
                context={"request_model": request.model, "client_default": self._default_model},
            )
        body = self._build_body(request, model=model, stream=False)
        try:
            raw = await self._client.chat.completions.create(**body)
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError"},
            ) from e
        except RateLimitError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "RateLimitError"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context=_classify(
                    {"model": model, "type": type(e).__name__},
                    str(e),
                    getattr(e, "code", None),
                ),
            ) from e
        except json.JSONDecodeError as e:
            # BR-021: a 200 labelled JSON whose body is not JSON. The openai
            # client decodes it outside its own error handling (measured on
            # 2.43.0), so the decode error would otherwise escape complete()
            # raw. `e.doc` is the decoded body. Deliberately NOT `ValueError`:
            # a `UnicodeEncodeError` (a ValueError) from encoding the request
            # must not be reported as a provider body. BR-022 escapes the
            # surrogate code points in the tool-result messages the loop
            # builds and adds no arm here (an arm would also catch httpx's
            # ascii UnicodeEncodeError for a non-ASCII API key). A surrogate
            # code point in text the loop does not build (the model's own
            # completion or tool names echoed back, messages passed to
            # `run()`) still raises raw from this `try`.
            # The cost, measured on 2.43.0: a JSON-labelled body whose integer
            # literal exceeds the interpreter's digit limit (`ValueError`), or
            # that is not valid UTF-8 (`UnicodeDecodeError`, also a
            # ValueError), still escapes raw, as it did before BR-021. Both
            # are tracked as BR-023.
            raise _provider_body_error(
                e.doc,
                model=model,
                type_="NonJsonProviderBody",
                prefix=_NON_JSON_BODY_PREFIX,
            ) from e
        return self._map_response(raw, model=model)

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream a completion as incremental chunks.

        Each yielded :class:`ChatResponse` carries the delta in
        ``message.content`` (not the running accumulation). Intermediate
        chunks emit ``finish_reason="in_progress"``; the terminal chunk
        carries a real terminal reason (``"stop"`` / ``"length"`` /
        ``"tool_calls"`` / ``"content_filter"`` / ``"error"``) mapped from
        the upstream provider's value. Consumers can therefore branch on
        ``finish_reason`` without misreading an intermediate delta as
        terminal.

        Non-event-stream responses (BR-021)
            A 200 whose media type is not ``text/event-stream`` (a missing
            ``Content-Type`` included) is read whole before its first chunk
            is yielded, so a direct caller gets that response's chunks only
            after its body has arrived. If it then yields no chunk, the
            adapter raises :class:`LLMError` with ``context["type"] ==
            "NonStreamProviderBody"`` and the body's start in the message,
            whatever the body holds: an SSE-framed body with only
            ``data: [DONE]`` gets this error too. If reading the body fails,
            the adapter raises :class:`LLMError` whose ``context["type"]``
            is the ``httpx`` exception's class name (for example
            ``"ReadError"``) and ``context["phase"]`` is ``"stream"``. A
            ``text/event-stream`` response is not read ahead. One case is
            not recoverable: a body without SSE fields (plain text, or a
            JSON error envelope) labelled ``text/event-stream`` yields no
            chunk and no error, because the ``openai`` SSE decoder discards
            it (measured on 2.43.0).

        Args:
            request: The chat completion request.

        Yields:
            :class:`ChatResponse` chunks containing the latest delta.

        Raises:
            fifty_agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call. Errors mid-stream surface from the iterator.
                A stream event whose data is not JSON, or is a bare JSON
                string, is ``context["type"] == "NonJsonProviderBody"``
                (before BR-021 the first was ``"JSONDecodeError"`` and the
                second ``"MalformedChunk"``). A context-window
                overflow is ``"ContextLengthExceeded"`` (see the module
                docstring); a failure of the ``httpx`` read itself, such as
                a mid-stream ``ReadTimeout``, is never classified.
                Provider-body errors carry ``context["phase"] == "stream"``.
        """
        model = request.model or self._default_model
        if not model:
            raise LLMError(
                "No model specified: pass `model` to ChatRequest or set a default on the client.",
                context={"request_model": request.model, "client_default": self._default_model},
            )
        body = self._build_body(request, model=model, stream=True)
        try:
            stream = await self._client.chat.completions.create(**body)
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError"},
            ) from e
        except RateLimitError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "RateLimitError"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context=_classify(
                    {"model": model, "type": type(e).__name__},
                    str(e),
                    getattr(e, "code", None),
                ),
            ) from e

        try:
            # BR-021 (D8): the openai client iterates a 200 stream without
            # looking at its Content-Type, and its SSE decoder silently drops
            # a body that is not SSE, so a plain-text error would look like an
            # empty stream. Read every non-event-stream body first (the openai
            # iterator then replays it from httpx's cache), so its text is
            # still available if it yields no chunk. `getattr` + `isinstance`
            # keep a stream object without an httpx `response` (another
            # openai release, a test double) on the unbuffered path.
            response = getattr(stream, "response", None)
            buffered: httpx.Response | None = None
            media_type = ""
            if isinstance(response, httpx.Response):
                media_type = _media_type(response.headers.get("content-type"))
                if media_type != _EVENT_STREAM_MEDIA_TYPE:
                    await response.aread()
                    buffered = response
            upstream_chunks = 0
            async for raw_chunk in stream:
                upstream_chunks += 1
                yield self._map_chunk(raw_chunk, model=model)
            if buffered is not None and upstream_chunks == 0:
                raise _provider_body_error(
                    buffered.text,
                    model=model,
                    type_="NonStreamProviderBody",
                    prefix=_NON_STREAM_BODY_PREFIX,
                    extra={
                        "phase": "stream",
                        "content_type": media_type[:_MAX_CONTENT_TYPE_LENGTH],
                    },
                )
        except APITimeoutError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APITimeoutError", "phase": "stream"},
            ) from e
        except APIConnectionError as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": "APIConnectionError", "phase": "stream"},
            ) from e
        except APIError as e:
            raise LLMError(
                str(e),
                context=_classify(
                    {"model": model, "type": type(e).__name__, "phase": "stream"},
                    str(e),
                    getattr(e, "code", None),
                ),
            ) from e
        except LLMError:
            # Allow defensive mapping errors raised from `_map_chunk` (and the
            # BR-021 provider-body errors above) to surface unchanged.
            raise
        except json.JSONDecodeError as e:
            # BR-021: SSE-framed data that is not JSON (`data: <text>`). `e.doc`
            # is the event data. Deliberately NOT `ValueError`: only a decode
            # failure carries provider text in `e.doc`, so any other
            # ValueError raised while iterating keeps the generic wrap below
            # rather than reading as a provider body. Encoding the request
            # happens in the `create()` call above, outside this `try`: a
            # surrogate code point in a message raises `UnicodeEncodeError`
            # raw there (measured for BR-022 on openai 2.43.0 and 2.54.0).
            # BR-022 escapes those code points in the tool-result messages
            # the loop builds and added no arm (see `complete()`).
            raise _provider_body_error(
                e.doc,
                model=model,
                type_="NonJsonProviderBody",
                prefix=_NON_JSON_EVENT_PREFIX,
                extra={"phase": "stream"},
            ) from e
        except Exception as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__, "phase": "stream"},
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
        """Serialize a single :class:`ChatMessage` for the request wire.

        For every message EXCEPT an assistant turn carrying native
        ``tool_calls``, this is byte-for-byte ``msg.model_dump(exclude_none=True)``
        (the pre-BR-008 behavior). For an assistant turn carrying
        ``tool_calls`` (only set when native function-calling is on), the
        message is translated into the OpenAI envelope:

            {role:"assistant", content, tool_calls: [
                {id, type:"function",
                 function:{name, arguments: dumps_for_model(args)}}
            ]}

        The ``id`` is sourced PER ENTRY from the entry's own
        :attr:`~fifty_agent_sdk.llm.types.ToolCall.id` (BR-006): the loop
        mints a distinct id per dispatched call so a multi-call assistant
        turn carries N DISTINCT ids, and each subsequent ``role="tool"``
        reply pairs with exactly one entry (a strict OpenAI endpoint returns
        400 otherwise). The fallback
        ``tc.id if tc.id is not None else msg.tool_call_id`` preserves the
        BR-008 single-call wire byte-for-byte: the single-call native path
        sets ``ToolCall(id=None)`` and ``msg.tool_call_id=call_id``, so
        ``tc.id is None`` falls back to ``msg.tool_call_id`` and emits the
        same id BR-008 did. ``arguments`` is emitted as a JSON STRING (not an
        object), per the OpenAI function-calling spec. Since 1.10.2 that
        string keeps non-ASCII text literal rather than as ``\\uXXXX``
        escapes (a value holding a surrogate code point keeps the escaped
        form; :func:`fifty_agent_sdk._model_json.dumps_for_model`, BR-020);
        it decodes to the same value the 1.10.1 string did. Only
        ``arguments`` changed: the ``id`` expression above is untouched.

        Flag-OFF proof: ``ChatMessage.tool_calls`` defaults to ``None`` and is
        only populated on a native turn (which requires
        ``native_tools_enabled=True`` to even reach the provider), so every
        flag-OFF message takes the ``else`` branch — identical to today.
        """
        if msg.role == "assistant" and msg.tool_calls:
            tool_calls_envelope = [
                {
                    "id": tc.id if tc.id is not None else msg.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": dumps_for_model(tc.args),
                    },
                }
                for tc in msg.tool_calls
            ]
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls_envelope,
            }
        return msg.model_dump(exclude_none=True)

    def _max_tokens_key(self, model: str) -> _MaxTokensParam:
        """Resolve the body key for ``max_tokens`` (see ``max_tokens_param``)."""
        if self._max_tokens_param is not None:
            return self._max_tokens_param
        if _MAX_COMPLETION_TOKENS_MODEL_RE.match(model):
            return "max_completion_tokens"
        return "max_tokens"

    def _build_body(self, request: ChatRequest, *, model: str, stream: bool) -> dict[str, Any]:
        """Build the kwargs passed to ``client.chat.completions.create``.

        ``temperature`` is omitted entirely when :attr:`ChatRequest.temperature`
        is ``None`` (the opt-out for providers that reject a non-default
        temperature); at the ``0.0`` default it is sent, preserving the
        pre-existing wire shape for every caller that does not opt out.

        :attr:`ChatRequest.max_tokens` is sent under the key chosen by
        :meth:`_max_tokens_key` — ``max_tokens`` unless the client option or
        the model-name rule selects ``max_completion_tokens``.

        :attr:`ChatRequest.reasoning_effort` (FR-002) is omitted when ``None``,
        so the returned dict gains no key. When set, it is placed under
        ``extra_body``, which the ``openai`` SDK merges into the top level of
        the JSON request, so the provider receives a top-level
        ``reasoning_effort``. It is sent for every model name; there is no
        model-name filter.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [self._serialize_message(m) for m in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body[self._max_tokens_key(model)] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.tools is not None:
            body["tools"] = request.tools
            body["tool_choice"] = request.tool_choice or "auto"
        if request.reasoning_effort is not None:
            # FR-002 D5: `extra_body`, NOT the typed `reasoning_effort=` kwarg
            # of `chat.completions.create`. The declared floor `openai>=1.30.0`
            # predates that kwarg, and generated `create()` methods take no
            # **kwargs, so on an older in-range `openai` the typed kwarg raises
            # a TypeError that the APIError arms in complete()/stream() never
            # wrap into LLMError. `extra_body` exists across the whole declared
            # range and merges at the top level of the JSON body.
            # FR-002 D3: no model-name gate (unlike BR-018's max_tokens key).
            # That rule TRANSLATES a portable field; a gate here would only
            # FILTER out an explicit consumer instruction on a guess from the
            # model name, which is silent when wrong. Sent whenever set; a
            # provider that rejects it fails loudly with LLMError.
            body["extra_body"] = {"reasoning_effort": request.reasoning_effort}
        return body

    @staticmethod
    def _normalize_finish_reason(raw: str | None) -> FinishReason:
        """Map an upstream ``finish_reason`` to our :data:`FinishReason` union.

        Unknown or absent values map to ``"error"`` so consumers can detect
        provider drift without crashing.
        """
        if raw is None:
            return "error"
        return _FINISH_REASON_MAP.get(raw, "error")

    @classmethod
    def _map_response(cls, raw: Any, *, model: str) -> ChatResponse:  # noqa: ANN401 - SDK shape is opaque
        """Map a non-streaming SDK response into our :class:`ChatResponse`.

        Defensive: if any required field is missing or has the wrong shape,
        raise :class:`LLMError` rather than letting a ``KeyError`` /
        ``AttributeError`` leak.

        When the upstream provider emits structured ``tool_calls`` (OpenAI
        function-calling), each entry is normalized into the SDK's
        :class:`~fifty_agent_sdk.llm.types.ToolCall` (``{name, args: dict}``)
        by parsing the provider's JSON-string ``arguments``. Parse failure of
        ``arguments`` raises :class:`LLMError` (``type="MalformedResponse"``)
        with the offending ``tool_call_id`` and a bounded
        ``arguments_excerpt`` — a provider sending malformed ``arguments`` is
        an unrecoverable envelope error, not model drift. So do
        ``arguments`` nested deeper than 64 levels, which are refused
        before decoding (BR-019; see :meth:`_map_tool_calls`).

        A ``str`` ``raw`` is a provider body that could not be read as a
        JSON object: measured on ``openai`` 2.43.0, the client returns a 200
        body as text when its Content-Type is not JSON and it does not
        decode, and returns a bare JSON string as that string. It raises
        :class:`LLMError` with ``context["type"] == "NonJsonProviderBody"``
        (or ``"ContextLengthExceeded"``) and the body's start in the
        message, instead of ``"MalformedResponse"`` (BR-021). A body that
        decodes to a JSON object without ``choices``, such as an error
        envelope ``{"error": {...}}``, is not a ``str`` and still raises
        ``"MalformedResponse"`` without its text (out of BR-021's scope).
        """
        if isinstance(raw, str):
            raise _provider_body_error(
                raw,
                model=model,
                type_="NonJsonProviderBody",
                prefix=_NON_JSON_BODY_PREFIX,
            )
        try:
            choices = raw.choices
            if not choices:
                raise LLMError(
                    "Provider returned no choices.",
                    context={"model": model, "type": "MalformedResponse"},
                )
            choice = choices[0]
            sdk_message = choice.message
            content = sdk_message.content if sdk_message.content is not None else ""
            finish_reason = cls._normalize_finish_reason(choice.finish_reason)
            usage = cls._map_usage(raw.usage)
            tool_calls = cls._map_tool_calls(getattr(sdk_message, "tool_calls", None), model=model)
        except LLMError:
            raise
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError(
                f"Malformed provider response: {e}",
                context={"model": model, "type": "MalformedResponse"},
            ) from e
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def _map_tool_calls(cls, raw: Any, *, model: str) -> list[ToolCall] | None:  # noqa: ANN401 - SDK shape is opaque
        """Normalize upstream OpenAI ``tool_calls`` into SDK :class:`ToolCall`.

        Each upstream entry has the OpenAI shape
        ``{id, type, function: {name, arguments(JSON string)}}``. The SDK
        :class:`ToolCall` is ``{name, args: dict}`` (no ``id`` — see BR-007
        D7: the loop synthesizes the pairing ``call_id`` for history replay).
        The provider's JSON-string ``arguments`` is parsed into ``args``;
        parse failure raises :class:`LLMError` (``type="MalformedResponse"``).

        Returns ``None`` (not ``[]``) when there are no upstream tool calls so
        ``model_dump(exclude_none=True)`` in :meth:`_build_body` stays clean
        on the request wire for every existing caller.

        Args:
            raw: The upstream ``message.tool_calls`` value, or ``None``.
            model: Model identifier for the error context payload.

        Returns:
            A list of SDK :class:`ToolCall` values, or ``None`` when
            ``raw`` is absent/empty.

        Raises:
            fifty_agent_sdk.errors.LLMError: When an entry's ``arguments``
                is not valid JSON or not a JSON object, or (BR-019) nests
                deeper than :data:`~fifty_agent_sdk._json_depth.
                MAX_TOOL_ARGS_DEPTH` levels. The depth check reads
                brackets before ``json.loads`` runs, so invalid text whose
                brackets open more than 64 levels (``x`` followed by 100
                ``[``, which ``json.loads`` rejects at its first character)
                gets the depth error where earlier releases raised the
                invalid-JSON one. All three are
                ``"MalformedResponse"``; only the depth error carries
                ``context["max_tool_args_depth"]``, and it has no
                ``__cause__``. The first such entry in ``raw`` ends the
                mapping, so no :class:`ToolCall` is returned for any entry.
        """
        if not raw:
            return None
        mapped: list[ToolCall] = []
        for tc in raw:
            function = tc.function
            arguments = function.arguments if function.arguments is not None else ""
            # BR-019: check the nesting of the exact text json.loads would
            # decode, before decoding it. Deeply nested arguments used to
            # decode and then make the next request's re-encode raise a raw
            # RecursionError, or raise it here, past the decoder's own limit
            # (BR-019 evidence, P1).
            # The message is fixed and holds no model text; the excerpt goes
            # in `context`, as in the two arms below. A non-`str` value skips
            # the check and keeps its pre-BR-019 path: an empty or false one
            # becomes `{}` without decoding, and any other makes json.loads
            # raise TypeError, which `_map_response` reports as
            # MalformedResponse.
            if (
                isinstance(arguments, str)
                and arguments
                and json_nesting_exceeds(arguments, MAX_TOOL_ARGS_DEPTH)
            ):
                raise LLMError(
                    f"provider tool_call arguments nest deeper than {MAX_TOOL_ARGS_DEPTH} levels",
                    context={
                        "model": model,
                        "type": "MalformedResponse",
                        "tool_call_id": getattr(tc, "id", None),
                        "arguments_excerpt": arguments[:_MAX_TOOL_CALL_ARG_EXCERPT],
                        "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
                    },
                )
            # No RecursionError arm (BR-019): after the check, json.loads sees
            # at most MAX_TOOL_ARGS_DEPTH levels. An arm would also hide the
            # check's removal from part of the regression suite: with the
            # check removed and such an arm added, the four codec-limit loop
            # cases and the past-the-decoder client test passed on CPython
            # 3.11.15 and 3.13.2, where their arguments fail inside
            # json.loads itself; the tests one level past the limit stayed
            # red (BR-019 evidence, mutant M1a). Not measured: a host whose
            # stack is nearly exhausted when this decodes up to 64 levels.
            try:
                args = json.loads(arguments) if arguments else {}
            except ValueError as e:
                # json.loads also raises bare ValueError for interpreter
                # limits such as oversized integers. Provider payloads must
                # remain contained as LLMError without parsing exception text.
                raise LLMError(
                    "provider tool_call arguments is not valid JSON",
                    context={
                        "model": model,
                        "type": "MalformedResponse",
                        "tool_call_id": getattr(tc, "id", None),
                        "arguments_excerpt": arguments[:_MAX_TOOL_CALL_ARG_EXCERPT],
                    },
                ) from e
            if not isinstance(args, dict):
                # The OpenAI spec requires `arguments` to be a JSON object;
                # a non-object (e.g. a bare string or number) is malformed.
                raise LLMError(
                    "provider tool_call arguments is not a JSON object",
                    context={
                        "model": model,
                        "type": "MalformedResponse",
                        "tool_call_id": getattr(tc, "id", None),
                        "arguments_excerpt": arguments[:_MAX_TOOL_CALL_ARG_EXCERPT],
                    },
                )
            mapped.append(ToolCall(name=function.name, args=args))
        return mapped

    @classmethod
    def _map_chunk(cls, raw: Any, *, model: str) -> ChatResponse:  # noqa: ANN401 - SDK shape is opaque
        """Map a streaming SDK chunk into a delta-carrying :class:`ChatResponse`.

        ``message.content`` is the chunk's delta. ``finish_reason`` is
        ``"in_progress"`` on intermediate chunks (and on header-only chunks
        with no ``choices``) and only resolves to a terminal value
        (``"stop"``/``"length"``/``"tool_calls"``/``"content_filter"``/``"error"``)
        on the chunk whose upstream ``finish_reason`` is non-``None``. This
        lets consumers branch on ``finish_reason`` without misreading an
        intermediate delta as terminal.

        A ``str`` ``raw`` is a stream event whose data is a bare JSON string.
        It raises :class:`LLMError` with ``context["type"] ==
        "NonJsonProviderBody"`` (or ``"ContextLengthExceeded"``),
        ``context["phase"] == "stream"`` and the text in the message,
        instead of ``"MalformedChunk"`` (BR-021).
        """
        if isinstance(raw, str):
            raise _provider_body_error(
                raw,
                model=model,
                type_="NonJsonProviderBody",
                prefix=_NON_JSON_EVENT_PREFIX,
                extra={"phase": "stream"},
            )
        try:
            choices = raw.choices
            if not choices:
                # Some providers emit a header chunk with only usage and no choices.
                usage = cls._map_usage(getattr(raw, "usage", None))
                return ChatResponse(
                    message=ChatMessage(role="assistant", content=""),
                    usage=usage,
                    finish_reason="in_progress",
                )
            choice = choices[0]
            delta = choice.delta
            content = ""
            # NOTE: native tool_calls are intentionally NOT mapped here. A
            # native tool-decision turn is non-streamed (BR-007 D5), and the
            # loop's native precedence branch only fires when a real
            # ChatResponse carries tool_calls — streamed turns never go native.
            if delta is not None and delta.content is not None:
                content = delta.content
            upstream_finish = choice.finish_reason
            finish_reason: FinishReason
            if upstream_finish is None:
                finish_reason = "in_progress"
            else:
                finish_reason = cls._normalize_finish_reason(upstream_finish)
            usage = cls._map_usage(getattr(raw, "usage", None))
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError(
                f"Malformed provider stream chunk: {e}",
                context={"model": model, "type": "MalformedChunk"},
            ) from e
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            usage=usage,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _map_usage(raw: Any) -> Usage:  # noqa: ANN401 - SDK shape is opaque
        """Map an SDK usage object into our :class:`Usage`.

        Missing or absent usage data resolves to zeros — providers vary in
        whether they emit usage on streaming chunks.
        """
        if raw is None:
            return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
        completion_tokens = getattr(raw, "completion_tokens", 0) or 0
        total_tokens = getattr(raw, "total_tokens", 0) or 0
        return Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )


__all__ = ["OpenAICompatibleClient"]
