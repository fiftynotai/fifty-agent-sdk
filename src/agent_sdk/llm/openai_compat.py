"""OpenAI-compatible :class:`LLMClient` adapter.

Backed by the official ``openai`` Python SDK pointed at any OpenAI-compatible
``/v1/chat/completions`` endpoint — OpenAI itself, Google Distributed Cloud
(GDC), local OSS servers (vLLM, Ollama via the openai-compat layer, etc.).
The provider differences are absorbed by ``base_url``.

All provider SDK exceptions are wrapped into :class:`agent_sdk.errors.LLMError`
at the public method boundary, in line with the
:class:`agent_sdk.llm.protocol.LLMClient` contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from agent_sdk.errors import LLMError
from agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
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


class OpenAICompatibleClient:
    """:class:`LLMClient` implementation backed by the ``openai`` Python SDK.

    Works against any OpenAI-compatible ``/v1/chat/completions`` endpoint.
    Provider variation is absorbed by ``base_url`` — the same client class
    drives OpenAI itself, GDC, and local OSS servers.

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
            SDK builds its own client.
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
    ) -> None:
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

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a single non-streaming completion.

        Args:
            request: The chat completion request.

        Returns:
            The mapped :class:`ChatResponse`.

        Raises:
            agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call (network, timeout, rate-limit, malformed
                envelope, missing fields).
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
                context={"model": model, "type": type(e).__name__},
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

        Args:
            request: The chat completion request.

        Yields:
            :class:`ChatResponse` chunks containing the latest delta.

        Raises:
            agent_sdk.errors.LLMError: Wraps any failure of the underlying
                provider call. Errors mid-stream surface from the iterator.
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
                context={"model": model, "type": type(e).__name__},
            ) from e

        try:
            async for raw_chunk in stream:
                yield self._map_chunk(raw_chunk, model=model)
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
                context={"model": model, "type": type(e).__name__, "phase": "stream"},
            ) from e
        except LLMError:
            # Allow defensive mapping errors raised from `_map_chunk` to surface unchanged.
            raise
        except Exception as e:
            raise LLMError(
                str(e),
                context={"model": model, "type": type(e).__name__, "phase": "stream"},
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_body(request: ChatRequest, *, model: str, stream: bool) -> dict[str, Any]:
        """Build the kwargs passed to ``client.chat.completions.create``."""
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = request.response_format
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
        """
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
        except LLMError:
            raise
        except (AttributeError, IndexError, TypeError) as e:
            raise LLMError(
                f"Malformed provider response: {e}",
                context={"model": model, "type": "MalformedResponse"},
            ) from e
        return ChatResponse(
            message=ChatMessage(role="assistant", content=content),
            usage=usage,
            finish_reason=finish_reason,
        )

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
        """
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
