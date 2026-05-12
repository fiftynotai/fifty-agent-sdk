"""The :class:`LLMClient` protocol — the SDK's LLM contract.

Anything that quacks like an LLM client can be plugged into the agent loop:
the real OpenAI-compatible adapter, a fake for tests, a local OSS server, a
record-and-replay shim. The structural typing means no inheritance is
required — implement the methods with matching signatures and you satisfy
the protocol.

Implementations MUST raise :class:`agent_sdk.errors.LLMError` (with
structured context) on any failure: network errors, HTTP non-2xx, malformed
provider envelopes, timeouts, or missing required fields. Provider SDK
exceptions MUST be wrapped at the adapter boundary so they never leak into
caller code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agent_sdk.llm.types import ChatRequest, ChatResponse


@runtime_checkable
class LLMClient(Protocol):
    """Provider-agnostic chat-completion client.

    The contract is intentionally narrow: a non-streaming :meth:`complete` and
    a streaming :meth:`stream`. Both accept a :class:`ChatRequest` and yield
    :class:`ChatResponse` instances.

    Error contract:
        Implementations MUST raise :class:`agent_sdk.errors.LLMError` (or a
        subclass) on any failure. The error's ``context`` should include at
        least the model name and the underlying exception type for triage.
        Never let provider-SDK-specific exceptions escape these methods.
    """

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a single non-streaming chat completion.

        Args:
            request: The provider-agnostic chat request.

        Returns:
            A :class:`ChatResponse` containing the assistant's message,
            token-usage figures, and finish reason.

        Raises:
            agent_sdk.errors.LLMError: If the provider call fails for any
                reason. The error wraps the underlying cause via ``raise ...
                from`` and carries structured context.
        """
        ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream a chat completion as incremental chunks.

        Each yielded :class:`ChatResponse` carries a delta in
        ``message.content`` (not the accumulated content). The final chunk
        MUST have ``finish_reason`` populated to one of the
        :data:`agent_sdk.llm.types.FinishReason` literals. ``usage`` may be
        zero on intermediate chunks and is only guaranteed on the final
        chunk for providers that emit it.

        This method is declared as a sync ``def`` returning an
        :class:`~collections.abc.AsyncIterator` so consumers can do
        ``async for chunk in client.stream(request)`` without awaiting first.
        Concrete implementations typically use ``async def`` with ``yield``
        (an async generator), which is itself an :class:`AsyncIterator`.

        Args:
            request: The provider-agnostic chat request.

        Returns:
            An async iterator of :class:`ChatResponse` chunks.

        Raises:
            agent_sdk.errors.LLMError: If the stream fails to open or fails
                mid-iteration. Errors — both during the underlying connection
                open and mid-iteration — surface from ``__anext__`` because
                implementations are typically async generators.
        """
        ...


__all__ = ["LLMClient"]
