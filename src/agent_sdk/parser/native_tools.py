"""Stub for native function-calling parsers.

Most providers (OpenAI, Anthropic, etc.) can emit tool invocations as a
structured field on the chat response rather than embedding them in the
assistant's text content. The SDK's text-mode parsers (:class:`JsonModeParser`,
:class:`ProseModeParser`) consume the *completion string*; a native-tools
parser consumes the *structured response object* instead.

Integration point
-----------------
When the LLM adapter layer adds a ``tool_calls`` field to
:class:`agent_sdk.llm.types.ChatMessage` (deferred to a later brief — the
field does not yet exist on the SDK's provider-agnostic message shape), the
ReACT runner will check ``ChatResponse.message.tool_calls`` first and dispatch
through :class:`NativeToolsParser` when present, bypassing text parsing
entirely. Until that field is added, :class:`NativeToolsParserStub` raises
:class:`NotImplementedError` so callers see a loud failure rather than a
silent no-op.

The Protocol is kept distinct from :class:`agent_sdk.parser.base.Parser`
because the input shape differs: :class:`Parser` takes ``completion: str`` and
:class:`NativeToolsParser` takes ``response: ChatResponse``. Keeping the
signatures narrow avoids forcing every text-mode parser to accept a wider
union input.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_sdk.llm.types import ChatResponse
from agent_sdk.parser.base import ParseResult


@runtime_checkable
class NativeToolsParser(Protocol):
    """Reserved Protocol for provider-native tool-call parsing.

    Distinct from :class:`agent_sdk.parser.base.Parser` because the input is
    a structured :class:`agent_sdk.llm.types.ChatResponse`, not a completion
    string. Implementations consume the response's native ``tool_calls``
    field (when populated by the adapter) and return a
    :data:`~agent_sdk.parser.base.ParseResult`.

    Marked :func:`runtime_checkable` so callers can plug in test fakes
    structurally; the actual concrete implementation will land in a future
    brief alongside the :class:`agent_sdk.llm.types.ChatMessage` schema
    extension.
    """

    def parse(self, response: ChatResponse) -> ParseResult:
        """Convert a provider-native tool-call response into a parse result.

        Args:
            response: The chat response carrying native ``tool_calls``.

        Returns:
            A :data:`~agent_sdk.parser.base.ParseResult`.

        Raises:
            agent_sdk.errors.ParserError: When the response cannot be
                interpreted as a valid tool-call envelope.
        """
        ...


class NativeToolsParserStub:
    """Concrete no-op stub for :class:`NativeToolsParser`.

    Calling :meth:`parse` raises :class:`NotImplementedError`. Exists so the
    public surface is non-empty and so future code can be written against
    the eventual contract before the underlying support lands.
    """

    def parse(self, response: ChatResponse) -> ParseResult:
        """Always raises :class:`NotImplementedError`.

        Args:
            response: Ignored; the stub does not inspect the response.

        Raises:
            NotImplementedError: Always. See module docstring for the
                planned integration point.
        """
        raise NotImplementedError(
            "native function calling not yet supported by any configured "
            "provider — use JsonModeParser or ProseModeParser instead. See "
            "agent_sdk.parser.native_tools module docstring for the planned "
            "integration point."
        )


__all__ = ["NativeToolsParser", "NativeToolsParserStub"]
