"""Provider-native function-calling parser.

Most providers (OpenAI, Anthropic, etc.) can emit tool invocations as a
structured field on the chat response rather than embedding them in the
assistant's text content. The SDK's text-mode parsers
(:class:`fifty_agent_sdk.parser.json_mode.JsonModeParser`,
:class:`fifty_agent_sdk.parser.prose_mode.ProseModeParser`) consume the
*completion string*; a native-tools parser consumes the *structured response
object* instead.

Integration point
-----------------
When the LLM adapter layer populates ``tool_calls`` on
:class:`fifty_agent_sdk.llm.types.ChatMessage`, the ReACT runner checks
``ChatResponse.message.tool_calls`` FIRST and dispatches through the concrete
:class:`NativeToolsParser` when present, bypassing text parsing entirely.
When ``tool_calls`` is empty/absent the unchanged text path runs (precedence,
not replacement — see BR-007).

The :class:`NativeToolsParserProtocol` is kept distinct from
:class:`fifty_agent_sdk.parser.base.Parser` because the input shape differs:
``Parser`` takes ``completion: str`` and the native parser takes
``response: ChatResponse``. Keeping the signatures narrow avoids forcing
every text-mode parser to accept a wider union input. The concrete
:class:`NativeToolsParser` satisfies the Protocol and is what the loop
instantiates; the Protocol exists so test fakes can be plugged in
structurally.

Scope note (BR-007): a native response that carries MULTIPLE tool calls is
parsed to its FIRST call only; the remainder are dropped until BR-006
(parallel/concurrent dispatch) lands. The truncation is logged at DEBUG
(``native_tool_calls_truncated`` with ``count=N``) so a multi-call response is
observable, not silent. Multi-call behavior is net-new (today's text path is
strictly single-call), so this is not a regression.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import structlog

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ChatResponse
from fifty_agent_sdk.parser.base import ParseResult, ThoughtAction

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger. DEBUG for the multi-call truncation signal."""


@runtime_checkable
class NativeToolsParserProtocol(Protocol):
    """Protocol for provider-native tool-call parsing.

    Distinct from :class:`fifty_agent_sdk.parser.base.Parser` because the input
    is a structured :class:`fifty_agent_sdk.llm.types.ChatResponse`, not a
    completion string. Implementations consume the response's native
    ``tool_calls`` field (when populated by the adapter) and return a
    :data:`~fifty_agent_sdk.parser.base.ParseResult`.

    Marked :func:`runtime_checkable` so callers can plug in test fakes
    structurally. The concrete :class:`NativeToolsParser` (below) satisfies
    this Protocol; the loop instantiates the concrete class directly.
    """

    def parse(self, response: ChatResponse) -> ParseResult:
        """Convert a provider-native tool-call response into a parse result.

        Args:
            response: The chat response carrying native ``tool_calls``.

        Returns:
            A :data:`~fifty_agent_sdk.parser.base.ParseResult`.

        Raises:
            fifty_agent_sdk.errors.ParserError: When the response cannot be
                interpreted as a valid tool-call envelope.
        """
        ...


class NativeToolsParser:
    """Concrete parser for provider-native tool-call responses.

    Consumes the structured ``tool_calls`` field on
    :attr:`ChatResponse.message` (populated by the LLM adapter from the
    upstream provider's native function-calling envelope) and returns a
    :data:`ParseResult` in the SAME shape the text-mode parsers produce —
    a single :class:`ThoughtAction` carrying one
    :class:`fifty_agent_sdk.llm.types.ToolCall`.

    This keeps the loop's existing dispatch block reusable verbatim: whether a
    tool call originated from text parsing or native parsing,
    ``parsed.tool_call.name`` / ``parsed.tool_call.args`` have identical shape
    and the downstream event sequence is byte-for-byte the same. The class
    satisfies :class:`NativeToolsParserProtocol`.

    Failure surfaces (mirror the text-parser context schema —
    ``parser`` / ``error_phase`` / ``completion_excerpt``):

    * ``error_phase="empty_tool_calls"`` — the response carried no native
      ``tool_calls`` (``None`` or empty list). The loop's precedence branch
      SHOULD NOT route such a response here; a direct caller hit is still a
      loud, recoverable signal rather than a silent no-op.
    * ``error_phase="schema_validation"`` — a ``tool_calls`` entry was
      malformed (missing/empty ``name``, or non-dict ``args``). Defensive
      against a hand-built response, since the SDK's own
      :class:`~fifty_agent_sdk.llm.types.ToolCall` schema would normally
      reject these at construction.

    A malformed native array raises :class:`ParserError` and terminates; the
    native path does NOT participate in the BR-018 one-shot text-retry loop
    (a structured ``tool_calls`` array comes from the provider SDK already
    formed — a malformed one is an adapter/provider error, not recoverable
    model drift).

    Scope (BR-007): a response carrying more than one native tool call yields
    ONE :class:`ThoughtAction` from the FIRST call; the remaining calls are
    dropped and the truncation is logged at DEBUG
    (``native_tool_calls_truncated``, ``count=N``). Multi-call dispatch is
    BR-006's job.
    """

    def parse(self, response: ChatResponse) -> ParseResult:
        """Convert a provider-native tool-call response into a parse result.

        Args:
            response: The chat response carrying native ``tool_calls``.

        Returns:
            A :class:`ThoughtAction` built from the first native tool call.

        Raises:
            fifty_agent_sdk.errors.ParserError: When ``tool_calls`` is empty
                or an entry fails schema validation.
        """
        calls = response.message.tool_calls or []
        if not calls:
            raise ParserError(
                "response carried no native tool_calls",
                context={
                    "parser": "NativeToolsParser",
                    "error_phase": "empty_tool_calls",
                    "completion_excerpt": (response.message.content or "")[:_MAX_EXCERPT],
                },
            )
        if len(calls) > 1:
            # BR-007 scope: take the first call, drop the rest. Multi-call
            # dispatch is BR-006. Logged at DEBUG so the truncation is
            # observable, not silent.
            _log.debug(
                "native_tool_calls_truncated",
                count=len(calls),
            )
        first = calls[0]
        if not first.name or not isinstance(first.args, dict):
            raise ParserError(
                "native tool_call entry failed schema validation",
                context={
                    "parser": "NativeToolsParser",
                    "error_phase": "schema_validation",
                    "completion_excerpt": (response.message.content or "")[:_MAX_EXCERPT],
                },
            )
        return ThoughtAction(thought="", tool_call=first)


__all__ = ["NativeToolsParser", "NativeToolsParserProtocol"]
