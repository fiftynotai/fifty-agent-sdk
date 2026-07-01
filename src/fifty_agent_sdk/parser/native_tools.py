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

Scope note (BR-006): a native response carrying MULTIPLE tool calls is parsed
into a :class:`~fifty_agent_sdk.parser.base.MultiAction` carrying the FULL
list; the loop dispatches them concurrently under a bounded gather. A
single-call response yields a :class:`~fifty_agent_sdk.parser.base.
ThoughtAction` byte-identical to the pre-BR-006 path (and to a text-parsed
call). Multi-call dispatch is net-new (today's text path is strictly
single-call), so this is not a regression.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ChatResponse
from fifty_agent_sdk.parser.base import MultiAction, ParseResult, ThoughtAction

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""


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
    :data:`ParseResult`:

    * ONE call → :class:`~fifty_agent_sdk.parser.base.ThoughtAction`
      carrying the single call — byte-identical to the pre-BR-006 path and
      to a text-parsed call, so the loop's single-call dispatch block runs
      verbatim.
    * MORE THAN ONE call → :class:`~fifty_agent_sdk.parser.base.MultiAction`
      carrying the FULL list (in CALL order), which the loop dispatches
      concurrently under a bounded gather (BR-006). The truncation the
      pre-BR-006 parser applied to multi-call responses is removed.

    The class satisfies :class:`NativeToolsParserProtocol`.

    Failure surfaces (mirror the text-parser context schema —
    ``parser`` / ``error_phase`` / ``completion_excerpt``):

    * ``error_phase="empty_tool_calls"`` — the response carried no native
      ``tool_calls`` (``None`` or empty list). The loop's precedence branch
      SHOULD NOT route such a response here; a direct caller hit is still a
      loud, recoverable signal rather than a silent no-op.
    * ``error_phase="schema_validation"`` — a ``tool_calls`` entry was
      malformed (missing/empty ``name``, or non-dict ``args``). EVERY entry
      is validated (not just the first), since a multi-call response will
      all be dispatched. Defensive against a hand-built response, since the
      SDK's own :class:`~fifty_agent_sdk.llm.types.ToolCall` schema would
      normally reject these at construction.

    A malformed native array raises :class:`ParserError` and terminates; the
    native path does NOT participate in the BR-018 one-shot text-retry loop
    (a structured ``tool_calls`` array comes from the provider SDK already
    formed — a malformed one is an adapter/provider error, not recoverable
    model drift).
    """

    def parse(self, response: ChatResponse) -> ParseResult:
        """Convert a provider-native tool-call response into a parse result.

        A response carrying exactly ONE ``tool_calls`` entry yields a
        :class:`~fifty_agent_sdk.parser.base.ThoughtAction` (byte-identical to
        the pre-BR-006 single-call path). A response carrying MORE THAN ONE
        entry yields a :class:`~fifty_agent_sdk.parser.base.MultiAction`
        carrying the full list in CALL order; the loop dispatches them
        concurrently (BR-006).

        Args:
            response: The chat response carrying native ``tool_calls``.

        Returns:
            A :class:`~fifty_agent_sdk.parser.base.ThoughtAction` when the
            response carries a single call, otherwise a
            :class:`~fifty_agent_sdk.parser.base.MultiAction`.

        Raises:
            fifty_agent_sdk.errors.ParserError: When ``tool_calls`` is empty
                or ANY entry fails schema validation.
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
        # Validate EVERY entry's schema (not just the first): a multi-call
        # response will be dispatched in full under BR-006, so a malformed
        # entry in any position must be caught here rather than mid-dispatch.
        for entry in calls:
            if not entry.name or not isinstance(entry.args, dict):
                raise ParserError(
                    "native tool_call entry failed schema validation",
                    context={
                        "parser": "NativeToolsParser",
                        "error_phase": "schema_validation",
                        "completion_excerpt": (response.message.content or "")[:_MAX_EXCERPT],
                    },
                )
        if len(calls) == 1:
            # Single call: byte-identical to the pre-BR-006 path. The loop's
            # unchanged single-call ThoughtAction dispatch block runs verbatim.
            return ThoughtAction(thought="", tool_call=calls[0])
        # Multi-call: carry the full list. The loop's MultiAction branch
        # dispatches them concurrently under a bounded gather.
        return MultiAction(thought="", tool_calls=list(calls))


__all__ = ["NativeToolsParser", "NativeToolsParserProtocol"]
