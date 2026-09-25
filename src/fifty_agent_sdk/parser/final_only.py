"""Final-only text parser backing ``AgentLoop(tool_mode=ToolMode.NATIVE)`` (FR-001).

In the OpenAI-compatible native protocol, the choice between a tool call and
a final answer is STRUCTURAL: a tool call arrives in the response's
``tool_calls`` field (handled by
:class:`fifty_agent_sdk.parser.native_tools.NativeToolsParser`), and anything
else is the final answer, sent as plain ``content``. This parser handles the
second half. Every non-blank completion string becomes a
:class:`~fifty_agent_sdk.parser.base.FinalAnswer` whose ``content`` is the
completion **verbatim** (not stripped), so ``FinalEvent.text`` equals
``FinalEvent.raw_completion`` and matches what a runner persists.

Invariant (FR-001 AC-2): this parser NEVER returns a
:class:`~fifty_agent_sdk.parser.base.ThoughtAction` or
:class:`~fifty_agent_sdk.parser.base.MultiAction`. A completion that *looks*
like a text tool call — a JSON ``{"action": "tool", ...}`` envelope or a prose
``Action:`` block — is still a final answer. Under NATIVE mode the text path
therefore cannot dispatch a tool, so no ``role="tool"`` reply can ever follow
an assistant turn that carries no ``tool_calls``. The guarantee holds by
construction, not by filtering.

Failure surface (the text-parser context schema: ``parser`` / ``error_phase``
/ ``completion_excerpt``):

* ``error_phase="empty_completion"`` — empty or whitespace-only input. The
  loop feeds this into the existing BR-018 one-shot parser retry with the
  NATIVE reminder; a spent or disabled retry ends the run with the usual
  ``ErrorEvent`` + fallback ``FinalEvent``.

The class is private (``_``-prefixed, not re-exported). Native mode is
selected with ``AgentLoop(tool_mode=ToolMode.NATIVE)``, never by passing this
parser by hand. Exporting it would invite a second, half-configured way to
build a native loop, which is exactly the inconsistency FR-001 removes.
"""

from __future__ import annotations

from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser.base import FinalAnswer, ParseResult


class _FinalOnlyParser:
    """Text parser whose only non-error output is :class:`FinalAnswer`.

    Satisfies :class:`fifty_agent_sdk.parser.base.Parser` structurally. See the
    module docstring for the AC-2 invariant and the failure surface.
    """

    def parse(self, completion: str) -> ParseResult:
        """Return the completion, verbatim, as a final answer.

        Args:
            completion: The full assistant message content of a response that
                carried no native ``tool_calls``.

        Returns:
            ``FinalAnswer(thought="", content=completion)``, with the content
            unmodified.

        Raises:
            fifty_agent_sdk.errors.ParserError: With
                ``error_phase="empty_completion"`` when ``completion`` is empty
                or whitespace-only.
        """
        if not completion or not completion.strip():
            raise ParserError(
                "completion is empty",
                context={
                    "parser": "FinalOnlyParser",
                    "error_phase": "empty_completion",
                    "completion_excerpt": "",
                },
            )
        return FinalAnswer(thought="", content=completion)


__all__: list[str] = []
