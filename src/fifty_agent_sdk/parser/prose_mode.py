"""Tolerant ReACT prose parser.

Consumes the classic ``Thought / Action / Action Input`` (or
``Thought / Final Answer``) format taught by
:data:`fifty_agent_sdk.prompts.PROSE_MODE_OUTPUT_FORMAT`. Tolerant of whitespace
and header capitalization. Strict in one specific way: if neither pattern
matches the completion as a whole, the parser raises
:class:`fifty_agent_sdk.errors.ParserError` rather than guessing.

Tie-break: when the completion contains BOTH ``Action:`` and ``Final Answer:``
headers, the tool-call path wins. Rationale: the loop terminates only on a
:class:`fifty_agent_sdk.parser.base.FinalAnswer`, so mis-firing a stale tool call
is strictly recoverable (the next iteration will re-parse), while
prematurely terminating on a stray ``Final Answer:`` is not.
"""

from __future__ import annotations

import json
import re
from typing import Final

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds
from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ToolCall
from fifty_agent_sdk.parser.base import FinalAnswer, ParseResult, ThoughtAction
from fifty_agent_sdk.parser.json_mode import _strip_code_fences

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""


def _depth_error(completion: str) -> ParserError:
    """Build the fixed-message :class:`ParserError` for a too-deep ``Action Input`` (BR-019)."""
    return ParserError(
        f"could not decode Action Input JSON: nesting deeper than {MAX_TOOL_ARGS_DEPTH} levels",
        context={
            "parser": "ProseModeParser",
            "error_phase": "action_input_decode",
            "completion_excerpt": completion[:_MAX_EXCERPT],
            "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
        },
    )


_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*"
    r"Action:\s*(?P<tool>[^\n]+?)\s*\n\s*"
    r"Action\s+Input:\s*(?P<args>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""Matches the tool-call form: ``Thought: ... Action: ... Action Input: ...``.

* Non-greedy quantifiers on ``thought``/``args`` so they don't swallow
  subsequent headers or trailing whitespace.
* ``[^\\n]+?`` on the tool name keeps it single-line.
* Anchored at both ends; ``re.DOTALL`` lets the bodies span newlines.
* Compiled once at module scope to avoid per-call cost and ReDoS surface.
"""

_FINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*Thought:\s*(?P<thought>.*?)\s*"
    r"Final\s+Answer:\s*(?P<answer>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
"""Matches the final-answer form: ``Thought: ... Final Answer: ...``."""


class ProseModeParser:
    """Classic ReACT format parser; tolerant of whitespace and case variants.

    The parser attempts :data:`_TOOL_RE` first and only falls through to
    :data:`_FINAL_RE` when no tool form matches. See the module docstring
    for the rationale behind the tie-break.

    Failure surfaces:

    * ``error_phase="empty_completion"`` — empty/whitespace-only input.
    * ``error_phase="header_match"`` — neither pattern matched.
    * ``error_phase="action_input_decode"`` — the ``Action Input:`` body
      could not be parsed as JSON, even after the shared fence-stripping
      recovery pass, or the text left to decode nests deeper than 64 levels
      (:data:`fifty_agent_sdk._json_depth.MAX_TOOL_ARGS_DEPTH`, BR-019).
      Such text never reaches ``json.loads``: a too-deep body goes to the
      recovery pass, and a too-deep recovery candidate, or no candidate
      after a too-deep body, raises with ``context["max_tool_args_depth"]``
      (see :meth:`_decode_action_input`). A valid body that is too deep
      goes to that pass too, which earlier releases ran only after a decode
      failure, so it can now parse to a ``{``...``}`` candidate inside it.
      The check reads brackets rather than JSON, so invalid text whose
      brackets open more than 64 levels gets this error where earlier
      releases reported the invalid JSON when the recovery pass finds no
      candidate or the check also refuses the candidate (``x`` followed by
      100 ``[``, for example); a candidate within the limit is decoded as
      before. ``json.loads`` itself raises
      :class:`RecursionError` at a depth that depends on the CPython version
      and on the caller's stack, not on :func:`sys.getrecursionlimit` alone
      (the measured depths are in
      :class:`~fifty_agent_sdk.parser.json_mode.JsonModeParser`'s
      docstring); that error is still translated into :class:`ParserError`
      (BR-017) so the loop's ``ParserError`` contract holds.
    """

    def parse(self, completion: str) -> ParseResult:
        """See :meth:`fifty_agent_sdk.parser.base.Parser.parse`."""
        if not completion or not completion.strip():
            raise ParserError(
                "completion is empty",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "empty_completion",
                    "completion_excerpt": "",
                },
            )

        tool_match = _TOOL_RE.match(completion)
        if tool_match is not None:
            return self._parse_tool(tool_match, completion)

        final_match = _FINAL_RE.match(completion)
        if final_match is not None:
            return self._parse_final(final_match)

        raise ParserError(
            "no recognizable Thought/Action/Final Answer structure",
            context={
                "parser": "ProseModeParser",
                "error_phase": "header_match",
                "completion_excerpt": completion[:_MAX_EXCERPT],
            },
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _parse_tool(self, match: re.Match[str], completion: str) -> ThoughtAction:
        """Build a :class:`ThoughtAction` from a tool-form match."""
        thought = match.group("thought").strip()
        tool_name = match.group("tool").strip()
        args_body = match.group("args").strip()
        args = self._decode_action_input(args_body, completion)
        return ThoughtAction(
            thought=thought,
            tool_call=ToolCall(name=tool_name, args=args),
        )

    def _parse_final(self, match: re.Match[str]) -> FinalAnswer:
        """Build a :class:`FinalAnswer` from a final-form match."""
        thought = match.group("thought").strip()
        answer = match.group("answer").strip()
        return FinalAnswer(thought=thought, content=answer)

    def _decode_action_input(self, body: str, completion: str) -> dict[str, object]:
        """Decode the ``Action Input:`` body as JSON, with one fence retry.

        Before each ``json.loads``, the text it would decode is checked for
        nesting deeper than
        :data:`fifty_agent_sdk._json_depth.MAX_TOOL_ARGS_DEPTH` (64) levels
        (BR-019). A body that is too deep is not decoded and goes to the
        fence recovery, as a decode failure does. A valid body that is too
        deep takes that pass too; before BR-019 it was passed to
        ``json.loads`` as it stood (for example ``[{"a": 1}, <70-level
        array>]`` raised "Action Input JSON must decode to an object", and
        now parses to ``{"a": 1}``). A recovery candidate that
        is too deep, or none at all after a too-deep body, raises
        :class:`ParserError` (``error_phase="action_input_decode"``) with
        ``context["max_tool_args_depth"]`` and no ``__cause__``.

        :class:`RecursionError` from ``json.loads`` is still caught at BOTH
        decode attempts and translated into :class:`ParserError`
        (``error_phase="action_input_decode"``), as BR-017 added it — the
        same ``ParserError``-only contract the JSON-mode parser upholds on its
        own decode path.
        """
        if json_nesting_exceeds(body, MAX_TOOL_ARGS_DEPTH):
            decoded = self._recover_action_input(body, completion, None)
        else:
            try:
                decoded = json.loads(body)
            except RecursionError as depth_err:
                raise ParserError(
                    "could not decode Action Input JSON: nesting depth exceeds the recursion limit",
                    context={
                        "parser": "ProseModeParser",
                        "error_phase": "action_input_decode",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "cause": repr(depth_err),
                    },
                ) from depth_err
            except ValueError as first_err:
                # json.loads may raise bare ValueError for interpreter limits
                # (for example oversized integers). The public contract
                # classifies every decode failure identically without
                # inspecting error text. The recovery runs inside this
                # handler, as before BR-019, so implicit exception chaining
                # is unchanged (pinned by test_failed_decodes_keep_the_strict_
                # error_in_their_exception_chain).
                decoded = self._recover_action_input(body, completion, first_err)

        if not isinstance(decoded, dict):
            raise ParserError(
                "Action Input JSON must decode to an object",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "decoded_type": type(decoded).__name__,
                },
            )
        return decoded

    def _recover_action_input(
        self, body: str, completion: str, first_err: ValueError | None
    ) -> object:
        """Run the one fence/brace recovery pass of :meth:`_decode_action_input`.

        Args:
            body: The ``Action Input:`` body.
            completion: The raw completion, for the error excerpt.
            first_err: The strict pass's decode error, or ``None`` when the
                strict pass was skipped because the body nests too deeply
                (BR-019).

        Returns:
            The decoded recovery candidate (any JSON value; the caller checks
            that it is an object).

        Raises:
            ParserError: ``error_phase="action_input_decode"`` when no
                candidate exists, the candidate nests too deeply, or it does
                not decode.
        """
        recovered = _strip_code_fences(body)
        if recovered is None:
            if first_err is None:
                raise _depth_error(completion)
            raise ParserError(
                "could not decode Action Input JSON",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(first_err),
                },
            ) from first_err
        if json_nesting_exceeds(recovered, MAX_TOOL_ARGS_DEPTH):
            raise _depth_error(completion)
        try:
            return json.loads(recovered)
        except RecursionError as depth_err:
            raise ParserError(
                "could not decode Action Input JSON: nesting depth exceeds the recursion limit",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(depth_err),
                },
            ) from depth_err
        except ValueError as second_err:
            raise ParserError(
                "could not decode Action Input JSON after fence recovery",
                context={
                    "parser": "ProseModeParser",
                    "error_phase": "action_input_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(second_err),
                },
            ) from second_err


__all__ = ["ProseModeParser"]
