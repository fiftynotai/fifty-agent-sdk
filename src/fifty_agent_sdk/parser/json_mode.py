"""Default JSON-mode parser.

Consumes the JSON envelope shape locked in
:data:`fifty_agent_sdk.prompts.JSON_MODE_OUTPUT_FORMAT`:

.. code-block:: json

    {
      "thought":   string,
      "action":    "tool" | "final",
      "tool_name": string | null,
      "tool_args": object | null,
      "answer":    string | null
    }

Parsing is strict: an unknown top-level key or wrong ``action`` value raises
:class:`fifty_agent_sdk.errors.ParserError` with ``error_phase="schema_validation"``.

The parser performs **exactly one** syntactic recovery pass when
``json.loads`` fails on the raw input, or (BR-019) when the raw input nests
too deeply to be decoded — it strips Markdown code fences (` ``` `
or ` ```json `) and/or slices between the first ``{`` and the last ``}``, then
re-attempts decoding. Schema-validation failures are *not* retried.

The :func:`_strip_code_fences` helper is shared with
:mod:`fifty_agent_sdk.parser.prose_mode` so the prose parser can recover JSON in
``Action Input:`` bodies the same way.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds
from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.llm.types import ToolCall
from fifty_agent_sdk.parser.base import FinalAnswer, ParseResult, ThoughtAction

_MAX_EXCERPT: Final[int] = 200
"""Maximum length of ``completion_excerpt`` in :class:`ParserError` context."""

_MAX_ENVELOPE_DEPTH: Final[int] = MAX_TOOL_ARGS_DEPTH + 1
"""Deepest envelope text :class:`JsonModeParser` decodes (BR-019).

The envelope object is one level above ``tool_args``, so this lets
``tool_args`` nest :data:`fifty_agent_sdk._json_depth.MAX_TOOL_ARGS_DEPTH`
levels.
"""


def _depth_error(completion: str) -> ParserError:
    """Build the fixed-message :class:`ParserError` for a too-deep envelope (BR-019)."""
    return ParserError(
        "could not decode JSON envelope: nesting deeper than "
        f"{_MAX_ENVELOPE_DEPTH} levels (tool_args may nest at most {MAX_TOOL_ARGS_DEPTH})",
        context={
            "parser": "JsonModeParser",
            "error_phase": "json_decode",
            "completion_excerpt": completion[:_MAX_EXCERPT],
            "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
        },
    )


_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)
"""Captures the body of a Markdown code fence (``json`` lang tag optional).

Non-greedy so two adjacent fenced blocks are not joined; only the first match
is used during recovery.
"""


class _RawEnvelope(BaseModel):
    """Internal validator for the :data:`JSON_MODE_OUTPUT_FORMAT` schema.

    ``extra="forbid"`` rejects unknown top-level keys; ``action`` is a literal
    union so any value besides ``"tool"`` / ``"final"`` raises immediately.

    ``tool_args=None`` is tolerated for either action and treated as ``{}``
    during the envelope→ParseResult conversion. ``tool_name`` / ``answer``
    presence is enforced semantically in
    :meth:`JsonModeParser._to_parse_result`.
    """

    model_config = ConfigDict(extra="forbid")

    thought: str
    action: Literal["tool", "final"]
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    answer: str | None = None


def _strip_code_fences(text: str) -> str | None:
    """Best-effort extraction of a JSON object from arbitrary text.

    Strategy, in order:

    1. If the input contains a Markdown code fence (``\\`\\`\\`json ... \\`\\`\\```
       or bare ``\\`\\`\\``` ... ``\\`\\`\\```), return the first fence body.
    2. Otherwise slice between the first ``{`` and the last ``}`` (inclusive).
    3. If neither a fence body nor a brace pair exists, return ``None``.

    The returned string is NOT validated as JSON — callers must
    :func:`json.loads` it themselves. This helper exists so the JSON-mode and
    prose-mode parsers share a single recovery rule.

    Args:
        text: Arbitrary text that may contain a JSON object.

    Returns:
        The recovered candidate substring, or ``None`` when no candidate
        could be located.
    """
    match = _FENCE_RE.search(text)
    candidate = match.group("body") if match else text
    open_idx = candidate.find("{")
    close_idx = candidate.rfind("}")
    if open_idx == -1 or close_idx == -1 or close_idx < open_idx:
        return None
    return candidate[open_idx : close_idx + 1]


class JsonModeParser:
    """Strict JSON-envelope parser with one fence-stripping retry pass.

    The default parser when the loop instructs the model to use JSON mode
    (via :func:`fifty_agent_sdk.prompts.json_mode_template`). Consumes the schema
    defined in :data:`fifty_agent_sdk.prompts.JSON_MODE_OUTPUT_FORMAT`.

    Failure surfaces:

    * ``error_phase="empty_completion"`` — empty/whitespace-only input.
    * ``error_phase="json_decode"`` — both the strict and the recovery pass
      failed to produce valid JSON, or the text left to decode nests deeper
      than 65 levels: the envelope object plus 64 for ``tool_args``
      (:data:`fifty_agent_sdk._json_depth.MAX_TOOL_ARGS_DEPTH`, BR-019).
      Such text never reaches ``json.loads``: a too-deep strict-pass text
      goes to the recovery pass, and a too-deep recovery candidate, or no
      candidate after a too-deep strict pass, raises with
      ``context["max_tool_args_depth"]`` (see :meth:`_load_json`). Valid
      JSON that is too deep goes to that pass too, which earlier releases
      ran only after a decode failure, so it can now parse to a
      ``{``...``}`` candidate inside it. The check covers the whole
      envelope, not only ``tool_args``: an ``answer`` or ``thought`` value
      nested more than 64 levels, which the schema rejects anyway, now gets
      this ``json_decode`` error where earlier releases raised
      ``schema_validation`` whenever ``json.loads`` could decode it. It also
      reads brackets rather than JSON, so invalid text whose brackets open
      more than 65 levels gets it where earlier releases reported the
      invalid JSON when the recovery pass finds no candidate or the check
      also refuses the candidate (``x`` followed by 100 ``[``, for
      example); a candidate within the limit is decoded as before.
      ``json.loads`` itself
      raises :class:`RecursionError` at a depth that depends on the CPython
      version and on the caller's stack, not on
      :func:`sys.getrecursionlimit` alone: measured for BR-019 with the
      default limit of 1000, from a module's top level it decoded up to
      995 levels on CPython 3.11.15, 9998 on 3.13.2 and 116,213 on 3.14.3,
      and inside an async pytest test up to 948, 9976 and about 116,100
      (the 3.14.3 figure moved by a level between runs). That
      ``RecursionError`` is still translated into :class:`ParserError`
      (BR-017) so the loop's ``ParserError`` contract holds.
    * ``error_phase="schema_validation"`` — JSON decoded but the envelope
      does not match the schema (unknown key, wrong ``action`` value,
      missing required field for the chosen action).
    """

    def parse(self, completion: str) -> ParseResult:
        """See :meth:`fifty_agent_sdk.parser.base.Parser.parse`."""
        if not completion or not completion.strip():
            raise ParserError(
                "completion is empty",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "empty_completion",
                    "completion_excerpt": "",
                },
            )
        raw = self._load_json(completion)
        envelope = self._validate(raw, completion)
        return self._to_parse_result(envelope, completion)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _load_json(self, completion: str) -> Any:
        """Strict-then-recover JSON decode. Raises on total failure.

        Before each ``json.loads``, the text it would decode is checked for
        nesting deeper than :data:`_MAX_ENVELOPE_DEPTH` (65: the envelope
        object plus 64 levels of ``tool_args``; BR-019). A strict-pass text
        that is too deep is not decoded and goes to the recovery pass, as a
        decode failure does, so prose with an unclosed bracket before a valid
        envelope still recovers. Valid JSON that is too deep takes that pass
        too; before BR-019 it was passed to ``json.loads`` as it stood (for
        example an array holding an envelope and a 70-level array raised
        ``schema_validation``, and now parses to that envelope). A recovery
        candidate that is too deep, or
        none at all after a too-deep strict pass, raises :class:`ParserError`
        (``error_phase="json_decode"``) with ``context["max_tool_args_depth"]``
        and no ``__cause__``.

        :class:`RecursionError` from ``json.loads`` is still caught at BOTH
        decode attempts and translated into :class:`ParserError`
        (``error_phase="json_decode"``), as BR-017 added it: the
        :class:`Parser` protocol mandates ``ParserError`` on malformed input,
        and the loop catches only ``ParserError`` — a raw ``RecursionError``
        would escape the async generator with no ``ErrorEvent`` and no
        terminal ``FinalEvent``. Since BR-019 the depth check keeps nesting
        far below the depths at which ``json.loads`` raised it in the
        BR-019 measurements (see the class docstring).
        """
        strict = completion.strip()
        if json_nesting_exceeds(strict, _MAX_ENVELOPE_DEPTH):
            return self._recover(completion, None)
        try:
            return json.loads(strict)
        except RecursionError as depth_err:
            raise ParserError(
                "could not decode JSON envelope: nesting depth exceeds the recursion limit",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "json_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(depth_err),
                },
            ) from depth_err
        except ValueError as first_err:
            # CPython also raises bare ValueError for implementation limits
            # such as oversized integer literals. Keep it in the established
            # decode phase instead of parsing version-specific error text.
            # The recovery runs inside this handler, as before BR-019, so
            # implicit exception chaining is unchanged (pinned by
            # test_failed_decodes_keep_the_strict_error_in_their_exception_chain).
            return self._recover(completion, first_err)

    def _recover(self, completion: str, first_err: ValueError | None) -> Any:
        """Run the one fence/brace recovery pass of :meth:`_load_json`.

        Args:
            completion: The raw completion.
            first_err: The strict pass's decode error, or ``None`` when the
                strict pass was skipped because its text nests too deeply
                (BR-019).

        Returns:
            The decoded recovery candidate.

        Raises:
            ParserError: ``error_phase="json_decode"`` when no candidate
                exists, the candidate nests too deeply, or it does not
                decode.
        """
        recovered = _strip_code_fences(completion)
        if recovered is None:
            if first_err is None:
                raise _depth_error(completion)
            raise ParserError(
                "could not decode JSON envelope",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "json_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(first_err),
                },
            ) from first_err
        if json_nesting_exceeds(recovered, _MAX_ENVELOPE_DEPTH):
            raise _depth_error(completion)
        try:
            return json.loads(recovered)
        except RecursionError as depth_err:
            raise ParserError(
                "could not decode JSON envelope: nesting depth exceeds the recursion limit",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "json_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(depth_err),
                },
            ) from depth_err
        except ValueError as second_err:
            raise ParserError(
                "could not decode JSON envelope after fence recovery",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "json_decode",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": repr(second_err),
                },
            ) from second_err

    def _validate(self, raw: Any, completion: str) -> _RawEnvelope:
        """Validate the decoded JSON against :class:`_RawEnvelope`."""
        try:
            return _RawEnvelope.model_validate(raw)
        except ValidationError as exc:
            raise ParserError(
                "JSON envelope did not match required schema",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "schema_validation",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "cause": str(exc),
                },
            ) from exc

    def _to_parse_result(self, env: _RawEnvelope, completion: str) -> ParseResult:
        """Convert a validated envelope into the public :data:`ParseResult`."""
        if env.action == "tool":
            # Strip before the emptiness check (and before emitting the
            # ToolCall): a whitespace-only name must take the same
            # schema_validation path as an empty one, and a padded name
            # must reach the registry in the same stripped form the prose
            # parser produces (prose_mode strips the Action: header).
            tool_name = env.tool_name.strip() if env.tool_name is not None else None
            if not tool_name:
                raise ParserError(
                    "action='tool' requires non-empty tool_name",
                    context={
                        "parser": "JsonModeParser",
                        "error_phase": "schema_validation",
                        "completion_excerpt": completion[:_MAX_EXCERPT],
                        "missing": "tool_name",
                    },
                )
            tool_call = ToolCall(
                name=tool_name,
                args=env.tool_args if env.tool_args is not None else {},
            )
            return ThoughtAction(thought=env.thought, tool_call=tool_call)

        # env.action == "final" — Literal narrows the alternative away.
        if env.answer is None:
            raise ParserError(
                "action='final' requires non-null answer",
                context={
                    "parser": "JsonModeParser",
                    "error_phase": "schema_validation",
                    "completion_excerpt": completion[:_MAX_EXCERPT],
                    "missing": "answer",
                },
            )
        return FinalAnswer(thought=env.thought, content=env.answer)


__all__ = ["JsonModeParser"]
