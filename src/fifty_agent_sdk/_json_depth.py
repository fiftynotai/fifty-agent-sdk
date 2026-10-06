"""Nesting-depth limit for model-written tool arguments (BR-019).

Rule: before the SDK decodes tool arguments a model wrote, it checks the
text with :func:`json_nesting_exceeds` and refuses arguments nested deeper
than :data:`MAX_TOOL_ARGS_DEPTH` levels with the typed error that decode site
already raises for unusable arguments (a text parser first sends a refused
strict-pass text to its recovery pass, as it does a decode failure). There
are three such sites, and each
checks exactly the text it is about to pass to ``json.loads``:

* ``OpenAICompatibleClient._map_tool_calls`` (native ``tool_calls``
  ``arguments``): ``LLMError`` with ``context["type"] ==
  "MalformedResponse"`` and ``context["max_tool_args_depth"]``;
* ``JsonModeParser`` (the whole envelope, checked against
  ``MAX_TOOL_ARGS_DEPTH + 1`` because the envelope object is one level above
  ``tool_args``): ``ParserError`` with ``error_phase="json_decode"``;
* ``ProseModeParser`` (the ``Action Input:`` body): ``ParserError`` with
  ``error_phase="action_input_decode"``.

What counts as a level: each JSON object or array on the deepest path.
``{"q": 1}`` is 1 level, ``{"q": [1]}`` is 2, and a scalar is 0. Brackets
inside strings, keys included, do not count.

Why a limit, and why 64:

* Before BR-019, a native tool call whose arguments ``json.loads`` decoded
  but the SDK could not re-encode for the next request ran its tool, and
  building that request raised a raw ``RecursionError``. Arguments nested
  past what ``json.loads`` decodes raised it from the decode instead. How
  deep each step reaches depends on the CPython version, the nesting shape
  and the call stack. Measured for BR-019 with stdlib ``json`` and its
  defaults, from a module's top level: ``json.loads`` and ``json.dumps``
  both reach 995 levels on CPython 3.11.5 and 3.11.15 and 9998 on 3.13.2,
  for nested objects and nested arrays alike; on 3.14.3 ``json.loads``
  reaches 116,213 levels, and ``json.dumps`` 61,525 for nested objects and
  104,591 for nested arrays. So the stdlib leaves a decode/encode gap only
  on 3.14.3. The SDK re-encodes the replayed arguments from a deeper call
  stack than it decodes them (3 Python frames deeper on 3.11.15). On
  3.11.15, where one more Python frame lowered both limits by one level
  (995 at a module's top level, 994 inside a function), that left a
  3-level band that decoded but did not re-encode; on 3.13.2 no such band
  was found. The check reads the text without recursion, so where it
  refuses does not depend on the interpreter or the stack, and 64 is far
  below every limit measured.
* Arguments that are decoded go on to pydantic serialisers and parsers,
  inside the SDK and in host code. Measured for BR-019 with pydantic 2.13.4
  (pydantic-core 2.46.4) on CPython 3.14.3, and 2.13.5 (2.46.5) on 3.11.15
  and 3.13.2: ``ToolCall.model_dump_json()``, ``ToolCall.model_dump(mode=
  "json")`` and ``ActionEvent.model_dump_json()`` handle arguments nested up
  to 255 levels, and ``ChatMessage.model_validate_json`` of an assistant
  turn holding them up to 197.
* When BR-019 was written, no tool argument that reached one of the three
  sites in this repository's test suite or golden fixtures nested deeper
  than 1 level (2 counting the JSON-mode envelope around it).

The limit is private and fixed: there is no setting. A real need can raise
it in a patch release; the tests derive their depths from the constant.

Not covered: a custom ``LLMClient`` or ``Parser`` decodes its own arguments,
and ``ToolCall`` values built by host code are never checked.

Reading rule of the check: outside a string, ``"`` opens one; inside a
string, ``\\`` makes the next character literal, an unescaped ``"`` closes the
string, and a string with no closing quote runs to the end of the text;
outside strings, ``[`` and ``{`` open a level and ``]`` and ``}`` close one.
For every text ``json.loads`` accepts, the result equals "the decoded value
nests deeper than ``max_depth``". For any other text, the check reads the
same tokens as ``json.loads`` up to the decoder's first invalid character,
so it returns ``True`` for every text on which ``json.loads`` would open
more than ``max_depth`` objects and arrays before failing. It also returns
``True`` for invalid text whose brackets open more than ``max_depth``
levels where ``json.loads`` rejects it earlier: for example ``x`` followed
by 100 ``[`` (rejected at its first character), or deep brackets after a
complete value. Such text was refused before BR-019 too, as invalid JSON.
The native site now reports it with the depth error. The text parsers
report it only when their recovery pass finds no candidate or the check
also refuses the candidate. A refusal therefore means
the check refused the text, not that the text decodes to a deeply nested
value.

Cost: text holding no more ``[`` and ``{`` characters in total than the
limit cannot nest deeper, and is accepted after two ``str.count`` calls.
Otherwise one compiled regular-expression pass removes the string tokens,
and Python then looks only at the brackets left, stopping at the first
level past the limit. Every string token matches where it starts (an
unterminated one runs to the end of the text), so the removal never retries
a match from a later quote, and its work grows linearly with the length of
the text. Measured for BR-019 on CPython 3.11.15, 3.13.2 and 3.14.3: 10 MB
of arguments holding few brackets took about 6.5 ms, less than
``json.loads`` of the same text; 10 MB of ``[[],[],...]`` took 0.69 s,
0.85 s and 1.02 s, 1.5 to 6.2 times ``json.loads``, growing linearly from
1 MB. The JSON-mode parser can check one completion twice (its strict text,
then its recovery candidate).

Tool output gets no walker here. BR-022 contained the failures where the
output is rendered (``loop._serialize_tool_output``, and the Runner's audit
summary) instead of bounding its depth, because nothing the SDK does with
the rendered text fails on depth.
This module is private and carries no semver protection. It must never use
the ``json`` module: BR-017's tests replace ``json.loads`` process-wide.
"""

from __future__ import annotations

import re
from typing import Final

MAX_TOOL_ARGS_DEPTH: Final[int] = 64
"""Deepest nesting of model-written tool arguments the SDK decodes (BR-019)."""

# One JSON string token. The possessive loops never backtrack, and the end
# alternative (`\\?\Z`) lets an unterminated string run to the end of the
# text, so the token matches wherever a quote outside a string starts one.
# Without that alternative, an unterminated string would fail the match and
# the search would retry from every later quote: quadratic on input such as
# '"' followed by many '\\"' pairs, which the model controls. The optional
# backslash covers text that ends inside a string with a lone `\`; without
# it that string fails the match too, its brackets are counted, and the
# search retries from every later quote (measured quadratic for BR-019 on
# CPython 3.14.3: 0.06 / 0.23 / 0.93 s at 2,000 / 4,000 / 8,000 escaped
# quotes).
_STRING_TOKEN: Final[re.Pattern[str]] = re.compile(r'"(?:[^"\\]++|\\[\s\S])*+(?:"|\\?\Z)')

_BRACKET: Final[re.Pattern[str]] = re.compile(r"[\[\]{}]")


def json_nesting_exceeds(text: str, max_depth: int) -> bool:
    """Return whether ``text`` nests JSON objects or arrays deeper than ``max_depth``.

    Reads ``text`` by the rule in the module docstring, without decoding it
    and without recursion, and stops as soon as the opening brackets read so
    far outnumber the closing ones by more than ``max_depth`` (for valid
    JSON, the number of objects and arrays open at that point).

    Args:
        text: The text the caller is about to pass to ``json.loads``.
        max_depth: The deepest nesting allowed. ``0`` allows only scalars.

    Returns:
        ``True`` if that count exceeds ``max_depth`` anywhere in ``text``,
        else ``False``.
    """
    # Every level opens with one of these characters, inside a string or
    # not, so text with no more of them than the limit cannot exceed it.
    # Most tool arguments end here, after two scans in C.
    if text.count("[") + text.count("{") <= max_depth:
        return False
    depth = 0
    for bracket in _BRACKET.finditer(_STRING_TOKEN.sub("", text)):
        if bracket.group() in "[{":
            depth += 1
            if depth > max_depth:
                return True
        else:
            depth -= 1
    return False


__all__ = ["MAX_TOOL_ARGS_DEPTH", "json_nesting_exceeds"]
