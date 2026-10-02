"""Tests for ``fifty_agent_sdk._json_depth`` (BR-019).

``json_nesting_exceeds(text, max_depth)`` must say whether ``text`` nests JSON
objects and arrays deeper than ``max_depth``, reading strings and escapes the
way ``json.loads`` does, without decoding the text.

Fixture note: the check starts with a ``str.count`` shortcut that accepts any
text holding no more ``[`` and ``{`` characters than ``max_depth``. A bare
chain nested exactly ``max_depth`` levels holds exactly that many, so it never
reaches the counting loop. The boundary tests therefore run each chain twice:
bare (the shortcut) and inside an array beside the string ``"["`` (one more
``[`` character than levels, so the counting loop runs), so a mutant of
either is seen.

What these do NOT pin: the cost of the check. Linearity on adversarial text
(an unterminated string full of escaped quotes) was timed outside the suite
(BR-019 evidence, P4 and mutant M12); no test asserts a duration.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import pytest

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds

# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _object_chain(depth: int) -> str:
    """``{"a":{"a":...0...}}``: ``depth`` objects, nothing else."""
    return '{"a":' * depth + "0" + "}" * depth


def _array_chain(depth: int) -> str:
    """``[[...0...]]``: ``depth`` arrays, nothing else."""
    return "[" * depth + "0" + "]" * depth


def _mixed_chain(depth: int) -> str:
    """Objects and arrays alternating, starting with an object, ``depth`` levels."""
    openers = ['{"k":' if i % 2 == 0 else "[" for i in range(depth)]
    closers = ["}" if i % 2 == 0 else "]" for i in reversed(range(depth))]
    return "".join(openers) + "0" + "".join(closers)


def _beside_a_bracket_string(chain: str) -> str:
    """The chain inside an array beside the string ``"["``: one level more, two ``[`` more."""
    return '["[",' + chain + "]"


def _tree_depth(value: Any) -> int:
    """Nesting depth of a decoded JSON value, walked iteratively (exact dicts and lists)."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if type(node) is dict:
            children: list[Any] = list(node.values())
        elif type(node) is list:
            children = node
        else:
            continue
        depth += 1
        deepest = max(deepest, depth)
        stack.extend((child, depth) for child in children)
    return deepest


_CHAINS: dict[str, Callable[[int], str]] = {
    "object": _object_chain,
    "array": _array_chain,
    "mixed": _mixed_chain,
}


# ---------------------------------------------------------------------- #
# The limit and the boundary                                             #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("sibling", [False, True], ids=["bare", "beside_bracket_string"])
@pytest.mark.parametrize("shape", list(_CHAINS))
@pytest.mark.parametrize("limit", [1, 5, MAX_TOOL_ARGS_DEPTH])
def test_depth_at_the_limit_is_allowed_and_one_past_is_refused(
    limit: int, shape: str, sibling: bool
) -> None:
    """At ``max_depth`` levels the text passes, and one level more is refused (BR-019)."""
    if sibling:
        # The wrapping array adds one level, so build the chain one shorter.
        at_limit = _beside_a_bracket_string(_CHAINS[shape](limit - 1))
        past_limit = _beside_a_bracket_string(_CHAINS[shape](limit))
        assert at_limit.count("[") + at_limit.count("{") == limit + 1  # reaches the loop
    else:
        at_limit = _CHAINS[shape](limit)
        past_limit = _CHAINS[shape](limit + 1)
        assert at_limit.count("[") + at_limit.count("{") == limit  # the shortcut
    assert _tree_depth(json.loads(at_limit)) == limit
    assert _tree_depth(json.loads(past_limit)) == limit + 1

    assert json_nesting_exceeds(at_limit, limit) is False
    assert json_nesting_exceeds(past_limit, limit) is True


def test_documented_limit_is_64() -> None:
    """The limit the README and CHANGELOG state is the constant's value (BR-019)."""
    assert MAX_TOOL_ARGS_DEPTH == 64


@pytest.mark.parametrize(
    ("text", "depth"),
    [("1", 0), ("null", 0), ('""', 0), ('"x"', 0), ("{}", 1), ("[]", 1), ('{"a":[]}', 2)],
)
def test_scalars_and_empty_containers(text: str, depth: int) -> None:
    """A scalar is 0 levels and each object or array is one more (BR-019).

    These texts hold no more ``[`` and ``{`` than their depth, so at
    ``max_depth == depth`` the ``str.count`` shortcut answers and the counting
    loop's comparison is not reached (mutant M2 passes here); the
    beside-bracket rows of the boundary test pin that comparison.
    """
    assert _tree_depth(json.loads(text)) == depth
    assert json_nesting_exceeds(text, depth) is False
    if depth > 0:
        assert json_nesting_exceeds(text, depth - 1) is True


# ---------------------------------------------------------------------- #
# Strings and escapes                                                    #
# ---------------------------------------------------------------------- #


def test_brackets_inside_strings_and_keys_do_not_count() -> None:
    """Brackets inside a string value or a key are not nesting (BR-019)."""
    in_value = json.dumps({"s": "[" * 100 + "{" * 100})
    in_key = json.dumps({"[[[[{{{{": 1})
    for text in (in_value, in_key):
        assert _tree_depth(json.loads(text)) == 1
        assert json_nesting_exceeds(text, 1) is False
        assert json_nesting_exceeds(text, 0) is True


def test_escapes_end_and_continue_strings_correctly() -> None:
    """An escaped quote keeps the string open, and an escaped backslash does not (BR-019)."""
    # {"s":"\"[[[["}: the escaped quote does not close the string, so the
    # brackets after it are inside it.
    escaped_quote = '{"s":"\\"[[[["}'
    assert json.loads(escaped_quote) == {"s": '"[[[['}
    assert json_nesting_exceeds(escaped_quote, 1) is False

    # ["\\", [[0]]]: the string holds one backslash, and the quote after it
    # closes the string, so the two arrays after it count.
    escaped_backslash = '["\\\\", [[0]]]'
    assert json.loads(escaped_backslash) == ["\\", [[0]]]
    assert json_nesting_exceeds(escaped_backslash, 2) is True
    assert json_nesting_exceeds(escaped_backslash, 3) is False

    # A bracket written as a JSON string is not a bracket.
    assert json_nesting_exceeds('"["', 0) is False


def test_unterminated_string_reads_to_the_end_and_a_deep_invalid_prefix_counts() -> None:
    """An unterminated string runs to the end; brackets opened before invalid text count (BR-019).

    The last assertion is the soundness rule: ``json.loads`` opens 100
    arrays before it fails at the ``x``, so the check must refuse the text.
    The third covers a string that ends in a lone backslash, which the
    string token's optional ``\\`` before the end of text exists for. This
    test cannot see the check's cost; the timings are in the BR-019
    evidence (P4, M12, and round 3 for the optional backslash).
    """
    assert json_nesting_exceeds('["abc' + "[" * 100, 1) is False
    assert json_nesting_exceeds('["' + '\\"' * 1000 + "[" * 100, 1) is False
    assert json_nesting_exceeds('["' + "[" * 100 + "\\", 1) is False

    deep_invalid = "[" * 100 + "x"
    with pytest.raises(ValueError):
        json.loads(deep_invalid)
    assert json_nesting_exceeds(deep_invalid, MAX_TOOL_ARGS_DEPTH) is True


# ---------------------------------------------------------------------- #
# Agreement with json.loads on a seeded corpus                           #
# ---------------------------------------------------------------------- #

_ALPHABET = [
    "a",
    "Z",
    "0",
    " ",
    "[",
    "]",
    "{",
    "}",
    '"',
    "\\",
    ",",
    ":",
    "é",
    "ج",
    "😀",
    "\n",
    "\t",
]


def _random_string(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, 8)))


def _random_value(rng: random.Random, depth: int) -> Any:
    """A value nested exactly ``depth`` levels, with shallow siblings along the way."""
    if depth == 0:
        return rng.choice([None, True, False, rng.randint(-5, 5), 0.5, _random_string(rng)])
    children = [_random_value(rng, depth - 1)]
    children += [
        _random_value(rng, rng.randint(0, min(depth - 1, 1))) for _ in range(rng.randint(0, 2))
    ]
    rng.shuffle(children)
    if rng.random() < 0.5:
        return children
    return {f"{_random_string(rng)}{i}": child for i, child in enumerate(children)}


def _naive_depth(text: str) -> int:
    """Bracket depth ignoring strings: what a scan without string handling would see."""
    depth = deepest = 0
    for ch in text:
        if ch in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif ch in "]}":
            depth -= 1
    return deepest


def test_verdict_matches_decoded_depth_on_a_seeded_corpus() -> None:
    """For every corpus text, the check agrees with the depth ``json.loads`` decodes (BR-019).

    500 values (seed 19), nested 0 to 80 levels, with strings holding
    brackets, quotes, backslashes and non-ASCII text, each dumped four ways
    (``ensure_ascii`` on and off, default and compact separators). The
    preconditions assert that strings change the bracket count of most
    texts, so the corpus can see a check that ignores strings, and that
    enough values sit at the limit and past it.
    """
    rng = random.Random(19)
    disagreements = 0
    strings_matter = 0
    near_limit = 0
    past_limit = 0
    for _ in range(500):
        value = _random_value(rng, rng.randint(0, 80))
        texts = [
            json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)
            for ensure_ascii in (True, False)
            for separators in ((", ", ": "), (",", ":"))
        ]
        # The four texts decode to the same tree, so one decode gives its depth.
        depth = _tree_depth(json.loads(texts[0]))
        if _naive_depth(texts[-1]) != depth:
            strings_matter += 1
        near_limit += depth in (
            MAX_TOOL_ARGS_DEPTH - 1,
            MAX_TOOL_ARGS_DEPTH,
            MAX_TOOL_ARGS_DEPTH + 1,
        )
        past_limit += depth > MAX_TOOL_ARGS_DEPTH
        for text in texts:
            for limit in (0, 1, 63, 64, 65):
                if json_nesting_exceeds(text, limit) != (depth > limit):
                    disagreements += 1
    assert strings_matter >= 250
    assert near_limit >= 5
    assert past_limit >= 25
    assert disagreements == 0
