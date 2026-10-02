"""Tests for ``fifty_agent_sdk.parser.prose_mode.ProseModeParser``."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import pytest

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds
from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import FinalAnswer, Parser, ProseModeParser, ThoughtAction
from fifty_agent_sdk.parser import prose_mode as prose_mode_module


def _parser() -> ProseModeParser:
    return ProseModeParser()


# ---------------------------------------------------------------------- #
# Happy paths                                                            #
# ---------------------------------------------------------------------- #


def test_happy_path_action() -> None:
    completion = 'Thought: I need to search.\nAction: search\nAction Input: {"q": "x"}'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "I need to search."
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_happy_path_final_answer() -> None:
    completion = "Thought: I have the answer.\nFinal Answer: 42"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "I have the answer."
    assert result.content == "42"


def test_action_input_multiline_json_is_decoded() -> None:
    completion = (
        'Thought: Looking it up.\nAction: search\nAction Input: {\n  "q": "x",\n  "limit": 5\n}'
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {"q": "x", "limit": 5}


def test_action_input_with_code_fences_is_recovered() -> None:
    completion = 'Thought: t\nAction: search\nAction Input: ```json\n{"q": "x"}\n```'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {"q": "x"}


def test_multiline_final_answer_is_captured() -> None:
    completion = "Thought: T\nFinal Answer: line1\nline2"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "line1\nline2"


# ---------------------------------------------------------------------- #
# Tolerance: case + whitespace                                           #
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "completion",
    [
        "THOUGHT: T\nFINAL ANSWER: ok",
        "thought: T\nfinal answer: ok",
        "Thought:   T\n   Final Answer:   ok",
        "  \n\nThought: T\nFinal Answer: ok\n\n  ",
    ],
)
def test_case_variants_are_tolerated_for_final(completion: str) -> None:
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "T"
    assert result.content == "ok"


@pytest.mark.parametrize(
    "completion",
    [
        'THOUGHT: T\nACTION: search\nACTION INPUT: {"q": "x"}',
        'thought: T\naction: search\naction input: {"q": "x"}',
    ],
)
def test_case_variants_are_tolerated_for_action(completion: str) -> None:
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_whitespace_around_headers_tolerated() -> None:
    completion = (
        '\n\n   Thought:    T   \n   Action:    search   \n   Action Input:    {"q": "x"}   \n\n'
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "T"
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


# ---------------------------------------------------------------------- #
# Tie-break + special cases                                              #
# ---------------------------------------------------------------------- #


def test_both_action_and_final_answer_present_prefers_action() -> None:
    """Documented tie-break: tool path wins when both shapes are present."""
    completion = 'Thought: T\nAction: search\nAction Input: {"q": "x"}\nFinal Answer: stale'
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"


def test_missing_final_answer_body_is_tolerated_as_empty() -> None:
    """Documented choice: prose parser is tolerant; empty content is fine."""
    completion = "Thought: T\nFinal Answer:"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == ""


# ---------------------------------------------------------------------- #
# Failure paths                                                          #
# ---------------------------------------------------------------------- #


def test_missing_action_input_raises_header_match() -> None:
    completion = "Thought: T\nAction: search"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "header_match"


def test_no_headers_raises_header_match() -> None:
    completion = "random prose with no structure at all"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "ProseModeParser"
    assert ctx["error_phase"] == "header_match"
    assert "completion_excerpt" in ctx


def test_action_input_invalid_json_raises_action_input_decode() -> None:
    completion = "Thought: T\nAction: search\nAction Input: not json"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "action_input_decode"


def test_action_input_non_object_json_raises() -> None:
    """Action Input must decode to an object, not a scalar/list."""
    completion = "Thought: T\nAction: search\nAction Input: [1, 2, 3]"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "action_input_decode"


def test_empty_completion_raises() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_whitespace_only_completion_raises() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("   \n\t  ")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_header_match_completion_excerpt_is_bounded() -> None:
    big = "no headers here " * 100  # > 200 chars, no headers
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(big)
    excerpt = excinfo.value.context["completion_excerpt"]
    assert len(excerpt) <= 200


def test_huge_whitespace_payload_does_not_hang() -> None:
    """ReDoS sanity check: large whitespace-only input fails fast."""
    payload = " " * 10000
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(payload)
    # whitespace-only triggers the empty_completion guard before regex.
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_strict_action_input_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins strict Action Input recursion containment."""
    sentinel = RecursionError("deterministic depth failure")
    calls = 0

    def raise_recursion(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise sentinel

    monkeypatch.setattr(prose_mode_module.json, "loads", raise_recursion)
    completion = 'Thought: T\nAction: search\nAction Input: {"q":"x"}'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "ProseModeParser"
    assert ctx["error_phase"] == "action_input_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 1


def test_recovery_action_input_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins recovery Action Input recursion containment."""
    first = json.JSONDecodeError("strict failed", "fence", 0)
    sentinel = RecursionError("deterministic recovery depth failure")
    errors = iter((first, sentinel))
    calls = 0

    def raise_scripted(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise next(errors)

    monkeypatch.setattr(prose_mode_module.json, "loads", raise_scripted)
    completion = 'Thought: T\nAction: search\nAction Input: ```json\n{"q":"x"}\n```'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "action_input_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 2


def test_oversized_integer_action_input_strict_decode_is_contained() -> None:
    """BR-013 contains strict Action Input bare ValueError."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    completion = f"Thought: T\nAction: search\nAction Input: {digits}"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode Action Input JSON"
    assert excinfo.value.context["error_phase"] == "action_input_decode"
    assert type(excinfo.value.__cause__) is ValueError


def test_oversized_integer_action_input_recovery_decode_is_contained() -> None:
    """BR-013 contains recovery Action Input bare ValueError."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    completion = f'Thought: T\nAction: search\nAction Input: ```json\n{{"n":{digits}}}\n```'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode Action Input JSON after fence recovery"
    assert excinfo.value.context["error_phase"] == "action_input_decode"
    assert type(excinfo.value.__cause__) is ValueError


def test_malformed_action_input_preserves_decode_message_and_json_cause() -> None:
    """BR-013 leaves malformed Action Input syntax behavior unchanged."""
    completion = "Thought: T\nAction: search\nAction Input: not json"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode Action Input JSON"
    assert excinfo.value.context == {
        "parser": "ProseModeParser",
        "error_phase": "action_input_decode",
        "completion_excerpt": completion,
        "cause": repr(excinfo.value.__cause__),
    }
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# ---------------------------------------------------------------------- #
# Nesting limit (BR-019)                                                 #
# ---------------------------------------------------------------------- #

_DEPTH_MESSAGE = (
    f"could not decode Action Input JSON: nesting deeper than {MAX_TOOL_ARGS_DEPTH} levels"
)


def _nested_args(depth: int) -> str:
    """``Action Input`` text nested ``depth`` levels: ``{"b":"[","a":[[...0...]]}`` (BR-019).

    The ``"["`` string gives the text one more ``[`` than levels, so the depth
    check reads past its ``str.count`` shortcut at the limit too.
    """
    return '{"b":"[","a":' + "[" * (depth - 1) + "0" + "]" * (depth - 1) + "}"


def _tool_completion(body: str) -> str:
    return "Thought: T\nAction: search\nAction Input: " + body


def test_action_input_nested_at_the_limit_parses() -> None:
    """An ``Action Input`` nested 64 levels parses as before (BR-019)."""
    body = _nested_args(MAX_TOOL_ARGS_DEPTH)
    result = _parser().parse(_tool_completion(body))
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"
    assert result.tool_call.args == json.loads(body)


def test_action_input_one_past_the_limit_raises_parser_error() -> None:
    """An ``Action Input`` one level past the limit raises a fixed-message ParserError (BR-019)."""
    completion = _tool_completion(_nested_args(MAX_TOOL_ARGS_DEPTH + 1))
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context == {
        "parser": "ProseModeParser",
        "error_phase": "action_input_decode",
        "completion_excerpt": completion[:200],
        "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
    }
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("wrap", "strict_pass_reaches_json_loads"),
    [
        (lambda body: "```json\n" + body + "\n```", False),
        (lambda body: '"' + body, True),
    ],
    ids=["fenced", "quote_prefixed"],
)
def test_too_deep_action_input_is_refused_in_recovery(
    wrap: Callable[[str], str], strict_pass_reaches_json_loads: bool
) -> None:
    """A too-deep body that only the recovery pass extracts is refused there too (BR-019).

    Two routes into the recovery pass: the fenced body is itself too deep,
    so its strict pass is skipped; behind a stray quote the deep part reads
    as string text, so the strict pass decodes, fails, and recovers. The
    control is the same body at the limit, which parses.
    """
    at_limit = wrap(_nested_args(MAX_TOOL_ARGS_DEPTH))
    past_limit = wrap(_nested_args(MAX_TOOL_ARGS_DEPTH + 1))
    assert json_nesting_exceeds(past_limit, MAX_TOOL_ARGS_DEPTH) is not (
        strict_pass_reaches_json_loads
    )

    assert isinstance(_parser().parse(_tool_completion(at_limit)), ThoughtAction)
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(_tool_completion(past_limit))
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context["error_phase"] == "action_input_decode"
    assert excinfo.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert excinfo.value.__cause__ is None


def test_invalid_text_past_the_limit_with_no_candidate_gets_the_depth_error() -> None:
    """Invalid ``Action Input`` text past the limit, with no ``{``...``}`` candidate, gets the depth error (BR-019).

    ``x`` followed by 100 ``[`` fails ``json.loads`` at its first character
    and nests nothing; the recovery pass finds no candidate. Before BR-019
    it got ``could not decode Action Input JSON`` with the decode error as
    ``__cause__``.
    """
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(_tool_completion("x" + "[" * 100))
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert excinfo.value.__cause__ is None


def test_invalid_deep_text_with_a_candidate_gets_the_recovery_error() -> None:
    """Invalid deep ``Action Input`` text whose candidate is within the limit gets the invalid-JSON error (BR-019).

    ``[`` x100 then ``{"a": oops}``: the body is refused, the recovery
    candidate ``{"a": oops}`` passes the check and fails to decode. The
    outcome is the same as before BR-019.
    """
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(_tool_completion("[" * 100 + '{"a": oops}'))
    assert str(excinfo.value) == "could not decode Action Input JSON after fence recovery"
    assert "max_tool_args_depth" not in excinfo.value.context
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_valid_but_too_deep_body_goes_to_the_recovery_pass() -> None:
    """A valid but too-deep ``Action Input`` is recovered to the object inside it (BR-019).

    ``[{"a": 1}, <70-level array>]`` is valid JSON, 71 levels. It is refused
    and goes to the recovery pass, which extracts ``{"a": 1}``. 1.10.1
    decoded it as it stood and raised ``Action Input JSON must decode to an
    object`` (measured on the BR-021 ``src``, whose parser modules are
    identical to 1.10.1's; BR-019 evidence, round 3), so the loop, with the
    retry enabled (as measured), took its parser retry where it now
    dispatches.
    """
    body = '[{"a":1}, ' + "[" * 70 + "]" * 70 + "]"
    json.loads(body)
    result = _parser().parse(_tool_completion(body))
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {"a": 1}


def test_failed_decodes_keep_the_strict_error_in_their_exception_chain() -> None:
    """The recovery runs inside the strict pass's ``except`` block, so exception chaining is as before BR-019.

    A failed recovery's own decode error carries the strict pass's error in
    its ``__context__`` chain, and the no-candidate error is raised while the strict
    error is being handled. Moving the recovery out of that block (mutant
    MGp) keeps every message and ``__cause__`` and drops both.
    """
    body = 'note {"q": nope} end'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(_tool_completion(body))
    assert str(excinfo.value) == "could not decode Action Input JSON after fence recovery"
    second = excinfo.value.__cause__
    assert isinstance(second, json.JSONDecodeError)
    assert second.doc == '{"q": nope}'
    # The stdlib may raise the second error while handling its own internal
    # StopIteration, so the strict error is searched along the chain.
    chain = []
    node = second.__context__
    while node is not None:
        chain.append(node)
        node = node.__context__
    assert any(isinstance(n, json.JSONDecodeError) and n.doc == body for n in chain)

    with pytest.raises(ParserError) as excinfo:
        _parser().parse(_tool_completion("no json here"))
    assert str(excinfo.value) == "could not decode Action Input JSON"
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
    assert excinfo.value.__context__ is excinfo.value.__cause__


def test_prose_bracket_before_an_action_input_at_the_limit_still_recovers() -> None:
    """An unclosed bracket before a 64-level body still recovers, as on 1.10.1 (BR-019).

    The body nests 65 levels as written, so it is not decoded; it goes to
    the recovery pass instead of raising. 1.10.1 reached the same recovery
    because its strict ``json.loads`` failed on ``[note``.
    """
    body = _nested_args(MAX_TOOL_ARGS_DEPTH)
    result = _parser().parse(_tool_completion("[note " + body))
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == json.loads(body)


# ---------------------------------------------------------------------- #
# Protocol                                                               #
# ---------------------------------------------------------------------- #


def test_parser_protocol_satisfied() -> None:
    assert isinstance(_parser(), Parser)
