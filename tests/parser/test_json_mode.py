"""Tests for ``fifty_agent_sdk.parser.json_mode.JsonModeParser``."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import pytest

from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH, json_nesting_exceeds
from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import (
    FinalAnswer,
    JsonModeParser,
    Parser,
    ThoughtAction,
)
from fifty_agent_sdk.parser import json_mode as json_mode_module
from fifty_agent_sdk.parser.json_mode import _RawEnvelope
from fifty_agent_sdk.prompts import JSON_MODE_OUTPUT_FORMAT


def _parser() -> JsonModeParser:
    return JsonModeParser()


# ---------------------------------------------------------------------- #
# Happy paths                                                            #
# ---------------------------------------------------------------------- #


def test_happy_path_tool_call() -> None:
    completion = json.dumps(
        {
            "thought": "I should search.",
            "action": "tool",
            "tool_name": "search",
            "tool_args": {"q": "x"},
            "answer": None,
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.thought == "I should search."
    assert result.tool_call.name == "search"
    assert result.tool_call.args == {"q": "x"}


def test_happy_path_final_answer() -> None:
    completion = json.dumps(
        {
            "thought": "Now I know.",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": "hello world",
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.thought == "Now I know."
    assert result.content == "hello world"


def test_tool_args_defaults_to_empty_when_null() -> None:
    completion = json.dumps(
        {
            "thought": "t",
            "action": "tool",
            "tool_name": "noop",
            "tool_args": None,
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {}


def test_tool_args_missing_key_defaults_to_empty() -> None:
    """Pydantic default + None-coalesce means a missing key is fine too."""
    completion = json.dumps(
        {
            "thought": "t",
            "action": "tool",
            "tool_name": "noop",
        }
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == {}


# ---------------------------------------------------------------------- #
# Recovery paths                                                         #
# ---------------------------------------------------------------------- #


def test_code_fence_wrapped_json_is_recovered() -> None:
    inner = json.dumps({"thought": "t", "action": "final", "answer": "ok"})
    completion = f"```json\n{inner}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "ok"


def test_code_fence_without_lang_tag_is_recovered() -> None:
    inner = json.dumps({"thought": "t", "action": "final", "answer": "ok"})
    completion = f"```\n{inner}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)


def test_extra_prose_around_json_is_recovered() -> None:
    inner = '{"thought":"t","action":"final","answer":"ok"}'
    completion = f"Sure! Here you go: {inner} -- hope that helps"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "ok"


def test_double_fence_takes_first_block() -> None:
    first = json.dumps({"thought": "a", "action": "final", "answer": "first"})
    second = json.dumps({"thought": "b", "action": "final", "answer": "second"})
    completion = f"```json\n{first}\n```\n\n```json\n{second}\n```"
    result = _parser().parse(completion)
    assert isinstance(result, FinalAnswer)
    assert result.content == "first"


# ---------------------------------------------------------------------- #
# Failure paths                                                          #
# ---------------------------------------------------------------------- #


def test_malformed_json_raises_parser_error() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("not json at all")
    ctx = excinfo.value.context
    assert ctx["parser"] == "JsonModeParser"
    assert ctx["error_phase"] == "json_decode"
    assert "completion_excerpt" in ctx


def test_recovery_attempt_still_invalid_raises_json_decode() -> None:
    # Has braces so the recovery slice triggers, but contents are not JSON.
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("{ this is not json but has braces }")
    assert excinfo.value.context["error_phase"] == "json_decode"


def test_action_tool_missing_tool_name_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "tool"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "tool_name"


def test_action_tool_empty_tool_name_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "tool", "tool_name": "", "tool_args": {}})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["missing"] == "tool_name"


def test_action_tool_whitespace_only_tool_name_raises() -> None:
    """A whitespace-only name is blank-after-strip and takes the same
    schema_validation path as an empty one (the prose parser strips the
    ``Action:`` header, so the JSON parser must match)."""
    completion = json.dumps(
        {"thought": "t", "action": "tool", "tool_name": "   \t  ", "tool_args": {}}
    )
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "tool_name"


def test_action_tool_padded_tool_name_is_stripped() -> None:
    """Surrounding whitespace is stripped so the registry sees the same clean
    name the prose parser would emit."""
    completion = json.dumps(
        {"thought": "t", "action": "tool", "tool_name": "  search  ", "tool_args": {}}
    )
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"


def test_action_final_missing_answer_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "final"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "schema_validation"
    assert ctx["missing"] == "answer"


def test_unknown_action_value_raises() -> None:
    completion = json.dumps({"thought": "t", "action": "banana", "answer": "x"})
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "schema_validation"


def test_extra_top_level_field_raises_schema_error() -> None:
    completion = json.dumps(
        {
            "thought": "t",
            "action": "final",
            "answer": "a",
            "junk": 1,
        }
    )
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert excinfo.value.context["error_phase"] == "schema_validation"


def test_empty_completion_raises_with_empty_completion_phase() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_whitespace_only_completion_raises_with_empty_completion_phase() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("   \n\t  ")
    assert excinfo.value.context["error_phase"] == "empty_completion"


def test_parser_error_context_excerpt_truncated() -> None:
    big = "garbage " * 100  # > 200 chars, no valid JSON
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(big)
    excerpt = excinfo.value.context["completion_excerpt"]
    assert isinstance(excerpt, str)
    assert len(excerpt) <= 200


def test_parser_error_chains_cause_via_raise_from() -> None:
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("not json")
    assert excinfo.value.__cause__ is not None


def test_strict_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins strict-pass ``RecursionError`` translation."""
    sentinel = RecursionError("deterministic depth failure")
    calls = 0

    def raise_recursion(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise sentinel

    monkeypatch.setattr(json_mode_module.json, "loads", raise_recursion)
    completion = '{"thought":"t","action":"final","answer":"ok"}'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["parser"] == "JsonModeParser"
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 1


def test_recovery_recursion_error_is_translated_to_parser_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BR-017 deterministically pins recovery-pass ``RecursionError`` translation."""
    first = json.JSONDecodeError("strict failed", "prefix", 0)
    sentinel = RecursionError("deterministic recovery depth failure")
    errors = iter((first, sentinel))
    calls = 0

    def raise_scripted(_payload: str) -> object:
        nonlocal calls
        calls += 1
        raise next(errors)

    monkeypatch.setattr(json_mode_module.json, "loads", raise_scripted)
    completion = 'prefix {"thought":"t","action":"final","answer":"ok"} suffix'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    ctx = excinfo.value.context
    assert ctx["error_phase"] == "json_decode"
    assert "RecursionError" in str(ctx["cause"])
    assert len(str(ctx["completion_excerpt"])) <= 200
    assert excinfo.value.__cause__ is sentinel
    assert calls == 2


def test_oversized_integer_strict_decode_is_contained() -> None:
    """BR-013 contains bare ValueError from strict ``json.loads`` decoding."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(digits)
    assert str(excinfo.value) == "could not decode JSON envelope"
    assert excinfo.value.context["error_phase"] == "json_decode"
    assert len(str(excinfo.value.context["completion_excerpt"])) <= 200
    assert type(excinfo.value.__cause__) is ValueError


def test_oversized_integer_recovery_decode_is_contained() -> None:
    """BR-013 contains bare ValueError from the JSON recovery decode."""
    digits = "9" * (sys.get_int_max_str_digits() + 1)
    completion = f'prefix {{"thought":"t","action":"final","answer":{digits}}} suffix'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode JSON envelope after fence recovery"
    assert excinfo.value.context["error_phase"] == "json_decode"
    assert type(excinfo.value.__cause__) is ValueError


def test_malformed_json_preserves_decode_message_and_json_cause() -> None:
    """BR-013 leaves the common malformed-syntax contract unchanged."""
    completion = "not json"
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode JSON envelope"
    assert excinfo.value.context == {
        "parser": "JsonModeParser",
        "error_phase": "json_decode",
        "completion_excerpt": completion,
        "cause": repr(excinfo.value.__cause__),
    }
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# ---------------------------------------------------------------------- #
# Nesting limit (BR-019)                                                 #
# ---------------------------------------------------------------------- #

_DEPTH_MESSAGE = (
    "could not decode JSON envelope: nesting deeper than "
    f"{MAX_TOOL_ARGS_DEPTH + 1} levels (tool_args may nest at most {MAX_TOOL_ARGS_DEPTH})"
)


def _nested_args(depth: int) -> str:
    """``tool_args`` text nested ``depth`` levels: ``{"b":"[","a":[[...0...]]}`` (BR-019).

    The ``"["`` string gives the text one more ``[`` than levels, so the depth
    check reads past its ``str.count`` shortcut at the limit too.
    """
    return '{"b":"[","a":' + "[" * (depth - 1) + "0" + "]" * (depth - 1) + "}"


def _envelope(args_text: str) -> str:
    """A tool envelope around ``args_text``: one level more than the arguments."""
    return (
        '{"thought":"t","action":"tool","tool_name":"search","tool_args":'
        + args_text
        + ',"answer":null}'
    )


def test_tool_args_nested_at_the_limit_parse() -> None:
    """``tool_args`` nested 64 levels (the envelope 65) parse as before (BR-019)."""
    args_text = _nested_args(MAX_TOOL_ARGS_DEPTH)
    result = _parser().parse(_envelope(args_text))
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.name == "search"
    assert result.tool_call.args == json.loads(args_text)


def test_tool_args_one_past_the_limit_raise_parser_error() -> None:
    """``tool_args`` one level past the limit raise a fixed-message json_decode ParserError (BR-019)."""
    completion = _envelope(_nested_args(MAX_TOOL_ARGS_DEPTH + 1))
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context == {
        "parser": "JsonModeParser",
        "error_phase": "json_decode",
        "completion_excerpt": completion[:200],
        "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
    }
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("wrap", "strict_pass_reaches_json_loads"),
    [
        (lambda envelope: "```json\n" + envelope + "\n```", False),
        (lambda envelope: '"' + envelope, True),
    ],
    ids=["fenced", "quote_prefixed"],
)
def test_too_deep_envelope_is_refused_in_recovery(
    wrap: Callable[[str], str], strict_pass_reaches_json_loads: bool
) -> None:
    """A too-deep envelope that only the recovery pass extracts is refused there too (BR-019).

    Two routes into the recovery pass: the fenced text is itself too deep,
    so its strict pass is skipped; behind a stray quote the deep part reads
    as string text, so the strict pass decodes, fails, and recovers. The
    control is the same text at the limit, which parses.
    """
    at_limit = wrap(_envelope(_nested_args(MAX_TOOL_ARGS_DEPTH)))
    past_limit = wrap(_envelope(_nested_args(MAX_TOOL_ARGS_DEPTH + 1)))
    assert json_nesting_exceeds(past_limit.strip(), MAX_TOOL_ARGS_DEPTH + 1) is not (
        strict_pass_reaches_json_loads
    )

    assert isinstance(_parser().parse(at_limit), ThoughtAction)
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(past_limit)
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context["error_phase"] == "json_decode"
    assert excinfo.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert excinfo.value.__cause__ is None


def test_invalid_text_past_the_limit_with_no_candidate_gets_the_depth_error() -> None:
    """Invalid text whose brackets open past the limit, with no ``{``...``}`` candidate, gets the depth error (BR-019).

    ``x`` followed by 100 ``[`` fails ``json.loads`` at its first character
    and nests nothing; the recovery pass finds no candidate. Before BR-019
    it got ``could not decode JSON envelope`` with the decode error as
    ``__cause__``.
    """
    completion = "x" + "[" * 100
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context["max_tool_args_depth"] == MAX_TOOL_ARGS_DEPTH
    assert excinfo.value.__cause__ is None


def test_any_envelope_value_past_the_limit_gets_the_depth_error() -> None:
    """The check covers the whole envelope: an ``answer`` nested past 64 levels gets the depth error (BR-019).

    The schema rejects such an ``answer`` anyway; before BR-019 the envelope
    decoded and the error was ``schema_validation``. The control, an
    ``answer`` nested 64 levels (envelope 65), still gets ``schema_validation``.
    """

    def final_with_deep_answer(depth: int) -> str:
        return (
            '{"thought":"t","action":"final","tool_name":null,"tool_args":null,"answer":'
            + "[" * depth
            + "0"
            + "]" * depth
            + "}"
        )

    with pytest.raises(ParserError) as excinfo:
        _parser().parse(final_with_deep_answer(MAX_TOOL_ARGS_DEPTH))
    assert excinfo.value.context["error_phase"] == "schema_validation"

    with pytest.raises(ParserError) as excinfo:
        _parser().parse(final_with_deep_answer(MAX_TOOL_ARGS_DEPTH + 1))
    assert str(excinfo.value) == _DEPTH_MESSAGE
    assert excinfo.value.context["error_phase"] == "json_decode"


def test_invalid_deep_text_with_a_candidate_gets_the_recovery_error() -> None:
    """Invalid deep text whose recovery candidate is within the limit gets the invalid-JSON error, not the depth error (BR-019).

    ``[`` x100 then ``{"a": oops}``: the strict text is refused, the
    recovery candidate ``{"a": oops}`` passes the check and fails to
    decode. The outcome is the same as before BR-019, where the strict
    decode failed and the same candidate failed.
    """
    with pytest.raises(ParserError) as excinfo:
        _parser().parse("[" * 100 + '{"a": oops}')
    assert str(excinfo.value) == "could not decode JSON envelope after fence recovery"
    assert "max_tool_args_depth" not in excinfo.value.context
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    ("envelope", "expected"),
    [
        ('{"thought":"t","action":"final","answer":"hi"}', "final"),
        (
            '{"thought":"t","action":"tool","tool_name":"search","tool_args":{"q":1}}',
            "tool",
        ),
    ],
    ids=["final_envelope", "tool_envelope"],
)
def test_valid_but_too_deep_text_goes_to_the_recovery_pass(envelope: str, expected: str) -> None:
    """Valid JSON that is too deep is recovered to the envelope inside it (BR-019).

    The completion is an array holding an envelope and a 70-level array:
    valid JSON, 71 levels. It is refused and goes to the recovery pass,
    which extracts the envelope. 1.10.1 decoded it as it stood and raised
    ``schema_validation`` (measured on the BR-021 ``src``, whose parser
    modules are identical to 1.10.1's; BR-019 evidence, round 3), so the
    loop, with the retry enabled (as measured), took its parser retry where
    it now finishes or dispatches.
    """
    completion = "[" + envelope + ", " + "[" * 70 + "]" * 70 + "]"
    assert _tree_depth_of(completion) == 71
    result = _parser().parse(completion)
    if expected == "final":
        assert isinstance(result, FinalAnswer)
        assert result.content == "hi"
    else:
        assert isinstance(result, ThoughtAction)
        assert result.tool_call.args == {"q": 1}


def _tree_depth_of(text: str) -> int:
    """Nesting depth of a JSON text, decoded and walked iteratively."""
    deepest = 0
    stack = [(json.loads(text), 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, (dict, list)):
            depth += 1
            deepest = max(deepest, depth)
            children = node.values() if isinstance(node, dict) else node
            stack.extend((child, depth) for child in children)
    return deepest


def test_failed_decodes_keep_the_strict_error_in_their_exception_chain() -> None:
    """The recovery runs inside the strict pass's ``except`` block, so exception chaining is as before BR-019.

    A failed recovery's own decode error carries the strict pass's error in
    its ``__context__`` chain, and the no-candidate error is raised while the strict
    error is being handled. Moving the recovery out of that block (mutant MG)
    keeps every message and ``__cause__`` and drops both.
    """
    completion = 'note {"thought": nope} end'
    with pytest.raises(ParserError) as excinfo:
        _parser().parse(completion)
    assert str(excinfo.value) == "could not decode JSON envelope after fence recovery"
    second = excinfo.value.__cause__
    assert isinstance(second, json.JSONDecodeError)
    assert second.doc == '{"thought": nope}'
    # The stdlib may raise the second error while handling its own internal
    # StopIteration, so the strict error is searched along the chain.
    chain = []
    node = second.__context__
    while node is not None:
        chain.append(node)
        node = node.__context__
    assert any(isinstance(n, json.JSONDecodeError) and n.doc == completion for n in chain)

    with pytest.raises(ParserError) as excinfo:
        _parser().parse("no envelope here")
    assert str(excinfo.value) == "could not decode JSON envelope"
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
    assert excinfo.value.__context__ is excinfo.value.__cause__


def test_prose_bracket_before_an_envelope_at_the_limit_still_recovers() -> None:
    """An unclosed bracket before a 65-level envelope still recovers, as on 1.10.1 (BR-019).

    The strict text nests 66 levels, so it is not decoded; it goes to the
    recovery pass instead of raising. 1.10.1 reached the same recovery
    because its strict ``json.loads`` failed on ``[draft``.
    """
    args_text = _nested_args(MAX_TOOL_ARGS_DEPTH)
    completion = "[draft " + _envelope(args_text)
    result = _parser().parse(completion)
    assert isinstance(result, ThoughtAction)
    assert result.tool_call.args == json.loads(args_text)


# ---------------------------------------------------------------------- #
# Protocol / cross-brief contract                                        #
# ---------------------------------------------------------------------- #


def test_parser_protocol_satisfied() -> None:
    assert isinstance(_parser(), Parser)


def test_json_mode_parser_consumes_keys_taught_by_prompt() -> None:
    """Mirror of the prompts-side pin test.

    Every JSON envelope key advertised by JSON_MODE_OUTPUT_FORMAT must be a
    field on the parser's internal validator. Drift on either side breaks
    the cross-brief contract.
    """
    fields = set(_RawEnvelope.model_fields.keys())
    for key in ("thought", "action", "tool_name", "tool_args", "answer"):
        assert key in fields, f"parser missing field for prompt key: {key}"
        assert key in JSON_MODE_OUTPUT_FORMAT
