"""Tests for ``fifty_agent_sdk.parser.base``: ParseResult union + Parser Protocol."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from fifty_agent_sdk.llm.types import ToolCall
from fifty_agent_sdk.parser import FinalAnswer, Parser, ParseResult, ThoughtAction


def test_thought_action_constructs_from_tool_call() -> None:
    ta = ThoughtAction(
        thought="I should look this up.",
        tool_call=ToolCall(name="search", args={"q": "x"}),
    )
    assert ta.kind == "thought_action"
    assert ta.thought == "I should look this up."
    assert ta.tool_call.name == "search"
    assert ta.tool_call.args == {"q": "x"}


def test_final_answer_constructs() -> None:
    fa = FinalAnswer(thought="Done.", content="The answer is 42.")
    assert fa.kind == "final_answer"
    assert fa.thought == "Done."
    assert fa.content == "The answer is 42."


def test_parse_result_discriminated_union_round_trip_tool() -> None:
    adapter: TypeAdapter[ParseResult] = TypeAdapter(ParseResult)
    original = ThoughtAction(
        thought="t",
        tool_call=ToolCall(name="n", args={"a": 1}),
    )
    dumped = original.model_dump()
    rebuilt = adapter.validate_python(dumped)
    assert rebuilt == original
    assert isinstance(rebuilt, ThoughtAction)


def test_parse_result_discriminated_union_round_trip_final() -> None:
    adapter: TypeAdapter[ParseResult] = TypeAdapter(ParseResult)
    original = FinalAnswer(thought="t", content="hello")
    dumped = original.model_dump()
    rebuilt = adapter.validate_python(dumped)
    assert rebuilt == original
    assert isinstance(rebuilt, FinalAnswer)


def test_thought_action_is_frozen() -> None:
    ta = ThoughtAction(
        thought="t",
        tool_call=ToolCall(name="n"),
    )
    with pytest.raises(ValidationError):
        ta.thought = "mutated"  # type: ignore[misc]


def test_final_answer_is_frozen() -> None:
    fa = FinalAnswer(thought="t", content="c")
    with pytest.raises(ValidationError):
        fa.content = "mutated"  # type: ignore[misc]


def test_thought_action_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        ThoughtAction(  # type: ignore[call-arg]
            thought="t",
            tool_call=ToolCall(name="n"),
            unexpected="boom",
        )


def test_final_answer_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        FinalAnswer(  # type: ignore[call-arg]
            thought="t",
            content="c",
            unexpected="boom",
        )


def test_parser_protocol_runtime_checkable() -> None:
    """A trivial class with the right method shape satisfies the Protocol."""

    class _Fake:
        def parse(self, completion: str) -> ParseResult:
            return FinalAnswer(thought="t", content=completion)

    assert isinstance(_Fake(), Parser)


def test_non_parser_class_is_not_a_parser() -> None:
    class _Other:
        def parse(self, completion: str, extra: int) -> ParseResult:  # wrong arity
            raise NotImplementedError

    # runtime_checkable only checks attribute existence, not signature.
    # A class lacking `parse` entirely must fail isinstance.
    class _Empty:
        pass

    assert not isinstance(_Empty(), Parser)
