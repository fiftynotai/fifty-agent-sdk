"""Tests for ``fifty_agent_sdk.parser.final_only._FinalOnlyParser`` (FR-001 D6, D8, AC-2).

The final-only parser backs ``ToolMode.NATIVE``'s text path. Its contract:
every non-blank completion is a :class:`FinalAnswer` carrying the completion
verbatim, a blank one raises ``ParserError(error_phase="empty_completion")``,
and it can never return a tool call — even for text that looks like one.
"""

from __future__ import annotations

import json

import pytest

import fifty_agent_sdk
from fifty_agent_sdk.errors import ParserError
from fifty_agent_sdk.parser import __all__ as parser_all
from fifty_agent_sdk.parser.base import FinalAnswer, Parser
from fifty_agent_sdk.parser.final_only import _FinalOnlyParser

_JSON_TOOL_ENVELOPE = json.dumps(
    {
        "thought": "I should search",
        "action": "tool",
        "tool_name": "search",
        "tool_args": {"query": "x"},
        "answer": None,
    }
)
_PROSE_ACTION_BLOCK = 'Thought: I should search\nAction: search\nAction Input: {"query": "x"}'


def test_final_only_parser_satisfies_parser_protocol() -> None:
    """The private parser structurally satisfies the public ``Parser`` protocol."""
    assert isinstance(_FinalOnlyParser(), Parser)


def test_final_only_parser_is_not_exported() -> None:
    """The parser stays private: absent from the package root and the parser package (FR-001 D8)."""
    assert "_FinalOnlyParser" not in fifty_agent_sdk.__all__
    assert "FinalOnlyParser" not in fifty_agent_sdk.__all__
    assert "_FinalOnlyParser" not in parser_all
    assert "FinalOnlyParser" not in parser_all


@pytest.mark.parametrize(
    "completion",
    ["The answer is 42.", "  padded answer\n", "line one\nline two"],
)
def test_final_only_parser_returns_content_verbatim(completion: str) -> None:
    """Non-blank text is a FinalAnswer whose content is the completion, unstripped."""
    result = _FinalOnlyParser().parse(completion)

    assert result == FinalAnswer(thought="", content=completion)


@pytest.mark.parametrize("completion", ["", " ", "\n\t  \n"])
def test_final_only_parser_blank_raises_empty_completion(completion: str) -> None:
    """Blank input raises ParserError with the text-parser context schema (FR-001 D6)."""
    with pytest.raises(ParserError) as excinfo:
        _FinalOnlyParser().parse(completion)

    assert excinfo.value.context == {
        "parser": "FinalOnlyParser",
        "error_phase": "empty_completion",
        "completion_excerpt": "",
    }


@pytest.mark.parametrize(
    "completion",
    [_JSON_TOOL_ENVELOPE, _PROSE_ACTION_BLOCK, f"```json\n{_JSON_TOOL_ENVELOPE}\n```"],
    ids=["json_envelope", "prose_action_block", "fenced_json_envelope"],
)
def test_final_only_parser_never_returns_a_tool_call(completion: str) -> None:
    """Text shaped like a tool call is still a final answer, verbatim (FR-001 AC-2)."""
    result = _FinalOnlyParser().parse(completion)

    assert isinstance(result, FinalAnswer)
    assert result.content == completion
