"""Tests for agent_sdk.tools.protocol — Tool/ToolSchema/ToolResult contracts."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_sdk.tools import ToolCall as ToolsToolCall
from agent_sdk.tools.protocol import Tool, ToolResult, ToolSchema
from agent_sdk.tools.protocol import ToolCall as ProtocolToolCall

# ---------------------------------------------------------------------------
# ToolSchema
# ---------------------------------------------------------------------------


def test_tool_schema_defaults() -> None:
    schema = ToolSchema()
    assert schema.type == "object"
    assert schema.properties == {}
    assert schema.required == []
    assert schema.additionalProperties is False


def test_tool_schema_accepts_populated_fields() -> None:
    schema = ToolSchema(
        type="object",
        properties={"name": {"type": "string"}},
        required=["name"],
        additionalProperties=False,
    )
    assert schema.properties == {"name": {"type": "string"}}
    assert schema.required == ["name"]


def test_tool_schema_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolSchema(unknown_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_defaults() -> None:
    result = ToolResult()
    assert result.output is None
    assert result.is_error is False
    assert result.error is None


def test_tool_result_round_trip() -> None:
    original = ToolResult(output={"answer": 42}, is_error=False, error=None)
    dumped = original.model_dump()
    rebuilt = ToolResult.model_validate(dumped)
    assert rebuilt == original


def test_tool_result_error_path() -> None:
    result = ToolResult(output=None, is_error=True, error="ValueError: nope")
    assert result.is_error is True
    assert result.error == "ValueError: nope"
    assert result.output is None


def test_tool_result_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolResult(output=1, is_error=False, error=None, extra="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tool Protocol (runtime_checkable)
# ---------------------------------------------------------------------------


class _GoodTool:
    """A handcrafted Tool that satisfies the Protocol structurally."""

    name = "good"
    description = "A perfectly valid tool."
    schema = ToolSchema()

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output=args)


class _MissingDescription:
    name = "bad"
    schema = ToolSchema()

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(output=None)


def test_runtime_checkable_accepts_well_formed_tool() -> None:
    assert isinstance(_GoodTool(), Tool)


def test_runtime_checkable_rejects_missing_attribute() -> None:
    # runtime_checkable verifies attribute *presence* (not types). Missing
    # `description` should fail the check.
    assert not isinstance(_MissingDescription(), Tool)


def test_runtime_checkable_rejects_plain_object() -> None:
    assert not isinstance(object(), Tool)


# ---------------------------------------------------------------------------
# ToolCall re-export identity
# ---------------------------------------------------------------------------


def test_tool_call_reexport_is_same_object() -> None:
    """Guard against accidentally defining a duplicate ToolCall class."""
    from agent_sdk.llm.types import ToolCall as LlmToolCall

    assert ProtocolToolCall is LlmToolCall
    assert ToolsToolCall is LlmToolCall


def test_tool_call_round_trip_through_reexport() -> None:
    call = ProtocolToolCall(name="search", args={"query": "tea"})
    assert call.name == "search"
    assert call.args == {"query": "tea"}
