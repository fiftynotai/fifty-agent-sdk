"""Tests for ``agent_sdk.streaming``: the AgentEvent discriminated union."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_sdk.streaming import (
    ActionEvent,
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ThoughtEvent,
    TokenEvent,
    ToolFailedEvent,
    ToolProgressEvent,
    ToolStartedEvent,
)
from agent_sdk.tools.protocol import ToolResult


def _ts() -> datetime:
    """Convenience fixed timestamp for round-trip tests."""
    return datetime(2026, 5, 5, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Discriminator literal defaults
# ---------------------------------------------------------------------------


def test_thought_event_constructs_without_event_type() -> None:
    ev = ThoughtEvent(sequence=0, timestamp=_ts(), text="thinking")
    assert ev.event_type == "thought"
    assert ev.text == "thinking"


def test_action_event_constructs_without_event_type() -> None:
    ev = ActionEvent(
        sequence=0, timestamp=_ts(), tool_name="search", args={"q": "x"}
    )
    assert ev.event_type == "action"
    assert ev.tool_name == "search"
    assert ev.args == {"q": "x"}


def test_tool_started_event_constructs_without_event_type() -> None:
    ev = ToolStartedEvent(
        sequence=0, timestamp=_ts(), tool_name="t", call_id="abc"
    )
    assert ev.event_type == "tool_started"


def test_tool_progress_event_constructs_without_event_type() -> None:
    ev = ToolProgressEvent(
        sequence=0,
        timestamp=_ts(),
        tool_name="t",
        call_id="abc",
        message="halfway",
    )
    assert ev.event_type == "tool_progress"


def test_observation_event_constructs_without_event_type() -> None:
    ev = ObservationEvent(
        sequence=0,
        timestamp=_ts(),
        tool_name="t",
        call_id="abc",
        result=ToolResult(output="hi"),
    )
    assert ev.event_type == "observation"


def test_tool_failed_event_constructs_without_event_type() -> None:
    ev = ToolFailedEvent(
        sequence=0,
        timestamp=_ts(),
        tool_name="t",
        call_id="abc",
        error="boom",
    )
    assert ev.event_type == "tool_failed"


def test_token_event_constructs_without_event_type() -> None:
    ev = TokenEvent(sequence=0, timestamp=_ts(), text="he")
    assert ev.event_type == "token"


def test_final_event_constructs_without_event_type() -> None:
    ev = FinalEvent(sequence=0, timestamp=_ts(), text="done")
    assert ev.event_type == "final"


def test_error_event_constructs_without_event_type() -> None:
    ev = ErrorEvent(
        sequence=0, timestamp=_ts(), error_type="X", message="m"
    )
    assert ev.event_type == "error"


# ---------------------------------------------------------------------------
# Immutability & extra-forbid
# ---------------------------------------------------------------------------


def test_thought_event_is_frozen() -> None:
    ev = ThoughtEvent(sequence=0, timestamp=_ts(), text="t")
    with pytest.raises(ValidationError):
        ev.text = "mutated"  # type: ignore[misc]


def test_action_event_is_frozen() -> None:
    ev = ActionEvent(sequence=0, timestamp=_ts(), tool_name="n", args={})
    with pytest.raises(ValidationError):
        ev.tool_name = "mutated"  # type: ignore[misc]


def test_extra_field_forbidden_thought() -> None:
    with pytest.raises(ValidationError):
        ThoughtEvent(  # type: ignore[call-arg]
            sequence=0, timestamp=_ts(), text="t", extra="boom"
        )


def test_extra_field_forbidden_error() -> None:
    with pytest.raises(ValidationError):
        ErrorEvent(  # type: ignore[call-arg]
            sequence=0,
            timestamp=_ts(),
            error_type="X",
            message="m",
            unknown="x",
        )


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_sequence_is_non_negative() -> None:
    with pytest.raises(ValidationError):
        ThoughtEvent(sequence=-1, timestamp=_ts(), text="t")


def test_sequence_can_be_zero() -> None:
    ev = ThoughtEvent(sequence=0, timestamp=_ts(), text="t")
    assert ev.sequence == 0


def test_error_event_context_defaults_to_empty_dict() -> None:
    ev = ErrorEvent(sequence=0, timestamp=_ts(), error_type="X", message="m")
    assert ev.context == {}


def test_action_event_args_defaults_to_empty_dict() -> None:
    ev = ActionEvent(sequence=0, timestamp=_ts(), tool_name="n")
    assert ev.args == {}


def test_observation_event_carries_full_tool_result() -> None:
    """ObservationEvent.result must round-trip a full ToolResult payload."""
    result = ToolResult(output={"foo": 1}, is_error=False, error=None)
    ev = ObservationEvent(
        sequence=0,
        timestamp=_ts(),
        tool_name="t",
        call_id="abc",
        result=result,
    )
    assert ev.result.output == {"foo": 1}
    assert ev.result.is_error is False
    assert ev.result.error is None


# ---------------------------------------------------------------------------
# Discriminated-union round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        ThoughtEvent(sequence=0, timestamp=_ts(), text="t"),
        ActionEvent(
            sequence=1, timestamp=_ts(), tool_name="n", args={"a": 1}
        ),
        ToolStartedEvent(
            sequence=2, timestamp=_ts(), tool_name="n", call_id="id"
        ),
        ToolProgressEvent(
            sequence=3,
            timestamp=_ts(),
            tool_name="n",
            call_id="id",
            message="step",
        ),
        ObservationEvent(
            sequence=4,
            timestamp=_ts(),
            tool_name="n",
            call_id="id",
            result=ToolResult(output="r"),
        ),
        ToolFailedEvent(
            sequence=5,
            timestamp=_ts(),
            tool_name="n",
            call_id="id",
            error="boom",
        ),
        TokenEvent(sequence=6, timestamp=_ts(), text="hi"),
        FinalEvent(sequence=7, timestamp=_ts(), text="done"),
        ErrorEvent(
            sequence=8,
            timestamp=_ts(),
            error_type="X",
            message="m",
            context={"k": "v"},
        ),
    ],
)
def test_discriminator_round_trip_for_each_event(event: Any) -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    dumped = event.model_dump()
    rebuilt = adapter.validate_python(dumped)
    assert rebuilt == event
    assert type(rebuilt) is type(event)


def test_discriminator_rejects_unknown_event_type() -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "event_type": "not_a_real_event",
                "sequence": 0,
                "timestamp": _ts(),
                "text": "x",
            }
        )


def test_discriminator_dispatches_correctly_on_event_type() -> None:
    """Sanity: setting event_type='action' picks ActionEvent over ThoughtEvent."""
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    rebuilt = adapter.validate_python(
        {
            "event_type": "action",
            "sequence": 0,
            "timestamp": _ts(),
            "tool_name": "x",
            "args": {},
        }
    )
    assert isinstance(rebuilt, ActionEvent)
