"""Tests for the fifty_agent_sdk.errors hierarchy."""

from __future__ import annotations

import pytest

from fifty_agent_sdk.errors import (
    AgentSdkError,
    LLMError,
    MaxIterationsExceeded,
    ParserError,
    StateStoreError,
    ToolNotFound,
    ToolTimeout,
)


def test_agent_sdk_error_stores_message_and_context() -> None:
    err = AgentSdkError("boom", context={"foo": "bar"})
    assert err.message == "boom"
    assert err.context == {"foo": "bar"}
    assert str(err) == "boom"


def test_agent_sdk_error_default_context_is_empty_dict() -> None:
    err = AgentSdkError("boom")
    assert err.context == {}
    assert isinstance(err.context, dict)


def test_agent_sdk_error_explicit_none_context_becomes_empty_dict() -> None:
    err = AgentSdkError("boom", context=None)
    assert err.context == {}
    assert isinstance(err.context, dict)


def test_agent_sdk_error_repr_is_deterministic() -> None:
    err = AgentSdkError("boom", context={"foo": "bar"})
    r = repr(err)
    assert r == "AgentSdkError('boom', context={'foo': 'bar'})"


def test_agent_sdk_error_context_is_mutable_after_construction() -> None:
    err = AgentSdkError("boom")
    err.context["added"] = 1
    assert err.context == {"added": 1}


@pytest.mark.parametrize(
    "cls",
    [
        LLMError,
        ToolNotFound,
        ToolTimeout,
        MaxIterationsExceeded,
        ParserError,
        StateStoreError,
    ],
)
def test_subclasses_inherit_from_agent_sdk_error(cls: type[AgentSdkError]) -> None:
    """Every subclass is catchable as both itself and the base class."""
    with pytest.raises(cls):
        raise cls("kaboom", context={"k": "v"})

    with pytest.raises(AgentSdkError):
        raise cls("kaboom")


@pytest.mark.parametrize(
    "cls",
    [
        LLMError,
        ToolNotFound,
        ToolTimeout,
        MaxIterationsExceeded,
        ParserError,
        StateStoreError,
    ],
)
def test_subclasses_preserve_message_and_context(cls: type[AgentSdkError]) -> None:
    err = cls("oops", context={"model": "gpt-4o"})
    assert err.message == "oops"
    assert err.context == {"model": "gpt-4o"}


def test_chained_raise_from_preserves_cause() -> None:
    original = ValueError("root cause")
    try:
        try:
            raise original
        except ValueError as e:
            raise LLMError("wrapped", context={"src": "test"}) from e
    except LLMError as outer:
        assert outer.__cause__ is original
        assert outer.message == "wrapped"
        assert outer.context == {"src": "test"}


def test_repr_includes_subclass_name() -> None:
    err = ToolTimeout("slow", context={"limit_s": 5})
    assert repr(err) == "ToolTimeout('slow', context={'limit_s': 5})"
