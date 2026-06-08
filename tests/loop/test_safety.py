"""Tests for ``agent_sdk.safety``: SafetyConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_sdk.safety import SafetyConfig


def test_safety_config_defaults() -> None:
    cfg = SafetyConfig()
    assert cfg.max_iterations == 10
    assert cfg.tool_timeout_seconds == 30.0
    assert cfg.fallback_message
    assert isinstance(cfg.fallback_message, str)


def test_safety_config_is_frozen() -> None:
    cfg = SafetyConfig()
    with pytest.raises(ValidationError):
        cfg.max_iterations = 5  # type: ignore[misc]


def test_safety_config_min_iterations_one_ok() -> None:
    cfg = SafetyConfig(max_iterations=1)
    assert cfg.max_iterations == 1


def test_safety_config_validation_min_iterations_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(max_iterations=0)


def test_safety_config_validation_min_iterations_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(max_iterations=-1)


def test_safety_config_validation_zero_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(tool_timeout_seconds=0.0)


def test_safety_config_validation_negative_timeout_rejected() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(tool_timeout_seconds=-1.0)


def test_safety_config_validation_non_empty_fallback() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(fallback_message="")


def test_safety_config_none_timeout_disables() -> None:
    cfg = SafetyConfig(tool_timeout_seconds=None)
    assert cfg.tool_timeout_seconds is None


def test_safety_config_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(unexpected="boom")  # type: ignore[call-arg]


def test_safety_config_custom_fallback_message() -> None:
    cfg = SafetyConfig(fallback_message="hit the cap")
    assert cfg.fallback_message == "hit the cap"


# ---------------------------------------------------------------------------
# Require-tool-before-final (BR-036)
# ---------------------------------------------------------------------------


def test_safety_config_require_tool_defaults_false() -> None:
    """The BR-036 knob is OFF by default — backward-compat for every consumer."""
    cfg = SafetyConfig()
    assert cfg.require_tool_before_final is False
    # A sensible non-empty generic default reminder is present.
    assert isinstance(cfg.tool_required_reminder, str)
    assert cfg.tool_required_reminder


def test_safety_config_require_tool_custom_round_trips() -> None:
    cfg = SafetyConfig(
        require_tool_before_final=True,
        tool_required_reminder="call a tool first",
    )
    assert cfg.require_tool_before_final is True
    assert cfg.tool_required_reminder == "call a tool first"


def test_safety_config_require_tool_empty_reminder_rejected() -> None:
    with pytest.raises(ValidationError):
        SafetyConfig(tool_required_reminder="")
