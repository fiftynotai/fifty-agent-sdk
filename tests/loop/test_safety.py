"""Tests for ``fifty_agent_sdk.safety``: SafetyConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fifty_agent_sdk.safety import SafetyConfig


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


# ---------------------------------------------------------------------------
# Native tools (BR-008)
# ---------------------------------------------------------------------------


def test_safety_config_native_tools_defaults_false() -> None:
    """The BR-008 opt-in knob is OFF by default — backward-compat for every consumer."""
    cfg = SafetyConfig()
    assert cfg.native_tools_enabled is False


def test_safety_config_native_tools_opt_in_round_trips() -> None:
    cfg = SafetyConfig(native_tools_enabled=True)
    assert cfg.native_tools_enabled is True


# ---------------------------------------------------------------------------
# Error fallback text (BR-021)
# ---------------------------------------------------------------------------

# The nine fields SafetyConfig had in 1.10.1, written out so an accidental
# rename or removal is caught here.
_V1_10_1_FIELDS = {
    "max_iterations",
    "tool_timeout_seconds",
    "fallback_message",
    "parser_retry_enabled",
    "parser_retry_reminder",
    "require_tool_before_final",
    "tool_required_reminder",
    "native_tools_enabled",
    "max_concurrent_tool_calls",
}

# A 1.10.1-shaped config with every field at a non-default value.
_V1_10_1_VALUES = {
    "max_iterations": 3,
    "tool_timeout_seconds": 7.5,
    "fallback_message": "Ran out of steps.",
    "parser_retry_enabled": False,
    "parser_retry_reminder": "Answer in JSON.",
    "require_tool_before_final": True,
    "tool_required_reminder": "Use a tool.",
    "native_tools_enabled": True,
    "max_concurrent_tool_calls": 4,
}

# The same config as a frozen JSON literal, as a 1.10.1 consumer would store it.
_V1_10_1_JSON = (
    '{"max_iterations": 3, "tool_timeout_seconds": 7.5, '
    '"fallback_message": "Ran out of steps.", "parser_retry_enabled": false, '
    '"parser_retry_reminder": "Answer in JSON.", "require_tool_before_final": true, '
    '"tool_required_reminder": "Use a tool.", "native_tools_enabled": true, '
    '"max_concurrent_tool_calls": 4}'
)


def test_safety_config_error_fallback_default() -> None:
    """The error text defaults to a fixed sentence that is not the step-limit text (BR-021)."""
    cfg = SafetyConfig()
    assert cfg.error_fallback_message == "Something went wrong while answering. Please try again."
    assert cfg.error_fallback_message != cfg.fallback_message


def test_safety_config_validation_non_empty_error_fallback() -> None:
    """An empty error_fallback_message is rejected, like an empty fallback_message (BR-021)."""
    with pytest.raises(ValidationError):
        SafetyConfig(error_fallback_message="")


@pytest.mark.parametrize("how", ["kwargs", "model_validate", "model_validate_json"])
def test_safety_config_1_10_1_shaped_config_validates_unchanged(how: str) -> None:
    """A config written for 1.10.1 validates unchanged and keeps every value (BR-021 AC-4).

    The new field takes its default and is not counted as set.
    """
    assert set(_V1_10_1_VALUES) == _V1_10_1_FIELDS
    if how == "kwargs":
        cfg = SafetyConfig(**_V1_10_1_VALUES)  # type: ignore[arg-type]
    elif how == "model_validate":
        cfg = SafetyConfig.model_validate(_V1_10_1_VALUES)
    else:
        cfg = SafetyConfig.model_validate_json(_V1_10_1_JSON)

    for name, value in _V1_10_1_VALUES.items():
        assert getattr(cfg, name) == value
    assert cfg.error_fallback_message == SafetyConfig().error_fallback_message
    assert "error_fallback_message" not in cfg.model_fields_set


def test_safety_config_adds_exactly_one_field_to_1_10_1() -> None:
    """BR-021 adds error_fallback_message and changes no other field name."""
    assert set(SafetyConfig.model_fields) == _V1_10_1_FIELDS | {"error_fallback_message"}


def test_safety_config_round_trips_through_model_dump() -> None:
    """A config with both texts customised survives model_dump and model_validate (BR-021)."""
    cfg = SafetyConfig(fallback_message="cap", error_fallback_message="error")
    assert SafetyConfig.model_validate(cfg.model_dump()) == cfg
    assert cfg.model_dump()["error_fallback_message"] == "error"
