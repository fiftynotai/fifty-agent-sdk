"""Construction-time tests for ``fifty_agent_sdk.tool_mode`` through ``AgentLoop`` (FR-001).

Covers the public enum (AC-1), the legacy-path resolution (AC-3), the
owned-knob conflict rule and ``stream`` rejection (AC-4), and the
mode-appropriate retry reminder selection (AC-6). Everything is driven through
the public ``AgentLoop(tool_mode=...)`` constructor. Behaviour on the wire
(request bodies, routing, replay) lives in ``tests/loop/test_tool_mode_loop.py``.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

import fifty_agent_sdk
from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    AgentLoop,
    JsonModeParser,
    Parser,
    ParseResult,
    PromptSections,
    ProseModeParser,
    Registry,
    SafetyConfig,
    ToolMode,
    render_system_prompt,
)
from fifty_agent_sdk.parser.final_only import _FinalOnlyParser
from fifty_agent_sdk.tool_mode import (
    _NATIVE_PARSER_RETRY_REMINDER,
    _PROSE_PARSER_RETRY_REMINDER,
)
from tests.loop.conftest import FakeLLMClient, FakeTool

_PERSONA = "You are a test agent."

# --- Test doubles -----------------------------------------------------------


class _ContainedParser:
    """A consumer-style wrapper around another parser (the containment shape)."""

    def __init__(self, inner: Parser) -> None:
        self._inner = inner

    def parse(self, completion: str) -> ParseResult:
        return self._inner.parse(completion)


# --- Helpers ----------------------------------------------------------------


def _loop(
    tool_mode: ToolMode | str | None,
    *,
    parser: Parser | None = None,
    prompts: PromptSections | None = None,
    output_format: str = "",
    tool_message_role: Literal["tool", "user", "assistant"] | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    registry: Registry | None = None,
) -> AgentLoop:
    if registry is None:
        registry = Registry()
        registry.register(FakeTool("search"))
    return AgentLoop(
        llm=FakeLLMClient([]),
        registry=registry,
        parser=parser,
        prompts=prompts if prompts is not None else PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
        stream=stream,
        output_format=output_format,
        tool_message_role=tool_message_role,
        tool_mode=tool_mode,  # type: ignore[arg-type]
    )


# --- AC-1: the public enum --------------------------------------------------


def test_tool_mode_is_exported_and_is_str_enum() -> None:
    """ToolMode is public, has exactly three members, and round-trips its string values (FR-001 AC-1)."""
    assert "ToolMode" in fifty_agent_sdk.__all__
    assert fifty_agent_sdk.ToolMode is ToolMode
    assert [m.value for m in ToolMode] == ["json", "prose", "native"]
    assert ToolMode("native") is ToolMode.NATIVE
    assert ToolMode.JSON == "json"


@pytest.mark.parametrize("value", ["json", "prose", "native"])
def test_tool_mode_string_value_is_accepted(value: str) -> None:
    """A config-driven string is normalised to the enum member (FR-001 D1)."""
    loop = _loop(value)

    assert loop._tool_mode is ToolMode(value)


def test_unknown_tool_mode_string_raises() -> None:
    """An unknown mode string raises ValueError naming the valid values (FR-001 AC-4)."""
    with pytest.raises(ValueError, match=r"unknown tool_mode='xml'.*'json', 'prose', 'native'"):
        _loop("xml")


# --- Mode defaults -----------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "parser_type", "role", "output_format", "native"),
    [
        (ToolMode.JSON, JsonModeParser, "assistant", JSON_MODE_OUTPUT_FORMAT, False),
        (ToolMode.PROSE, ProseModeParser, "assistant", PROSE_MODE_OUTPUT_FORMAT, False),
        (ToolMode.NATIVE, _FinalOnlyParser, "tool", "", True),
    ],
)
def test_mode_supplies_every_owned_knob(
    mode: ToolMode,
    parser_type: type,
    role: str,
    output_format: str,
    native: bool,
) -> None:
    """With no owned knob passed, the mode supplies parser, role, format and native flags (FR-001 AC-1)."""
    loop = _loop(mode)

    assert type(loop._parser) is parser_type
    assert loop._tool_message_role == role
    assert loop._declare_native_tools is native
    assert loop._native_precedence is native
    assert loop._omit_empty_tools is native
    assert loop._echo_blank_retry is False
    tool_block = "" if native else "- search: Test fake tool: search\n  args: {}"
    assert loop._system_prompt == render_system_prompt(
        PromptSections(persona=_PERSONA, tool_descriptions=tool_block, output_format=output_format)
    )


# --- AC-3: the legacy path ---------------------------------------------------


def test_omitting_parser_and_tool_mode_raises_type_error() -> None:
    """Without tool_mode, parser= stays required, raising TypeError as 1.7.0 did (FR-001 D2)."""
    with pytest.raises(TypeError, match=r"parser=.*tool_mode="):
        _loop(None)


@pytest.mark.parametrize("native_tools_enabled", [False, True])
def test_legacy_path_resolves_to_1_7_0_values(native_tools_enabled: bool) -> None:
    """Omitting tool_mode resolves every derived value to what 1.7.0 read directly (FR-001 D2)."""
    parser = JsonModeParser()
    safety = SafetyConfig(native_tools_enabled=native_tools_enabled)

    loop = _loop(None, parser=parser, safety=safety)

    assert loop._tool_mode is None
    assert loop._parser is parser
    assert loop._tool_message_role == "tool"
    assert loop._declare_native_tools is native_tools_enabled
    assert loop._native_precedence is True
    assert loop._omit_empty_tools is False
    assert loop._parser_retry_reminder == SafetyConfig().parser_retry_reminder
    assert loop._echo_blank_retry is True


def test_resolved_role_is_on_private_attr() -> None:
    """``_tool_message_role`` holds the resolved role and ``_safety`` is the caller's object (FR-001 D13).

    Pins the private names that downstream consumer TESTS read (logged in
    MAINTAINING.md). This is not a public contract; it only keeps FR-001 from
    breaking those reach-ins for free.
    """
    safety = SafetyConfig()

    loop = _loop(ToolMode.JSON, safety=safety)

    assert loop._tool_message_role == "assistant"
    assert loop._safety is safety


# --- AC-4: owned-knob conflicts ------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "parser"),
    [
        (ToolMode.NATIVE, JsonModeParser()),
        (ToolMode.NATIVE, ProseModeParser()),
        (ToolMode.NATIVE, _ContainedParser(JsonModeParser())),
        (ToolMode.JSON, ProseModeParser()),
        (ToolMode.PROSE, JsonModeParser()),
    ],
    ids=["native-json", "native-prose", "native-custom", "json-prose", "prose-json"],
)
def test_conflicting_parser_raises(mode: ToolMode, parser: Parser) -> None:
    """A parser that would mix protocols raises at construction (FR-001 AC-4)."""
    with pytest.raises(
        ValueError, match=rf"tool_mode=ToolMode\.{mode.name} conflicts with parser="
    ):
        _loop(mode, parser=parser)


@pytest.mark.parametrize(
    ("mode", "parser"),
    [
        (ToolMode.JSON, JsonModeParser()),
        (ToolMode.JSON, _ContainedParser(JsonModeParser())),
        (ToolMode.PROSE, ProseModeParser()),
        (ToolMode.PROSE, _ContainedParser(ProseModeParser())),
    ],
    ids=["json-json", "json-contained", "prose-prose", "prose-contained"],
)
def test_compatible_parser_is_honoured(mode: ToolMode, parser: Parser) -> None:
    """A compatible parser is used by identity, never replaced (FR-001 AC-4)."""
    loop = _loop(mode, parser=parser)

    assert loop._parser is parser


@pytest.mark.parametrize(
    ("mode", "slot", "value"),
    [
        (ToolMode.NATIVE, "kwarg", JSON_MODE_OUTPUT_FORMAT),
        (ToolMode.NATIVE, "kwarg", PROSE_MODE_OUTPUT_FORMAT),
        (ToolMode.NATIVE, "prompts", JSON_MODE_OUTPUT_FORMAT),
        (ToolMode.NATIVE, "prompts", PROSE_MODE_OUTPUT_FORMAT),
        (ToolMode.NATIVE, "kwarg", JSON_MODE_OUTPUT_FORMAT + "\nAlso cite sources."),
        (ToolMode.JSON, "kwarg", PROSE_MODE_OUTPUT_FORMAT),
        (ToolMode.JSON, "prompts", PROSE_MODE_OUTPUT_FORMAT),
        (ToolMode.PROSE, "kwarg", JSON_MODE_OUTPUT_FORMAT),
        (ToolMode.PROSE, "prompts", "Be brief.\n" + JSON_MODE_OUTPUT_FORMAT),
    ],
    ids=[
        "native-json-kwarg",
        "native-prose-kwarg",
        "native-json-prompts",
        "native-prose-prompts",
        "native-json-substring",
        "json-prose-kwarg",
        "json-prose-prompts",
        "prose-json-kwarg",
        "prose-json-substring-prompts",
    ],
)
def test_conflicting_output_format_raises(mode: ToolMode, slot: str, value: str) -> None:
    """Another mode's shipped format, in either slot or as a substring, raises (FR-001 AC-4)."""
    knob = "output_format" if slot == "kwarg" else "prompts.output_format"
    kwargs: dict[str, Any] = (
        {"output_format": value}
        if slot == "kwarg"
        else {"prompts": PromptSections(persona=_PERSONA, output_format=value)}
    )
    with pytest.raises(
        ValueError, match=rf"tool_mode=ToolMode\.{mode.name} conflicts with {knob}="
    ):
        _loop(mode, **kwargs)


def test_conflict_message_summarises_a_custom_format() -> None:
    """A conflict message names the constant but never echoes the custom format text (FR-001 D3)."""
    value = JSON_MODE_OUTPUT_FORMAT + "\nSENTINEL-CUSTOM-LINE"

    with pytest.raises(ValueError) as excinfo:
        _loop(ToolMode.NATIVE, output_format=value)

    message = str(excinfo.value)
    assert "<custom format containing JSON_MODE_OUTPUT_FORMAT>" in message
    assert "SENTINEL-CUSTOM-LINE" not in message


@pytest.mark.parametrize(
    ("mode", "slot", "value"),
    [
        (ToolMode.JSON, "kwarg", "Answer in the JSON envelope. Keep answers short."),
        (ToolMode.JSON, "kwarg", JSON_MODE_OUTPUT_FORMAT + "\nKeep answers short."),
        (ToolMode.PROSE, "prompts", PROSE_MODE_OUTPUT_FORMAT + "\nKeep answers short."),
        (ToolMode.NATIVE, "kwarg", "Answer in markdown."),
        (ToolMode.NATIVE, "prompts", "Answer in markdown."),
    ],
    ids=["json-custom", "json-extended", "prose-extended", "native-kwarg", "native-prompts"],
)
def test_custom_output_format_is_honoured(mode: ToolMode, slot: str, value: str) -> None:
    """A custom format that is not another mode's artifact reaches the system prompt (FR-001 AC-4)."""
    kwargs: dict[str, Any] = (
        {"output_format": value}
        if slot == "kwarg"
        else {"prompts": PromptSections(persona=_PERSONA, output_format=value)}
    )

    loop = _loop(mode, **kwargs)

    assert f"# Output Format\n{value}" in loop._system_prompt


@pytest.mark.parametrize(
    ("mode", "role"),
    [
        (ToolMode.NATIVE, "user"),
        (ToolMode.NATIVE, "assistant"),
        (ToolMode.JSON, "tool"),
        (ToolMode.PROSE, "tool"),
    ],
)
def test_conflicting_tool_message_role_raises(
    mode: ToolMode, role: Literal["tool", "user", "assistant"]
) -> None:
    """A result role that cannot pair with the mode's assistant turn raises (FR-001 AC-4)."""
    with pytest.raises(
        ValueError,
        match=rf"tool_mode=ToolMode\.{mode.name} conflicts with tool_message_role='{role}'",
    ):
        _loop(mode, tool_message_role=role)


@pytest.mark.parametrize(
    ("mode", "role"),
    [
        (ToolMode.JSON, "user"),
        (ToolMode.JSON, "assistant"),
        (ToolMode.PROSE, "user"),
        (ToolMode.PROSE, "assistant"),
        (ToolMode.NATIVE, "tool"),
    ],
)
def test_compatible_tool_message_role_is_honoured(
    mode: ToolMode, role: Literal["tool", "user", "assistant"]
) -> None:
    """An allowed explicit role is used as given (FR-001 AC-4, D4)."""
    assert _loop(mode, tool_message_role=role)._tool_message_role == role


@pytest.mark.parametrize(
    ("mode", "safety"),
    [
        (ToolMode.JSON, SafetyConfig(native_tools_enabled=True)),
        (ToolMode.PROSE, SafetyConfig(native_tools_enabled=True)),
        (ToolMode.NATIVE, SafetyConfig(native_tools_enabled=False)),
        (ToolMode.NATIVE, SafetyConfig().model_copy(update={"native_tools_enabled": False})),
        (ToolMode.NATIVE, SafetyConfig.model_validate({"native_tools_enabled": False})),
    ],
    ids=[
        "json-true",
        "prose-true",
        "native-false-ctor",
        "native-false-copy",
        "native-false-validate",
    ],
)
def test_conflicting_native_tools_enabled_raises(mode: ToolMode, safety: SafetyConfig) -> None:
    """The legacy flag conflicts when it contradicts the mode; an explicit False counts (FR-001 AC-4)."""
    with pytest.raises(
        ValueError,
        match=rf"tool_mode=ToolMode\.{mode.name} conflicts with SafetyConfig\(native_tools_enabled=",
    ):
        _loop(mode, safety=safety)


@pytest.mark.parametrize(
    "safety",
    [SafetyConfig(), SafetyConfig(native_tools_enabled=True), SafetyConfig(max_iterations=3)],
    ids=["default", "true", "other-field-set"],
)
def test_native_mode_accepts_unset_or_true_native_tools_enabled(safety: SafetyConfig) -> None:
    """NATIVE treats the default False as unset and accepts an explicit True (FR-001 D3)."""
    assert _loop(ToolMode.NATIVE, safety=safety)._declare_native_tools is True


def test_native_mode_with_stream_raises() -> None:
    """NATIVE rejects stream=True at construction, naming tool_mode (FR-001 AC-4)."""
    with pytest.raises(ValueError, match=r"tool_mode=ToolMode\.NATIVE conflicts with stream=True"):
        _loop(ToolMode.NATIVE, stream=True)


def test_explicit_mode_leaves_the_callers_safety_config_untouched() -> None:
    """Resolution never copies or mutates the frozen SafetyConfig (FR-001 D13)."""
    safety = SafetyConfig(max_iterations=4)
    before = safety.model_dump()

    loop = _loop(ToolMode.NATIVE, safety=safety)

    assert loop._safety is safety
    assert safety.model_dump() == before
    assert safety.native_tools_enabled is False


# --- AC-6: the retry reminder -------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ToolMode.JSON, SafetyConfig().parser_retry_reminder),
        (ToolMode.PROSE, _PROSE_PARSER_RETRY_REMINDER),
        (ToolMode.NATIVE, _NATIVE_PARSER_RETRY_REMINDER),
    ],
)
def test_default_reminder_is_mode_appropriate(mode: ToolMode, expected: str) -> None:
    """Left unset, the reminder comes from the mode; only JSON keeps the JSON wording (FR-001 AC-6)."""
    loop = _loop(mode)

    assert loop._parser_retry_reminder == expected
    if mode is not ToolMode.JSON:
        assert "JSON" not in loop._parser_retry_reminder


@pytest.mark.parametrize("mode", list(ToolMode))
def test_explicit_reminder_wins_in_every_mode(mode: ToolMode) -> None:
    """A consumer-set reminder is used verbatim whatever the mode (FR-001 D11)."""
    safety = SafetyConfig(parser_retry_reminder="Custom reminder.")

    assert _loop(mode, safety=safety)._parser_retry_reminder == "Custom reminder."
