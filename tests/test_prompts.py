"""Tests for fifty_agent_sdk.prompts — template rendering and presets."""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from fifty_agent_sdk.prompts import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    PromptSections,
    json_mode_template,
    prose_mode_template,
    render_system_prompt,
)


def test_render_only_persona_contains_persona_and_no_other_headings() -> None:
    out = render_system_prompt(PromptSections(persona="You are a coding assistant."))
    assert "You are a coding assistant." in out
    assert "# Persona" in out
    assert "# Tools" not in out
    assert "# Output Format" not in out
    assert "# Additional Context" not in out


def test_render_all_slots_emits_every_section_in_order() -> None:
    sections = PromptSections(
        persona="P",
        tool_descriptions="T",
        output_format="F",
        additional_context="A",
    )
    out = render_system_prompt(sections)
    assert "# Persona" in out
    assert "# Tools" in out
    assert "# Output Format" in out
    assert "# Additional Context" in out
    # Order: persona → tools → output_format → additional_context
    assert (
        out.index("# Persona")
        < out.index("# Tools")
        < out.index("# Output Format")
        < out.index("# Additional Context")
    )
    # All bodies appear after their headings.
    assert out.index("\nP") < out.index("# Tools")
    assert out.index("\nT") < out.index("# Output Format")
    assert out.index("\nF") < out.index("# Additional Context")
    assert "\nA" in out


def test_render_skips_empty_persona_cleanly() -> None:
    out = render_system_prompt(PromptSections(persona="", tool_descriptions="just tools"))
    assert "# Persona" not in out
    assert "# Tools" in out
    assert "just tools" in out


def test_render_is_deterministic() -> None:
    sections = PromptSections(persona="P", tool_descriptions="T")
    assert render_system_prompt(sections) == render_system_prompt(sections)


def test_render_has_no_leading_or_trailing_whitespace() -> None:
    sections = PromptSections(persona="P", tool_descriptions="T")
    out = render_system_prompt(sections)
    assert out == out.strip()


def test_render_uses_single_blank_line_between_sections() -> None:
    sections = PromptSections(persona="P", tool_descriptions="T")
    out = render_system_prompt(sections)
    # Exactly two newlines between the persona body and the tools heading,
    # with no triple-newline drift.
    assert "\n\n\n" not in out
    assert "P\n\n# Tools" in out


def test_render_omits_empty_optional_sections_with_no_stray_headings() -> None:
    sections = PromptSections(persona="P", output_format="F")
    out = render_system_prompt(sections)
    assert "# Tools" not in out
    assert "# Additional Context" not in out
    # Should jump straight from persona to output format.
    assert "P\n\n# Output Format\nF" in out


def test_prompt_sections_is_frozen() -> None:
    sections = PromptSections(persona="P")
    with pytest.raises(FrozenInstanceError):
        sections.persona = "Q"  # type: ignore[misc]


def test_json_mode_template_embeds_json_output_format() -> None:
    out = json_mode_template("You are an assistant.", "list_files()")
    assert JSON_MODE_OUTPUT_FORMAT in out
    assert "You are an assistant." in out
    assert "list_files()" in out


def test_json_mode_template_without_optional_args() -> None:
    out = json_mode_template("Persona only.")
    assert JSON_MODE_OUTPUT_FORMAT in out
    assert "Persona only." in out
    assert "# Tools" not in out
    assert "# Additional Context" not in out


def test_prose_mode_template_embeds_prose_output_format() -> None:
    out = prose_mode_template(
        "You are an assistant.",
        "search(query)",
        "additional info here",
    )
    assert PROSE_MODE_OUTPUT_FORMAT in out
    assert "search(query)" in out
    assert "additional info here" in out


def test_prose_mode_template_minimal_inputs() -> None:
    out = prose_mode_template("Persona.")
    assert PROSE_MODE_OUTPUT_FORMAT in out
    assert "Persona." in out


def test_json_and_prose_templates_share_persona_position() -> None:
    json_out = json_mode_template("Same P", "Same T")
    prose_out = prose_mode_template("Same P", "Same T")
    # Persona + Tools sections come from the same render path, so the prefix
    # should be byte-identical up to the # Output Format heading.
    json_prefix = json_out.split("# Output Format")[0]
    prose_prefix = prose_out.split("# Output Format")[0]
    assert json_prefix == prose_prefix


def test_json_template_schema_aligned_with_parser_keys() -> None:
    out = json_mode_template("persona")
    # These keys are the contract the BR-005 parser will consume.
    for key in ("thought", "action", "tool_name", "tool_args", "answer"):
        assert key in out


def test_json_mode_output_format_carries_br018_strengthening() -> None:
    """Structural pin (BR-018): the strengthened body carries the schema
    keys, the Hard rules header, and the no-fences directive.

    Structural assertions only — no byte-equality on the full body so the
    text remains tunable without test churn.
    """
    body = JSON_MODE_OUTPUT_FORMAT
    for key in ("thought", "action", "tool_name", "tool_args", "answer"):
        assert key in body
    assert "Hard rules:" in body
    assert "Never wrap the JSON in ```json fences" in body


# --- FR-004: the skills slot -----------------------------------------------------------


def test_skills_section_renders_between_tools_and_output_format() -> None:
    """P1 (FR-004 D11): five slots render persona, tools, skills, output format, context."""
    out = render_system_prompt(
        PromptSections(
            persona="P",
            tool_descriptions="T",
            output_format="O",
            additional_context="C",
            skills="S",
        )
    )
    assert (
        out
        == "# Persona\nP\n\n# Tools\nT\n\n# Skills\nS\n\n# Output Format\nO\n\n# Additional Context\nC"
    )


def test_empty_skills_slot_renders_the_1_11_0_prompt() -> None:
    """P2 (FR-004): ``skills=""`` gives exactly the four-slot 1.11.0 output (a frozen literal)."""
    sections = PromptSections(
        persona="P", tool_descriptions="T", output_format="O", additional_context="C"
    )
    expected_1_11_0 = "# Persona\nP\n\n# Tools\nT\n\n# Output Format\nO\n\n# Additional Context\nC"
    assert render_system_prompt(sections) == expected_1_11_0
    assert render_system_prompt(dataclasses.replace(sections, skills="")) == expected_1_11_0


def test_skills_field_is_appended_last_and_positional_construction_still_works() -> None:
    """P3 (FR-004 D11): the field is last; four positional arguments mean what they meant."""
    assert [field.name for field in dataclasses.fields(PromptSections)][-1] == "skills"
    sections = PromptSections("P", "T", "O", "C")  # positional, as before 1.12.0
    assert (sections.persona, sections.tool_descriptions, sections.output_format) == ("P", "T", "O")
    assert (sections.additional_context, sections.skills) == ("C", "")


def test_skills_slot_without_persona_renders_cleanly() -> None:
    """P4 (FR-004): an empty persona and a skills slot give just the skills section."""
    assert render_system_prompt(PromptSections(persona="", skills="S")) == "# Skills\nS"
