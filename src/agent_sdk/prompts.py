"""System prompt template builder.

Provides a composable :class:`PromptSections` dataclass and a renderer that
joins non-empty slots into a single system prompt string. Two pre-built
templates ship with the SDK:

* :func:`json_mode_template` — strict JSON output for the BR-005 parser.
* :func:`prose_mode_template` — classic ``Thought / Action / Final Answer``
  ReACT format.

The JSON schema baked into :data:`JSON_MODE_OUTPUT_FORMAT` is the canonical
schema the parser will consume — keep these aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

JSON_MODE_OUTPUT_FORMAT: str = """\
Respond with a single JSON object. Schema:
{
  "thought": string,           // your reasoning
  "action": "tool" | "final",  // tool to call, or final answer
  "tool_name": string|null,    // required when action == "tool"
  "tool_args": object|null,    // required when action == "tool"
  "answer": string|null        // required when action == "final"
}
Do not include any text outside the JSON object."""
"""Strict JSON output schema for the BR-005 parser."""

PROSE_MODE_OUTPUT_FORMAT: str = """\
Use the classic ReACT format. On every step output exactly:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON object of args>

When you have the final answer, instead output:
Thought: <your reasoning>
Final Answer: <answer>"""
"""Classic ``Thought / Action / Final Answer`` ReACT output format."""

_SECTION_SEPARATOR = "\n\n"
_HEADINGS: dict[str, str] = {
    "persona": "# Persona",
    "tool_descriptions": "# Tools",
    "output_format": "# Output Format",
    "additional_context": "# Additional Context",
}
_SECTION_ORDER: tuple[str, ...] = (
    "persona",
    "tool_descriptions",
    "output_format",
    "additional_context",
)


@dataclass(frozen=True)
class PromptSections:
    """Overridable slots for the system prompt.

    Frozen so a built section bundle cannot be mutated in flight; this keeps
    accidental reuse safe across multiple loop iterations.

    Attributes:
        persona: Who the assistant is and how it should behave. The only
            section that is conventionally non-empty.
        tool_descriptions: Description of the tools available to the
            assistant. Empty when the loop has no tools.
        output_format: How the assistant should structure its output.
            Typically populated from :data:`JSON_MODE_OUTPUT_FORMAT` or
            :data:`PROSE_MODE_OUTPUT_FORMAT`.
        additional_context: Any extra context — domain facts, session
            metadata, retrieved-document summaries.
    """

    persona: str
    tool_descriptions: str = ""
    output_format: str = ""
    additional_context: str = ""


def render_system_prompt(sections: PromptSections) -> str:
    """Compose a system prompt from :class:`PromptSections`.

    Sections are emitted in a fixed order:
    ``persona → tool_descriptions → output_format → additional_context``.
    Empty sections are omitted entirely — no stray headings or blank
    paragraphs leak into the output.

    The output has no leading or trailing whitespace and uses a single
    blank line between sections. Rendering is deterministic: identical
    input always produces identical output.

    Args:
        sections: The prompt slots to render. An empty ``persona`` is
            permitted and simply omits the persona section.

    Returns:
        The composed system prompt string.
    """
    blocks: list[str] = []
    for key in _SECTION_ORDER:
        body = getattr(sections, key)
        if not body:
            continue
        heading = _HEADINGS[key]
        blocks.append(f"{heading}\n{body}")
    return _SECTION_SEPARATOR.join(blocks)


def json_mode_template(
    persona: str,
    tool_descriptions: str = "",
    additional_context: str = "",
) -> str:
    """Pre-built JSON-mode system prompt.

    Uses :data:`JSON_MODE_OUTPUT_FORMAT` as the output format slot. The
    resulting schema is aligned with the BR-005 parser.

    Args:
        persona: Assistant persona description.
        tool_descriptions: Description of the available tools.
        additional_context: Optional extra context to inject.

    Returns:
        A complete system prompt enforcing JSON output.
    """
    return render_system_prompt(
        PromptSections(
            persona=persona,
            tool_descriptions=tool_descriptions,
            output_format=JSON_MODE_OUTPUT_FORMAT,
            additional_context=additional_context,
        )
    )


def prose_mode_template(
    persona: str,
    tool_descriptions: str = "",
    additional_context: str = "",
) -> str:
    """Pre-built prose-mode (classic ReACT) system prompt.

    Uses :data:`PROSE_MODE_OUTPUT_FORMAT` as the output format slot.

    Args:
        persona: Assistant persona description.
        tool_descriptions: Description of the available tools.
        additional_context: Optional extra context to inject.

    Returns:
        A complete system prompt using the classic ReACT format.
    """
    return render_system_prompt(
        PromptSections(
            persona=persona,
            tool_descriptions=tool_descriptions,
            output_format=PROSE_MODE_OUTPUT_FORMAT,
            additional_context=additional_context,
        )
    )


__all__ = [
    "JSON_MODE_OUTPUT_FORMAT",
    "PROSE_MODE_OUTPUT_FORMAT",
    "PromptSections",
    "json_mode_template",
    "prose_mode_template",
    "render_system_prompt",
]
