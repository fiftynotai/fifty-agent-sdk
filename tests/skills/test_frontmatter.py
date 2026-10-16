"""Tests for the private SKILL.md frontmatter reader (FR-004 D1, plan F1-F24).

The reader accepts a YAML subset and refuses everything else with a reason
code and a 1-based line. These tests pin each grammar rule and each refusal
by itself; ``test_frontmatter_differential.py`` checks the accepted values
against PyYAML.
"""

from __future__ import annotations

import contextlib
import time

import pytest

from fifty_agent_sdk.skills._frontmatter import _FrontmatterError, parse_skill_md


def _doc(frontmatter: str, body: str = "Body text.") -> str:
    return f"---\n{frontmatter}---\n{body}\n"


def _fields(frontmatter: str) -> tuple[str, str]:
    parsed = parse_skill_md(_doc(frontmatter))
    return parsed.name, parsed.description


def _refusal(text: str) -> _FrontmatterError:
    with pytest.raises(_FrontmatterError) as caught:
        parse_skill_md(text)
    return caught.value


# --- F1: minimal valid file ------------------------------------------------------------


def test_minimal_file_yields_name_description_frontmatter_body_and_lines() -> None:
    """F1: the two keys, the raw frontmatter, the body and each key's line are returned."""
    parsed = parse_skill_md("---\nname: pdf\ndescription: Fill PDF forms.\n---\n# Steps\n")
    assert parsed.name == "pdf"
    assert parsed.description == "Fill PDF forms."
    assert parsed.frontmatter == "name: pdf\ndescription: Fill PDF forms.\n"
    assert parsed.body == "# Steps"
    assert (parsed.name_line, parsed.description_line) == (2, 3)


# --- F2: quoted values -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("'plain text'", "plain text"),
        ("'it''s here'", "it's here"),
        ("'a \\n b'", "a \\n b"),
        ("'  padded  '", "  padded  "),
        ("'x' # a comment", "x"),
        ('"plain text"', "plain text"),
        ('"back\\\\slash"', "back\\slash"),
        ('"say \\"hi\\""', 'say "hi"'),
        ('"a\\/b"', "a/b"),
        ('"line\\nbreak"', "line\nbreak"),
        ('"tab\\there"', "tab\there"),
        ('"cr\\rhere"', "cr\rhere"),
        ('"nul\\0here"', "nul\x00here"),
        ('"\\x41\\u00e9\\U0001F600"', "Aé\U0001f600"),
        ('"no"', "no"),
        ("'123'", "123"),
        ('"x" # a comment', "x"),
    ],
)
def test_quoted_values_are_read_with_their_escapes(value: str, expected: str) -> None:
    """F2: single quotes keep text (``''`` is a quote); double quotes take the listed escapes."""
    assert _fields(f"name: pdf\ndescription: {value}\n")[1] == expected


# --- F3: block scalars -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("|", "one\ntwo\n\nthree\n"),
        ("|-", "one\ntwo\n\nthree"),
        ("|+", "one\ntwo\n\nthree\n\n"),
        (">", "one two\nthree\n"),
        (">-", "one two\nthree"),
        (">+", "one two\nthree\n\n"),
        ("| # a comment", "one\ntwo\n\nthree\n"),
    ],
)
def test_block_scalars_follow_yaml_chomping_and_folding(header: str, expected: str) -> None:
    """F3: ``|`` keeps line breaks, ``>`` folds them; ``-`` strips, ``+`` keeps trailing ones."""
    frontmatter = f"name: pdf\ndescription: {header}\n  one\n  two\n\n  three\n\n"
    assert _fields(frontmatter)[1] == expected


def test_block_scalar_keeps_leading_blank_lines_and_extra_indentation() -> None:
    """F3: leading blank lines become ``\\n``; a literal keeps lines indented past the first."""
    frontmatter = "name: pdf\ndescription: |\n\n  a\n    b\n  c\n"
    assert _fields(frontmatter)[1] == "\na\n  b\nc\n"


def test_block_scalar_holds_comment_like_and_delimiter_like_lines_as_text() -> None:
    """F3: inside a block, an indented ``# x`` or ``---`` line is content."""
    frontmatter = "name: pdf\ndescription: |\n  # not a comment\n  ---\n"
    assert _fields(frontmatter)[1] == "# not a comment\n---\n"


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: pdf\ndescription: |2\n  a\n",
        "name: pdf\ndescription: | x\n  a\n",
        "name: pdf\ndescription: |#c\n  a\n",
        "name: pdf\ndescription: >\n  a\n    b\n",
        "name: pdf\ndescription: |\n    \n  a\n",
        "name: pdf\ndescription: |\n",
        "name: pdf\ndescription: |\n# c\n  a\n",
        "name: pdf\ndescription: |\n    a\n  b\n",
    ],
    ids=[
        "indentation_indicator",
        "text_after_header",
        "comment_without_space",
        "folded_more_indented",
        "leading_blank_longer_than_content",
        "no_content",
        "column_0_comment_before_content",
        "less_indented_line_after_content",
    ],
)
def test_block_scalars_outside_the_subset_are_refused(frontmatter: str) -> None:
    """F3: block forms the subset does not cover are ``invalid_field`` on ``description``."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field) == ("invalid_field", "description")


# --- F4: folded plain continuation -----------------------------------------------------


def test_plain_value_continued_on_indented_lines_is_folded_with_spaces() -> None:
    """F4: continuation lines join with one space; inner spacing and a final comment are kept/cut."""
    frontmatter = "name: pdf\ndescription: Fill in  forms\n    when asked\n  - carefully # note\n"
    assert _fields(frontmatter)[1] == "Fill in  forms when asked - carefully"


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: pdf\ndescription: one\n\n  two\n",
        "name: pdf\ndescription: one\n  # c\n  two\n",
        "name: pdf\ndescription: one # c\n  two\n",
    ],
    ids=["blank_line", "comment_line", "inline_comment"],
)
def test_plain_continuation_after_a_blank_or_comment_is_refused(frontmatter: str) -> None:
    """F4: a continuation after a blank or comment line is ``invalid_field`` (use ``|``/``>``)."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field) == ("invalid_field", "description")


# --- F5: comments and blank lines -------------------------------------------------------


def test_comments_and_blank_lines_are_skipped() -> None:
    """F5: column-0 and indented comments, blank lines and trailing comments are ignored."""
    frontmatter = "# top\n\nname: pdf # trailing\n  # indented comment\n\ndescription: d\n# end\n"
    assert _fields(frontmatter) == ("pdf", "d")


def test_hash_without_a_space_before_it_is_text() -> None:
    """F5: ``a#b`` keeps its ``#``; only `` #`` starts a comment."""
    assert _fields("name: pdf\ndescription: C#notes and a#b\n")[1] == "C#notes and a#b"


# --- F6: keys the reader does not interpret ---------------------------------------------


def test_unknown_keys_with_nested_values_are_tolerated_and_kept_raw() -> None:
    """F6: license, metadata, allowed-tools (string or list) and blocks are skipped, not read."""
    frontmatter = (
        "name: pdf\n"
        "license: Apache-2.0\n"
        "metadata:\n"
        "  author: someone\n"
        '  version: "1.0"\n'
        "  tags: [a, 'b c', {k: v}]\n"
        "allowed-tools: Bash(git:*) Read mcp__fs__read\n"
        "compatibility:\n"
        "  - claude-code\n"
        "  - |\n"
        '    "free text\n'
        "notes: >-\n"
        "  it's 'quoted\n"
        "description: d\n"
    )
    parsed = parse_skill_md(_doc(frontmatter))
    assert (parsed.name, parsed.description) == ("pdf", "d")
    assert parsed.frontmatter == frontmatter


def test_allowed_tools_as_a_list_is_tolerated() -> None:
    """F6: ``allowed-tools`` as a YAML list (as Claude Code writes it) is skipped."""
    frontmatter = (
        "name: pdf\ndescription: d\nallowed-tools:\n  - Bash\n  - mcp__github__get_issue\n"
    )
    assert _fields(frontmatter) == ("pdf", "d")


@pytest.mark.parametrize(
    ("frontmatter", "line"),
    [
        ("tags: [a,\nname: evil]\ndescription: d\n", 2),
        ('license: "MIT\nname: evil"\ndescription: d\n', 2),
        ("license: 'MIT\nname: evil'\ndescription: d\n", 2),
        ('metadata:\n  note: "open\nname: evil"\ndescription: d\n', 3),
        ("metadata:\n  note: {a: [1,\nname: evil]}\ndescription: d\n", 3),
        ("metadata:\n  - &x 'open\nname: pdf\ndescription: d\n", 3),
    ],
    ids=["flow_sequence", "double_quote", "single_quote", "nested_quote", "nested_flow", "anchor"],
)
def test_a_value_left_open_under_an_unknown_key_is_refused(frontmatter: str, line: int) -> None:
    """F6: a quote or ``[``/``{`` opened under an uninterpreted key must close on its line.

    YAML lets those span lines regardless of indentation, so the following
    ``name:`` line would be inside them, not a key.
    """
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.line) == ("malformed_frontmatter", line)


def test_quotes_inside_plain_text_of_an_unknown_key_do_not_open_anything() -> None:
    """F6: an apostrophe inside plain text, or quotes inside a block, are text."""
    frontmatter = (
        "metadata:\n  note: rock 'n roll\n  more: |\n    \"unbalanced\nname: pdf\ndescription: d\n"
    )
    assert _fields(frontmatter) == ("pdf", "d")


def test_merge_key_cannot_inject_a_name() -> None:
    """F6: ``<<:`` is not a key line here, so ``<<: *anchor`` cannot supply ``name``."""
    error = _refusal(_doc("x: &b {name: evil}\n<<: *b\ndescription: d\n"))
    assert (error.reason, error.line) == ("malformed_frontmatter", 3)


# --- F7, F8, F23: delimiters -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Just a body.\n",
        "----\nname: pdf\n---\n",
        "...\nname: pdf\n---\n",
        " ---\nname: pdf\n---\n",
    ],
    ids=["empty", "body_only", "four_dashes", "dots", "indented"],
)
def test_text_not_starting_with_a_delimiter_has_no_frontmatter(text: str) -> None:
    """F7: line 1 must be ``---`` (trailing whitespace allowed)."""
    error = _refusal(text)
    assert (error.reason, error.line) == ("no_frontmatter", 1)


@pytest.mark.parametrize(
    "text",
    [
        "---\nname: pdf\ndescription: d\n",
        "---\nname: pdf\n----\nbody\n",
        "---\nname: pdf\n...\nbody\n",
    ],
    ids=["no_closing_line", "four_dashes", "dots"],
)
def test_frontmatter_without_a_closing_delimiter_is_unclosed(text: str) -> None:
    """F8, F23: only a ``---`` line closes the frontmatter (``----`` and ``...`` do not)."""
    error = _refusal(text)
    assert (error.reason, error.line) == ("unclosed_frontmatter", 1)


def test_delimiters_may_carry_trailing_spaces_and_tabs() -> None:
    """F23: ``--- `` and ``---\\t`` are delimiters."""
    parsed = parse_skill_md("--- \t\nname: pdf\ndescription: d\n---  \nbody\n")
    assert (parsed.name, parsed.description, parsed.body) == ("pdf", "d", "body")


# --- F9, F10, F11, F18, F21: keys ----------------------------------------------------------


@pytest.mark.parametrize(
    ("frontmatter", "field"),
    [("description: d\n", "name"), ("name: pdf\n", "description"), ("", "name")],
)
def test_a_missing_field_is_named(frontmatter: str, field: str) -> None:
    """F9, F10: a missing ``name`` or ``description`` is ``missing_field`` with the field."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field, error.line) == ("missing_field", field, None)


@pytest.mark.parametrize(
    ("frontmatter", "field", "line"),
    [
        ("name: a\ndescription: d\nname: b\n", "name", 4),
        ("name: a\nlicense: x\ndescription: d\nlicense: y\n", None, 5),
    ],
)
def test_a_repeated_key_is_refused(frontmatter: str, field: str | None, line: int) -> None:
    """F11: every top-level key may appear once (PyYAML would keep the last silently)."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field, error.line) == ("duplicate_key", field, line)


@pytest.mark.parametrize(
    "frontmatter",
    ["name: pdf\ndescription:\n", "name: pdf\ndescription:   \n", "name: pdf\ndescription: # c\n"],
    ids=["nothing", "spaces", "comment"],
)
def test_an_empty_value_is_invalid(frontmatter: str) -> None:
    """F18: ``description:`` with nothing (or only a comment) after it is ``invalid_field``."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field, error.line) == ("invalid_field", "description", 3)


def test_a_value_on_the_line_after_the_key_is_refused() -> None:
    """F18: the value must start on the key line (or be a ``|``/``>`` block)."""
    error = _refusal(_doc("name: pdf\ndescription:\n  on the next line\n"))
    assert (error.reason, error.field) == ("invalid_field", "description")


@pytest.mark.parametrize(
    ("frontmatter", "line"),
    [
        ("name: pdf\n- item\ndescription: d\n", 3),
        ("name: pdf\n? key\ndescription: d\n", 3),
        ("name: pdf\n[a]\ndescription: d\n", 3),
        ("name: pdf\njust words\ndescription: d\n", 3),
        ("name:n\ndescription: d\n", 2),
        ('"name": n\ndescription: d\n', 2),
        ("name : n\ndescription: d\n", 2),
        ("  name: pdf\ndescription: d\n", 2),
        ("%YAML 1.2\nname: pdf\ndescription: d\n", 2),
    ],
    ids=[
        "sequence_entry",
        "explicit_key",
        "flow",
        "bare_words",
        "no_space_after_colon",
        "quoted_key",
        "space_before_colon",
        "indented_first_line",
        "directive",
    ],
)
def test_a_column_0_line_that_is_not_a_key_is_malformed(frontmatter: str, line: int) -> None:
    """F21: every line starting in column 1 must be ``KEY:`` then nothing or a space."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.line) == ("malformed_frontmatter", line)


# --- F12 and refused characters ----------------------------------------------------------


@pytest.mark.parametrize(
    ("frontmatter", "line"),
    [
        ("name: pdf\ndescription: d\nmetadata:\n\tk: v\n", 5),
        ("name: pdf\ndescription: a\tb\n", 3),
        ("name: pdf\ndescription: d\nlicense: x\t# c\n", 4),
    ],
    ids=["tab_indent", "tab_in_value", "tab_in_unknown_key"],
)
def test_a_tab_anywhere_in_the_frontmatter_is_refused(frontmatter: str, line: int) -> None:
    """F12: TAB is refused in the frontmatter (YAML forbids it as indentation; loaders differ elsewhere)."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.line) == ("malformed_frontmatter", line)
    assert "TAB" in error.detail


@pytest.mark.parametrize(
    "character", ["\x7f", "\x01", "\x1b", "\x85", "\x9b", "\u2028", "\u2029", "\ufeff", "\ufffe"]
)
def test_characters_yaml_refuses_or_breaks_lines_on_are_refused(character: str) -> None:
    """Characters PyYAML's reader refuses, or treats as line breaks, are refused with their code point."""
    error = _refusal(_doc(f"name: pdf\ndescription: a{character}b\n"))
    assert (error.reason, error.line) == ("malformed_frontmatter", 3)
    assert f"U+{ord(character):04X}" in error.detail
    assert character not in error.detail


def test_a_tab_or_control_character_in_the_body_is_kept() -> None:
    """The character rule covers the frontmatter only; the body is returned as written."""
    parsed = parse_skill_md(_doc("name: pdf\ndescription: d\n", body="a\tb\x1bc"))
    assert parsed.body == "a\tb\x1bc"


# --- F13, F14, F15: plain scalars ----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "no",
        "Yes",
        "on",
        "OFF",
        "y",
        "N",
        "true",
        "fAlse",
        "~",
        "null",
        "NULL",
        "123",
        "-7",
        "0x1f",
        "017",
        "0o17",
        "0b101",
        "1_000",
        "1.5",
        "1e5",
        ".5",
        ".inf",
        "-.INF",
        ".NaN",
        "1:30",
        "2024-01-01",
        "2001-12-14t21:59:43.10-05:00",
        "<<",
        "=",
    ],
)
def test_a_plain_value_yaml_would_type_must_be_quoted(value: str) -> None:
    """F13: booleans, nulls, numbers, dates, ``<<`` and ``=`` are refused with "quote this value"."""
    error = _refusal(_doc(f"name: {value}\ndescription: d\n"))
    assert (error.reason, error.field, error.line) == ("invalid_field", "name", 2)
    assert "quote this value" in error.detail


@pytest.mark.parametrize(
    "value",
    [
        "[a]",
        "]",
        "{a}",
        "}",
        ",a",
        "&a",
        "*a",
        "!a",
        "%a",
        "@a",
        "`a",
        "- a",
        "-",
        "? a",
        "?",
        ": a",
    ],
)
def test_a_plain_value_starting_with_an_indicator_is_refused(value: str) -> None:
    """F14: YAML indicator characters cannot start a plain value."""
    error = _refusal(_doc(f"name: pdf\ndescription: {value}\n"))
    assert (error.reason, error.field) == ("invalid_field", "description")


@pytest.mark.parametrize("value", ["-x", "?x", ":x", "a-b", "x?", "http://a.b/c:d"])
def test_indicator_characters_that_yaml_reads_as_text_are_accepted(value: str) -> None:
    """F14: ``-``, ``?`` and ``:`` not followed by a space are plain text."""
    assert _fields(f"name: pdf\ndescription: {value}\n")[1] == value


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: pdf\ndescription: note: this\n",
        "name: pdf\ndescription: ends with:\n",
        "name: pdf\ndescription: one\n  two: three\n",
    ],
    ids=["colon_space", "trailing_colon", "colon_space_in_continuation"],
)
def test_a_plain_value_holding_a_mapping_indicator_is_refused(frontmatter: str) -> None:
    """F15: ``: `` or a trailing ``:`` in a plain value must be quoted."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.field) == ("invalid_field", "description")


# --- F16, F17: encoding and line endings -------------------------------------------------


def test_a_leading_bom_is_stripped() -> None:
    """F16: one leading U+FEFF is not part of the text."""
    parsed = parse_skill_md("\ufeff" + _doc("name: pdf\ndescription: d\n"))
    assert (parsed.name, parsed.description) == ("pdf", "d")


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_crlf_and_lone_cr_are_line_breaks(newline: str) -> None:
    """F17: ``\\r\\n`` and ``\\r`` read as ``\\n``, in the frontmatter, the raw text and the body."""
    text = _doc("name: pdf\ndescription: |\n  a\n  b\n", body="x\ny").replace("\n", newline)
    parsed = parse_skill_md(text)
    assert parsed.description == "a\nb\n"
    assert parsed.frontmatter == "name: pdf\ndescription: |\n  a\n  b\n"
    assert parsed.body == "x\ny"


# --- F19: escapes --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        '"\\e"',
        '"\\a"',
        '"\\N"',
        '"\\_"',
        '"\\ "',
        '"\\q"',
        '"\\x4"',
        '"\\u12"',
        '"\\uZZZZ"',
        '"\\ud800"',
        '"\\udfff"',
        '"\\U00110000"',
        '"\\U0000d800"',
        '"unclosed',
        '"tail\\"',
        "'unclosed",
        "'x' y",
        '"x"y',
    ],
)
def test_unsupported_escapes_and_unclosed_quotes_are_refused(value: str) -> None:
    """F19: only the listed escapes, never a surrogate or out-of-range code point; quotes close."""
    error = _refusal(_doc(f"name: pdf\ndescription: {value}\n"))
    assert (error.reason, error.field, error.line) == ("invalid_field", "description", 3)


def test_a_quoted_value_spanning_lines_is_refused() -> None:
    """F19: a quoted value must be on one line, even when it is closed."""
    error = _refusal(_doc("name: pdf\ndescription: 'one'\n  'two'\n"))
    assert (error.reason, error.field, error.line) == ("invalid_field", "description", 4)


# --- F20: line numbers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frontmatter", "reason", "line"),
    [
        ("# c\n\nname: no\ndescription: d\n", "invalid_field", 4),
        ("name: pdf\n\n\ndescription: [x]\n", "invalid_field", 5),
        ("name: pdf\ndescription: |\n  a\n\n    \n  b\n x\n", "invalid_field", 8),
        (
            "name: pdf\ndescription: d\nmetadata:\n  k: v\n  bad: 'open\n",
            "malformed_frontmatter",
            6,
        ),
        ("name: pdf\ndescription: d\n\n\nname: m\n", "duplicate_key", 6),
        ("name: pdf\ndescription: d\n\x7f\n", "malformed_frontmatter", 4),
    ],
)
def test_every_refusal_names_its_1_based_line(frontmatter: str, reason: str, line: int) -> None:
    """F20: the line counts from the opening ``---`` (line 1) and skips nothing."""
    error = _refusal(_doc(frontmatter))
    assert (error.reason, error.line) == (reason, line)


# --- F22: body -----------------------------------------------------------------------------


def test_body_drops_the_newlines_after_the_delimiter_and_trailing_whitespace_only() -> None:
    """F22: the newlines right after ``---`` and trailing whitespace go; the rest is verbatim."""
    parsed = parse_skill_md("---\nname: pdf\ndescription: d\n---\n\n\n  indented\n\n# H\n\t \n\n")
    assert parsed.body == "  indented\n\n# H"


@pytest.mark.parametrize("after", ["", "\n", "\n \n\t\n"])
def test_a_blank_body_is_refused(after: str) -> None:
    """F22: nothing but whitespace after the frontmatter is ``empty_body``."""
    error = _refusal(f"---\nname: pdf\ndescription: d\n---{after}")
    assert (error.reason, error.field) == ("empty_body", "body")


def test_a_later_delimiter_line_belongs_to_the_body() -> None:
    """F22: only the first ``---`` after line 1 closes the frontmatter."""
    parsed = parse_skill_md("---\nname: pdf\ndescription: d\n---\nPart one\n---\nPart two\n")
    assert parsed.body == "Part one\n---\nPart two"


def test_no_message_quotes_the_file() -> None:
    """Refusal details name the problem, never the value (a sentinel is absent)."""
    for frontmatter in (
        "name: SENTINEL: x\ndescription: d\n",
        "name: pdf\ndescription: 'SENTINEL\n",
        "name: pdf\ndescription: d\nSENTINEL\n",
        "name: pdf\ndescription: d\nk: [SENTINEL\n",
    ):
        assert "SENTINEL" not in _refusal(_doc(frontmatter)).detail


# --- F24: linear time ----------------------------------------------------------------------

_SIZE = 262_144


@pytest.mark.parametrize(
    "frontmatter",
    [
        f"name: pdf\ndescription: {'a' * _SIZE}\n",
        f"name: pdf\ndescription: {'1' * _SIZE}x\n",
        "name: pdf\ndescription: d\nmetadata:\n" + "  k: v\n" * 10_000,
        "name: pdf\ndescription: d\nk: " + "'a' " * (_SIZE // 4) + "\n",
        "name: pdf\ndescription: d\nk: " + "[" * _SIZE + "\n",
        "name: pdf\ndescription: |\n" + "  line\n" * 10_000,
        "name: pdf\ndescription: x\n" + "  more words\n" * 10_000,
        "name: pdf\ndescription: d\nk: " + '"' + "\\" * _SIZE + "\n",
    ],
    ids=[
        "long_line",
        "long_digits",
        "many_nested_lines",
        "many_quotes",
        "deep_brackets",
        "long_block",
        "long_continuation",
        "backslashes",
    ],
)
def test_adversarial_256_kib_inputs_finish_quickly(frontmatter: str) -> None:
    """F24: inputs at the size cap finish in under 2 s each (linear work, no recursion).

    The bound is generous (2 s) so a slow CI runner cannot flake; quadratic
    work on these sizes takes minutes. Accepting or refusing is not asserted.
    """
    start = time.perf_counter()
    with contextlib.suppress(_FrontmatterError):
        parse_skill_md(_doc(frontmatter))
    assert time.perf_counter() - start < 2.0
