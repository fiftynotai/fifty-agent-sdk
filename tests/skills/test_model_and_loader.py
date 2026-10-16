"""Tests for :class:`Skill` and :meth:`Skill.from_directory` (FR-004 D3, D10; plan L1-L12)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from fifty_agent_sdk.errors import AgentSdkError
from fifty_agent_sdk.skills import Skill, SkillError
from tests.skills.conftest import make_skill_dir, skill_md


def _load_error(path: Path) -> SkillError:
    with pytest.raises(SkillError) as caught:
        Skill.from_directory(path)
    return caught.value


# --- L1: a valid directory -----------------------------------------------------------------


def test_valid_directory_loads_every_field(tmp_path: Path) -> None:
    """L1: name, stripped description, body, resolved directory and raw frontmatter."""
    frontmatter = "name: pdf-forms\ndescription: '  Fill PDF forms.  '\nlicense: MIT\n"
    directory = make_skill_dir(tmp_path, "pdf-forms", frontmatter=frontmatter, body="Do X.")
    loaded = Skill.from_directory(directory)
    assert loaded.name == "pdf-forms"
    assert loaded.description == "Fill PDF forms."
    assert loaded.body == "Do X."
    assert loaded.directory == directory.resolve()
    assert loaded.frontmatter == frontmatter


def test_from_directory_accepts_a_string_path(tmp_path: Path) -> None:
    """L1: ``str`` and ``os.PathLike`` paths both work."""
    directory = make_skill_dir(tmp_path, "pdf")
    assert Skill.from_directory(str(directory)).name == "pdf"


# --- L2: directory and file problems -------------------------------------------------------


def test_missing_directory_is_not_found(tmp_path: Path) -> None:
    """L2: ``not_found``, naming the path."""
    error = _load_error(tmp_path / "absent")
    assert error.context["reason"] == "not_found"
    assert str(tmp_path / "absent") in error.message


def test_a_file_instead_of_a_directory_is_not_a_directory(tmp_path: Path) -> None:
    """L2: ``not_a_directory``."""
    (tmp_path / "pdf").write_text("x", encoding="utf-8")
    error = _load_error(tmp_path / "pdf")
    assert error.context["reason"] == "not_a_directory"
    assert str(tmp_path / "pdf") in error.message


def test_a_directory_without_skill_md_is_missing_skill_md(tmp_path: Path) -> None:
    """L2: ``missing_skill_md``, naming the SKILL.md path."""
    (tmp_path / "pdf").mkdir()
    error = _load_error(tmp_path / "pdf")
    assert error.context["reason"] == "missing_skill_md"
    assert str(tmp_path.resolve() / "pdf" / "SKILL.md") in error.message


def test_a_skill_md_directory_is_not_a_file(tmp_path: Path) -> None:
    """L2: ``not_a_file`` when SKILL.md is a directory."""
    (tmp_path / "pdf" / "SKILL.md").mkdir(parents=True)
    error = _load_error(tmp_path / "pdf")
    assert error.context["reason"] == "not_a_file"
    assert "SKILL.md" in error.message


def test_frontmatter_errors_name_the_file_and_line(tmp_path: Path) -> None:
    """L2: a parser refusal becomes ``SkillError`` naming SKILL.md, the line and the reason."""
    directory = make_skill_dir(tmp_path, "pdf", frontmatter="name: pdf\ndescription: no\n")
    error = _load_error(directory)
    skill_file = directory.resolve() / "SKILL.md"
    assert error.message.startswith(f"Invalid skill file {skill_file}: frontmatter line 3: ")
    assert error.context == {
        "path": str(skill_file),
        "reason": "invalid_field",
        "line": 3,
        "field": "description",
    }


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("no frontmatter\n", "no_frontmatter"),
        ("---\nname: pdf\n", "unclosed_frontmatter"),
        ("---\nname: pdf\n---\nbody\n", "missing_field"),
        ("---\nname: pdf\nname: pdf\ndescription: d\n---\nbody\n", "duplicate_key"),
        ("---\nname: pdf\n- x\ndescription: d\n---\nbody\n", "malformed_frontmatter"),
    ],
)
def test_each_parser_reason_reaches_the_skill_error(tmp_path: Path, text: str, reason: str) -> None:
    """L2: the reason code is passed through unchanged."""
    directory = tmp_path / "pdf"
    directory.mkdir()
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    assert _load_error(directory).context["reason"] == reason


# --- L3: name mismatch ---------------------------------------------------------------------


def test_name_different_from_the_directory_is_rejected(tmp_path: Path) -> None:
    """L3 (D3, enforced): the frontmatter name must equal the directory name."""
    directory = make_skill_dir(tmp_path, "pdf", frontmatter="name: docx\ndescription: d\n")
    error = _load_error(directory)
    assert error.context["reason"] == "name_mismatch"
    assert error.context["directory_name"] == "pdf"
    assert error.context["line"] == 2
    assert "'docx'" in error.message and "'pdf'" in error.message


# --- L4: size cap --------------------------------------------------------------------------

_CAP = 262_144


def _sized_skill_md(size: int) -> bytes:
    head = skill_md("pdf", "d", body="").encode("utf-8")
    return head + b"x" * (size - len(head))


def test_skill_md_at_the_cap_loads(tmp_path: Path) -> None:
    """L4: exactly 262,144 bytes is accepted."""
    directory = tmp_path / "pdf"
    directory.mkdir()
    (directory / "SKILL.md").write_bytes(_sized_skill_md(_CAP))
    assert len(Skill.from_directory(directory).body) > 0


def test_skill_md_over_the_cap_is_too_large(tmp_path: Path) -> None:
    """L4: one byte more is ``too_large``."""
    directory = tmp_path / "pdf"
    directory.mkdir()
    (directory / "SKILL.md").write_bytes(_sized_skill_md(_CAP + 1))
    assert _load_error(directory).context["reason"] == "too_large"


# --- L5: encoding --------------------------------------------------------------------------


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_non_utf8_reports_the_byte_offset(tmp_path: Path, bom: bytes) -> None:
    """L5: ``not_utf8`` with the offset of the bad byte, counted after a BOM."""
    directory = tmp_path / "pdf"
    directory.mkdir()
    good = skill_md("pdf", "d", body="ok").encode("utf-8")
    (directory / "SKILL.md").write_bytes(bom + good + b"\xff")
    error = _load_error(directory)
    assert (error.context["reason"], error.context["offset"]) == ("not_utf8", len(good))


def test_a_bom_is_stripped(tmp_path: Path) -> None:
    """L5: a UTF-8 BOM before ``---`` is not an error."""
    directory = tmp_path / "pdf"
    directory.mkdir()
    (directory / "SKILL.md").write_bytes(b"\xef\xbb\xbf" + skill_md("pdf", "d").encode("utf-8"))
    assert Skill.from_directory(directory).name == "pdf"


# --- L6, L7, L8: field rules through files ---------------------------------------------------


def test_empty_body_is_rejected(tmp_path: Path) -> None:
    """L6: ``empty_body``, field ``body``."""
    directory = make_skill_dir(tmp_path, "pdf", body="   ")
    error = _load_error(directory)
    assert (error.context["reason"], error.context["field"]) == ("empty_body", "body")


@pytest.mark.parametrize(
    "name",
    ["a" * 65, "PDF", "pdf_forms", "-pdf", "pdf-", "pdf--forms", "pdf forms", "pdf.x", "ü"],
)
def test_an_invalid_name_names_the_file_and_the_field(tmp_path: Path, name: str) -> None:
    """L7: a name outside the standard's rule is ``invalid_field`` on ``name``, naming SKILL.md."""
    directory = tmp_path / "skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        skill_md(frontmatter=f"name: '{name}'\ndescription: d\n"), encoding="utf-8"
    )
    error = _load_error(directory)
    assert (error.context["reason"], error.context["field"]) == ("invalid_field", "name")
    assert error.context["line"] == 2
    assert "SKILL.md" in error.message


@pytest.mark.parametrize(
    "description",
    ["'" + "x" * 1025 + "'", "'   '", '"has \\x1b escape"', '"has \\x7f delete"', '"\\x85 nel"'],
    ids=["too_long", "blank", "esc", "del", "c1"],
)
def test_an_invalid_description_names_the_field(tmp_path: Path, description: str) -> None:
    """L8: over 1024 characters, blank, or holding a control character other than TAB/LF/CR."""
    directory = make_skill_dir(
        tmp_path, "pdf", frontmatter=f"name: pdf\ndescription: {description}\n"
    )
    error = _load_error(directory)
    assert (error.context["reason"], error.context["field"]) == ("invalid_field", "description")
    assert error.context["line"] == 3


def test_a_description_of_1024_characters_with_tab_and_newline_escapes_is_accepted(
    tmp_path: Path,
) -> None:
    """L8: the bound is inclusive, and TAB, LF and CR are allowed."""
    text = "a\tb\nc\r" + "x" * 1018
    directory = make_skill_dir(
        tmp_path,
        "pdf",
        frontmatter=f'name: pdf\ndescription: "{text.encode("unicode_escape").decode()}"\n',
    )
    assert Skill.from_directory(directory).description == text


# --- L9: direct construction -----------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "Bad", "description": "d", "body": "b"},
        {"name": "", "description": "d", "body": "b"},
        {"name": "ok", "description": "", "body": "b"},
        {"name": "ok", "description": "d\x00", "body": "b"},
        {"name": "ok", "description": "bad \ud800 text", "body": "b"},
        {"name": "ok", "description": "d", "body": "  \n "},
        {"name": "ok", "description": "d", "body": "bad \udfff text"},
        {"name": "ok", "description": "d", "body": "b", "extra": 1},
        {"name": 1, "description": "d", "body": "b"},
    ],
    ids=[
        "uppercase_name",
        "empty_name",
        "empty_description",
        "nul_description",
        "surrogate_description",
        "blank_body",
        "surrogate_body",
        "extra_field",
        "int_name",
    ],
)
def test_direct_construction_with_a_bad_field_raises_validation_error(
    fields: dict[str, object],
) -> None:
    """L9: ``Skill(...)`` validates like every SDK model (pydantic ``ValidationError``)."""
    with pytest.raises(ValidationError):
        Skill(**fields)  # type: ignore[arg-type]


def test_a_code_built_skill_is_frozen_and_has_no_directory() -> None:
    """L9: frozen; ``directory`` and ``frontmatter`` default to ``None`` and ``""``."""
    built = Skill(name="pdf", description=" d ", body="b")
    assert (built.directory, built.frontmatter, built.description) == (None, "", "d")
    with pytest.raises(ValidationError):
        built.name = "other"  # type: ignore[misc]


def test_a_code_built_directory_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L9: a relative ``directory`` is stored resolved, so containment compares absolute paths."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pdf").mkdir()
    built = Skill(name="pdf", description="d", body="b", directory=Path("pdf"))
    assert built.directory == (tmp_path / "pdf").resolve()


# --- L10, L11: the error type ----------------------------------------------------------------


def test_skill_error_is_an_agent_sdk_error_with_context_in_repr(tmp_path: Path) -> None:
    """L10: ``except AgentSdkError`` catches it, and ``repr`` shows the context."""
    error = _load_error(tmp_path / "absent")
    assert isinstance(error, AgentSdkError)
    assert "'reason': 'not_found'" in repr(error)


def test_a_field_error_is_chained_from_the_validation_error(tmp_path: Path) -> None:
    """L11: the pydantic error is the ``__cause__``."""
    directory = make_skill_dir(tmp_path, "pdf", frontmatter='name: pdf\ndescription: "\\x1b"\n')
    error = _load_error(directory)
    assert isinstance(error.__cause__, ValidationError)


def test_a_field_error_message_does_not_quote_the_value(tmp_path: Path) -> None:
    """L11: the message names the rule, not the description text."""
    long_text = "SENTINEL" + "x" * 1100
    directory = make_skill_dir(
        tmp_path, "pdf", frontmatter=f"name: pdf\ndescription: {long_text}\n"
    )
    assert "SENTINEL" not in _load_error(directory).message


# --- L12: symlinked skill directory -----------------------------------------------------------


def test_a_symlinked_directory_is_checked_against_the_link_name(tmp_path: Path) -> None:
    """L12: the name must equal the given path's last component; ``directory`` is the target."""
    target = make_skill_dir(tmp_path / "store", "real", frontmatter="name: pdf\ndescription: d\n")
    link = tmp_path / "pdf"
    link.symlink_to(target, target_is_directory=True)
    loaded = Skill.from_directory(link)
    assert loaded.name == "pdf"
    assert loaded.directory == target.resolve()
    assert _load_error(target).context["reason"] == "name_mismatch"


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can read a mode-000 file"
)
def test_an_unreadable_skill_md_is_unreadable_with_the_error_class(tmp_path: Path) -> None:
    """L2: an ``OSError`` reading SKILL.md is ``unreadable``, naming the class, never ``str(exc)``."""
    directory = make_skill_dir(tmp_path, "pdf")
    skill_file = directory / "SKILL.md"
    skill_file.chmod(0)
    try:
        error = _load_error(directory)
    finally:
        skill_file.chmod(0o600)
    assert (error.context["reason"], error.context["error_type"]) == (
        "unreadable",
        "PermissionError",
    )
    assert error.message.endswith("it could not be read (PermissionError)")


# --- aegis L3: control characters in a directory name -------------------------------------------


def test_control_characters_in_a_directory_name_are_escaped_in_the_message(tmp_path: Path) -> None:
    """L3: ESC and LF from a directory name are escaped in ``str(exc)``; ``context["path"]`` is raw."""
    odd = "x\x1b[31mRED\x1b[0m\nFAKE LOG LINE"
    directory = make_skill_dir(tmp_path, odd, frontmatter="name: pdf\ndescription: d\n")
    error = _load_error(directory)
    assert error.context["reason"] == "name_mismatch"
    assert "\x1b" not in str(error) and "\n" not in str(error)
    assert "x\\u001b[31mRED\\u001b[0m\\u000aFAKE LOG LINE" in str(error)
    assert error.context["path"] == str(directory.resolve() / "SKILL.md")
