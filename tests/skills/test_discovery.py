"""Tests for :func:`discover_skills` (FR-004; plan DS1-DS7)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from fifty_agent_sdk.skills import SkillError, discover_skills
from tests.skills.conftest import make_skill_dir


def test_skills_come_back_in_name_order(tmp_path: Path) -> None:
    """DS1: immediate subdirectories are visited in code-point order, whatever was created first."""
    for name in ("zeta", "alpha", "mid-1", "mid"):
        make_skill_dir(tmp_path, name)
    assert [skill.name for skill in discover_skills(tmp_path)] == ["alpha", "mid", "mid-1", "zeta"]


def test_hidden_directories_and_plain_files_are_skipped(tmp_path: Path) -> None:
    """DS2: a name starting with ``.`` is never loaded, and files are ignored."""
    make_skill_dir(tmp_path, "pdf")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("not even valid", encoding="utf-8")
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "SKILL.md").write_text("a stray file at the root", encoding="utf-8")
    assert [skill.name for skill in discover_skills(tmp_path)] == ["pdf"]


def test_a_directory_without_skill_md_is_skipped_with_an_escaped_debug_line(tmp_path: Path) -> None:
    """DS3: no SKILL.md means skipped, logged at DEBUG with the path escaped (BR-028 rule)."""
    make_skill_dir(tmp_path, "pdf")
    odd = tmp_path / "assets\x1b[31m"
    odd.mkdir()
    with capture_logs() as logs:
        found = discover_skills(tmp_path)
    assert [skill.name for skill in found] == ["pdf"]
    skipped = [entry for entry in logs if entry["event"] == "skills.directory_skipped"]
    assert len(skipped) == 1
    assert set(skipped[0]) == {"event", "log_level", "skill_dir"}
    assert skipped[0]["skill_dir"].endswith("assets\\u001b[31m")
    assert "\x1b" not in skipped[0]["skill_dir"]
    discovered = [entry for entry in logs if entry["event"] == "skills.discovered"]
    assert [(entry["skill_count"], set(entry)) for entry in discovered] == [
        (1, {"event", "log_level", "skill_dir", "skill_count"})
    ]


def test_discovery_does_not_recurse(tmp_path: Path) -> None:
    """DS4: a skill two levels down is not found."""
    make_skill_dir(tmp_path / "group", "pdf")
    assert discover_skills(tmp_path) == []


def test_several_roots_keep_root_order(tmp_path: Path) -> None:
    """DS5: roots are scanned in the order given, each in name order."""
    make_skill_dir(tmp_path / "second", "alpha")
    make_skill_dir(tmp_path / "first", "zeta")
    make_skill_dir(tmp_path / "first", "beta")
    found = discover_skills(tmp_path / "first", str(tmp_path / "second"))
    assert [skill.name for skill in found] == ["beta", "zeta", "alpha"]


def test_no_roots_finds_nothing() -> None:
    """DS5: ``discover_skills()`` with no roots is an empty list."""
    assert discover_skills() == []


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_root_that_is_not_a_directory_raises(tmp_path: Path, kind: str) -> None:
    """DS6: a missing root is ``not_found``; a file is ``not_a_directory``."""
    root = tmp_path / "skills"
    if kind == "file":
        root.write_text("x", encoding="utf-8")
    with pytest.raises(SkillError) as caught:
        discover_skills(root)
    assert caught.value.context["reason"] == (
        "not_found" if kind == "missing" else "not_a_directory"
    )
    assert str(root) in caught.value.message


def test_a_malformed_child_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    """DS7: a SKILL.md that fails to load fails discovery, loudly."""
    make_skill_dir(tmp_path, "good")
    make_skill_dir(tmp_path, "bad", frontmatter="name: bad\ndescription: yes\n")
    with pytest.raises(SkillError) as caught:
        discover_skills(tmp_path)
    assert caught.value.context["reason"] == "invalid_field"
    assert "bad" in caught.value.context["path"]


def test_a_symlinked_skill_directory_is_followed(tmp_path: Path) -> None:
    """DS2: a symlink to a directory counts as a directory; its link name is the skill name."""
    target = make_skill_dir(
        tmp_path / "store", "elsewhere", frontmatter="name: pdf\ndescription: d\n"
    )
    root = tmp_path / "skills"
    root.mkdir()
    (root / "pdf").symlink_to(target, target_is_directory=True)
    assert [skill.name for skill in discover_skills(root)] == ["pdf"]


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can read a mode-000 directory"
)
def test_an_unreadable_child_directory_is_reported_not_skipped(tmp_path: Path) -> None:
    """DS7: a child whose SKILL.md cannot be checked raises ``unreadable`` on every interpreter.

    ``Path.exists()`` raises ``PermissionError`` here on CPython 3.11.15,
    3.12.15 and 3.13.2 and returns ``False`` on 3.14.3 (measured, FR-004
    evidence), so the SDK checks with ``stat()`` itself.
    """
    locked = make_skill_dir(tmp_path, "locked")
    locked.chmod(0)
    try:
        with pytest.raises(SkillError) as caught:
            discover_skills(tmp_path)
    finally:
        locked.chmod(0o700)
    assert caught.value.context["reason"] == "unreadable"
    assert caught.value.context["error_type"] == "PermissionError"


def test_a_root_error_message_escapes_control_characters(tmp_path: Path) -> None:
    """L3: a root path's control characters are escaped in the message, kept raw in ``context``."""
    root = tmp_path / "missing\x1b[2J\nroot"
    with pytest.raises(SkillError) as caught:
        discover_skills(root)
    assert "\x1b" not in str(caught.value) and "\n" not in str(caught.value)
    assert caught.value.context["path"] == str(root)
