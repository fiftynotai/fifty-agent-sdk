"""Tests for :class:`SkillSet`: the catalog, the load tool's shape, collisions (FR-004; plan S1-S8)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import fifty_agent_sdk
import fifty_agent_sdk.skills as skills_package
from fifty_agent_sdk.errors import SkillError
from fifty_agent_sdk.skills import Skill, SkillSet
from fifty_agent_sdk.tools.protocol import Tool
from tests.skills.conftest import make_skill_dir, skill

_PREAMBLE = (
    "Skills are instructions for specific tasks. Each line below names a skill and says when "
    "to use it. Before working on a task that matches a skill, call the `load_skill` tool with "
    "the skill's name to read its instructions. A skill's instructions may mention other files "
    "in the skill; to read one, call `load_skill` again with `file` set to that file's path "
    "relative to the skill."
)


# --- S1, S2: the catalog -------------------------------------------------------------------


def test_catalog_text_is_exact_sorted_and_collapses_description_whitespace() -> None:
    """S1: the fixed preamble, a blank line, then one sorted line per skill, whitespace collapsed."""
    skills = SkillSet(
        [
            skill("pdf", "Fill in\tPDF\n forms.\r\nAlways."),
            skill("docx", "Edit Word files."),
            skill("csv-tools", "  Clean CSV data  "),
        ]
    )
    assert skills.render_catalog() == (
        f"{_PREAMBLE}\n\n"
        "- csv-tools: Clean CSV data\n"
        "- docx: Edit Word files.\n"
        "- pdf: Fill in PDF forms. Always."
    )


def test_catalog_does_not_depend_on_input_order() -> None:
    """S1: any permutation of the same skills renders the same text."""
    a, b, c = skill("a", "A."), skill("b", "B."), skill("c", "C.")
    rendered = {SkillSet(order).render_catalog() for order in ([a, b, c], [c, a, b], (b, c, a))}
    assert len(rendered) == 1


def test_catalog_keeps_non_ascii_literally() -> None:
    """S1: Arabic and emoji descriptions appear as written."""
    text = SkillSet([skill("arabic", "املأ النماذج 📄")]).render_catalog()
    assert text.endswith("- arabic: املأ النماذج 📄")


def test_an_empty_set_renders_nothing() -> None:
    """S2: no skills, no catalog text (so ``render_system_prompt`` drops the section)."""
    assert SkillSet([]).render_catalog() == ""
    assert len(SkillSet([])) == 0


# --- S3: duplicates ------------------------------------------------------------------------


def test_duplicate_names_raise_naming_both_origins(tmp_path: Path) -> None:
    """S3: two skills with one name raise ``ValueError`` naming both directories."""
    first = Skill.from_directory(make_skill_dir(tmp_path / "one", "pdf"))
    second = Skill.from_directory(make_skill_dir(tmp_path / "two", "pdf"))
    with pytest.raises(ValueError, match="Duplicate skill name 'pdf'") as caught:
        SkillSet([first, second])
    assert str(first.directory) in str(caught.value)
    assert str(second.directory) in str(caught.value)


def test_duplicate_code_built_skills_say_code() -> None:
    """S3: a skill built in code is named as ``code``."""
    with pytest.raises(ValueError, match="defined in code and in code"):
        SkillSet([skill("pdf"), skill("pdf")])


# --- S4: tool name -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "a" * 65, "load skill", "load.skill", "lädt", "a/b"])
def test_an_invalid_tool_name_raises_value_error(name: str) -> None:
    """S4: the name must match ``[A-Za-z0-9_-]{1,64}``."""
    with pytest.raises(ValueError, match="tool_name"):
        SkillSet([skill()], tool_name=name)


def test_a_non_string_tool_name_raises_type_error() -> None:
    """S4: ``tool_name`` must be a ``str``."""
    with pytest.raises(TypeError, match="tool_name"):
        SkillSet([skill()], tool_name=7)  # type: ignore[arg-type]


def test_a_custom_tool_name_reaches_the_catalog_and_the_tool() -> None:
    """S4: the catalog names the custom tool, and so does the tool's error text."""
    skills = SkillSet([skill()], tool_name="read_skill-2")
    assert skills.tool_name == "read_skill-2"
    assert skills.load_tool().name == "read_skill-2"
    assert "`read_skill-2`" in skills.render_catalog()
    assert "`load_skill`" not in skills.render_catalog()


# --- S5: the load tool's shape ----------------------------------------------------------------


def test_load_tool_is_a_tool_with_the_exact_schema_and_description() -> None:
    """S5: a ``Tool``, the same object on each call, with the pinned schema and description."""
    skills = SkillSet([skill()])
    tool = skills.load_tool()
    assert isinstance(tool, Tool)
    assert skills.load_tool() is tool
    assert tool.name == "load_skill"
    assert tool.description == (
        "Load a skill's instructions by name, or a file inside that skill. "
        "The available skills are listed in the Skills section."
    )
    assert tool.schema.model_dump() == {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill's name, exactly as listed in the Skills section.",
            },
            "file": {
                "type": "string",
                "description": (
                    "Optional. A file the skill's instructions mention, as a path relative "
                    'to the skill (for example "reference/forms.md"). Omit it to load the '
                    "skill's instructions."
                ),
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }


# --- S6, S7: accessors and types ----------------------------------------------------------------


def test_names_get_len_and_iteration_follow_name_order() -> None:
    """S6: ``names`` sorted, ``get`` by name or ``None``, ``len``, iteration in name order."""
    b, a = skill("b", "B."), skill("a", "A.")
    skills = SkillSet([b, a])
    assert skills.names == ("a", "b")
    assert skills.get("a") is a
    assert skills.get("missing") is None
    assert len(skills) == 2
    assert list(skills) == [a, b]
    assert repr(skills) == "SkillSet(names=('a', 'b'), tool_name='load_skill')"


def test_a_generator_of_skills_is_accepted() -> None:
    """S6: any iterable works; it is read once."""
    skills = SkillSet(skill(name) for name in ("x", "y"))
    assert skills.names == ("x", "y")


@pytest.mark.parametrize(
    "items",
    [["not a skill"], [skill(), {"name": "pdf"}], "pdf"],
    ids=["string_item", "dict_item", "string"],
)
def test_a_non_skill_item_raises_type_error(items: object) -> None:
    """S7: every item must be a ``Skill``."""
    with pytest.raises(TypeError, match="Skill"):
        SkillSet(items)  # type: ignore[arg-type]


def test_a_single_skill_instead_of_a_list_raises_type_error() -> None:
    """S7: ``SkillSet(skill)`` is a mistake (a pydantic model iterates as field pairs)."""
    with pytest.raises(TypeError, match=r"such as \[skill\]"):
        SkillSet(skill())  # type: ignore[arg-type]


# --- S8: the public surface ----------------------------------------------------------------------


def test_skills_public_surface() -> None:
    """S8: the four names are public at the root and in ``fifty_agent_sdk.skills``; the internals are not."""
    public = ["Skill", "SkillError", "SkillSet", "discover_skills"]
    for name in public:
        assert name in fifty_agent_sdk.__all__
        assert getattr(fifty_agent_sdk, name) is getattr(skills_package, name)
    assert sorted(skills_package.__all__) == public
    assert fifty_agent_sdk.SkillError is SkillError
    exported = set(fifty_agent_sdk.__all__) | set(skills_package.__all__)
    for private in ("_LoadSkillTool", "_frontmatter", "_files", "_load_tool", "parse_skill_md"):
        assert private not in exported


def test_the_sdk_never_imports_pyyaml(tmp_path: Path) -> None:
    """S8 (D1): importing the SDK and loading a skill from disk leaves ``yaml`` unimported."""
    make_skill_dir(tmp_path, "pdf")
    code = (
        "import sys, fifty_agent_sdk as f; "
        f"f.SkillSet(f.discover_skills({str(tmp_path)!r})).render_catalog(); "
        "print('yaml' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip().splitlines()[-1] == "False"


# --- aegis L2: messages built from host paths ------------------------------------------------------


def test_a_duplicate_name_message_escapes_surrogates_and_controls_in_directories() -> None:
    """L2: a non-UTF-8 POSIX name (surrogates) or a control character never reaches ``str(exc)`` raw."""
    odd = Skill(name="pdf", description="d", body="b", directory=Path("/srv/r\udcff\x1b[2J/pdf"))
    plain = Skill(name="pdf", description="d", body="b", directory=Path("/srv/other/pdf"))
    with pytest.raises(ValueError) as caught:
        SkillSet([odd, plain])
    message = str(caught.value)
    message.encode("utf-8")  # would raise UnicodeEncodeError on a raw surrogate
    assert "\x1b" not in message
    assert "r\\udcff\\u001b[2J" in message


def test_a_type_error_escapes_an_odd_class_name() -> None:
    """L2 sweep: a host class name may hold control characters (not surrogates: ``type()`` refuses them)."""
    odd = type("Odd\x1b[2J", (), {})
    with pytest.raises(TypeError) as caught:
        SkillSet([odd()])  # type: ignore[list-item]
    str(caught.value).encode("utf-8")
    assert "Odd\\u001b[2J" in str(caught.value) and "\x1b" not in str(caught.value)


def test_membership_tests_skill_names_and_never_raises() -> None:
    """MINOR-11: ``in`` is true for a known name, false for an unknown or non-``str`` value."""
    skills = SkillSet([skill("pdf")])
    assert "pdf" in skills
    assert "docx" not in skills
    for value in (None, 3, ["pdf"], {"pdf": 1}, b"pdf", skill("pdf")):
        assert value not in skills
