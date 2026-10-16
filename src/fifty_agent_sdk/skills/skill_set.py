"""The :class:`SkillSet`: a catalog of skills plus the ``load_skill`` tool (FR-004).

A host builds one from :class:`~fifty_agent_sdk.skills.Skill` objects (from
:func:`~fifty_agent_sdk.skills.discover_skills`, :meth:`Skill.from_directory`
or code) and either passes it to ``AgentLoop(skills=...)`` or wires the two
primitives itself::

    skills = SkillSet(discover_skills("skills"))
    registry.register(skills.load_tool())
    prompts = PromptSections(persona=..., skills=skills.render_catalog())
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Final

from fifty_agent_sdk._model_json import escape_for_log
from fifty_agent_sdk.skills._load_tool import _LoadSkillTool
from fifty_agent_sdk.skills.model import Skill
from fifty_agent_sdk.tools.protocol import Tool

_DEFAULT_TOOL_NAME: Final = "load_skill"
"""The load tool's name unless ``SkillSet(tool_name=...)`` says otherwise."""

_TOOL_NAME_PATTERN: Final = re.compile(r"[A-Za-z0-9_-]{1,64}")
"""A tool name every OpenAI-compatible provider accepts (the function-name charset)."""


def _catalog_preamble(tool_name: str) -> str:
    """The catalog's opening paragraph, naming the load tool."""
    return (
        "Skills are instructions for specific tasks. Each line below names a skill and "
        "says when to use it. Before working on a task that matches a skill, call the "
        f"`{tool_name}` tool with the skill's name to read its instructions. A skill's "
        "instructions may mention other files in the skill; to read one, call "
        f"`{tool_name}` again with `file` set to that file's path relative to the skill."
    )


def _origin(skill: Skill) -> str:
    """Where a skill came from, for a duplicate-name message.

    The directory goes through :func:`fifty_agent_sdk._model_json.escape_for_log`:
    a POSIX directory name that is not UTF-8 decodes to surrogate code points,
    and one may hold control characters, so the raw text could make printing
    the message raise or write a terminal control sequence (aegis L2).
    """
    return escape_for_log(str(skill.directory)) if skill.directory is not None else "code"


class SkillSet:
    """An immutable set of skills, the catalog that lists them, and their load tool (FR-004).

    Progressive disclosure: only each skill's name and description go into
    the system prompt (:meth:`render_catalog`); the model calls the load tool
    (:meth:`load_tool`) to read a skill's instructions, or a file inside the
    skill, when it needs them.

    Read it with :attr:`names`, :attr:`tool_name`, :meth:`get`, ``len()``,
    iteration (name order) and ``name in skill_set``.

    The skills are snapshotted here: the bodies were read when each
    :class:`Skill` was built, and the catalog is fixed. Reference files are
    read when the model asks for one. To pick up edited skills, build a new
    ``SkillSet`` (and a new ``AgentLoop``).

    Args:
        skills: The skills. Their order does not matter: they are kept sorted
            by name (code-point order).
        tool_name: The load tool's name, ``"load_skill"`` by default. Change it
            when a host tool already has that name. It must match
            ``[A-Za-z0-9_-]{1,64}``.

    Raises:
        TypeError: When ``skills`` is a single :class:`Skill` or holds anything
            that is not one, or when ``tool_name`` is not a ``str``.
        ValueError: When two skills share a name (the message names both
            directories, or ``code``), or when ``tool_name`` does not match
            the pattern above.
    """

    def __init__(self, skills: Iterable[Skill], *, tool_name: str = _DEFAULT_TOOL_NAME) -> None:
        if isinstance(skills, Skill):
            raise TypeError("SkillSet takes an iterable of Skill objects, such as [skill]")
        items = list(skills)
        for item in items:
            if not isinstance(item, Skill):
                raise TypeError(
                    f"SkillSet takes Skill objects; got {escape_for_log(type(item).__name__)}"
                )
        if not isinstance(tool_name, str):
            raise TypeError(
                f"tool_name must be a str; got {escape_for_log(type(tool_name).__name__)}"
            )
        if _TOOL_NAME_PATTERN.fullmatch(tool_name) is None:
            raise ValueError(
                "tool_name must be 1 to 64 ASCII letters, digits, underscores or hyphens"
            )
        by_name: dict[str, Skill] = {}
        for skill in items:
            earlier = by_name.get(skill.name)
            if earlier is not None:
                raise ValueError(
                    f"Duplicate skill name {skill.name!r}: defined in {_origin(earlier)} "
                    f"and in {_origin(skill)}"
                )
            by_name[skill.name] = skill
        self._skills: tuple[Skill, ...] = tuple(sorted(items, key=lambda skill: skill.name))
        self._by_name: dict[str, Skill] = {skill.name: skill for skill in self._skills}
        self._tool_name = tool_name
        self._tool = _LoadSkillTool(tool_name, self._by_name)

    @property
    def tool_name(self) -> str:
        """The load tool's name."""
        return self._tool_name

    @property
    def names(self) -> tuple[str, ...]:
        """The skill names, sorted."""
        return tuple(skill.name for skill in self._skills)

    def get(self, name: str) -> Skill | None:
        """The skill called ``name``, or ``None``."""
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        """Whether a skill called ``name`` is in the set (``"pdf" in skills``).

        ``False`` for an unknown name and for anything that is not a ``str``
        (an unhashable value included); it never raises.
        """
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills)

    def __repr__(self) -> str:
        return f"SkillSet(names={self.names!r}, tool_name={self._tool_name!r})"

    def render_catalog(self) -> str:
        """The body of the system prompt's ``# Skills`` section, or ``""`` for no skills.

        A fixed paragraph telling the model how to use the load tool, a blank
        line, then one ``- <name>: <description>`` line per skill in name
        order, with each run of whitespace in the description (TAB, LF and CR
        included) written as one space. The heading is added by
        :func:`fifty_agent_sdk.prompts.render_system_prompt`. The same set
        always renders the same text.
        """
        if not self._skills:
            return ""
        lines = [f"- {skill.name}: {' '.join(skill.description.split())}" for skill in self._skills]
        return _catalog_preamble(self._tool_name) + "\n\n" + "\n".join(lines)

    def load_tool(self) -> Tool:
        """The load tool, the same object on every call.

        It takes ``name`` (required) and ``file`` (optional, a path relative
        to the skill). It returns the skill's instructions, or the text of a
        file inside the skill's directory, and a tool error (never a raised
        exception) for an unknown name, bad arguments or a refused file:
        absolute paths, ``..``, ``:``, hidden names, symlinks leading out of
        the directory, anything but a regular file, files over 256 KiB and
        files that are not UTF-8. Its error texts never contain a host path.
        """
        return self._tool


__all__ = ["SkillSet"]
