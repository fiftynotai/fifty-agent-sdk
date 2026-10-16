"""Find the skills under one or more directories (FR-004).

:func:`discover_skills` treats each immediate subdirectory of a root that
holds a ``SKILL.md`` file as a skill and loads it with
:meth:`fifty_agent_sdk.skills.Skill.from_directory`. It does not recurse.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Final

import structlog

from fifty_agent_sdk._model_json import escape_for_log
from fifty_agent_sdk.errors import SkillError
from fifty_agent_sdk.skills._files import path_exists
from fifty_agent_sdk.skills.model import _SKILL_FILE_NAME, Skill

_log: Final = structlog.get_logger(__name__)
"""DEBUG ``skills.directory_skipped`` and ``skills.discovered``; paths escaped (BR-028 rule)."""


def discover_skills(*roots: str | os.PathLike[str]) -> list[Skill]:
    """Load every skill found directly under each of ``roots`` (FR-004).

    For each root, in the order given, the immediate subdirectories are
    visited in name order (code-point order). Names starting with ``.`` are
    skipped, and symlinked directories are followed. A subdirectory holding
    a ``SKILL.md`` becomes a skill through :meth:`Skill.from_directory`, so a
    malformed one raises instead of being skipped. A subdirectory without
    one is skipped, with a DEBUG ``skills.directory_skipped`` line. There is
    no recursion. This is blocking file I/O, meant for setup time.

    Pass the result to :class:`~fifty_agent_sdk.skills.SkillSet`, which
    rejects a name found under two roots.

    Args:
        *roots: Directories that hold skill directories.

    Returns:
        The skills, root by root, each root's in name order.

    Raises:
        fifty_agent_sdk.errors.SkillError: When a root does not exist, is not
            a directory or cannot be listed (``reason`` ``not_found``,
            ``not_a_directory`` or ``unreadable``), or when a skill directory
            is invalid (any reason :meth:`Skill.from_directory` raises).
    """
    skills: list[Skill] = []
    for root in roots:
        root_path = Path(root)
        found = _discover_under(root_path)
        skills.extend(found)
        _log.debug(
            "skills.discovered",
            skill_dir=escape_for_log(str(root_path)),
            skill_count=len(found),
        )
    return skills


def _discover_under(root: Path) -> list[Skill]:
    """The skills directly under one root, in name order."""
    try:
        # Explicit stat(), as in Skill.from_directory: only "not found" means
        # absent; any other OSError (and ValueError, for a NUL) is reported.
        if not path_exists(root):
            raise _root_error(root, "not_found", "the directory does not exist")
        if not stat.S_ISDIR(root.stat().st_mode):
            raise _root_error(root, "not_a_directory", "the path is not a directory")
        with os.scandir(root) as listing:
            entries = sorted(listing, key=lambda entry: entry.name)
    except (OSError, ValueError) as exc:
        raise _root_error(
            root,
            "unreadable",
            f"the directory could not be read ({type(exc).__name__})",
            error_type=type(exc).__name__,
        ) from None
    found: list[Skill] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        candidate = Path(entry.path)
        try:
            is_dir = entry.is_dir()  # follows symlinks
        except OSError:
            is_dir = False
        if not is_dir:
            continue
        if not _may_hold_skill_md(candidate):
            _log.debug("skills.directory_skipped", skill_dir=escape_for_log(str(candidate)))
            continue
        found.append(Skill.from_directory(candidate))
    return found


def _may_hold_skill_md(directory: Path) -> bool:
    """Whether ``directory/SKILL.md`` exists, or cannot be checked.

    A directory that cannot be checked (a ``PermissionError``, say) is not
    skipped: :meth:`Skill.from_directory` then raises ``SkillError``
    (``unreadable``) for it, so the problem is reported, not hidden.
    """
    try:
        return path_exists(directory / _SKILL_FILE_NAME)
    except OSError:
        return True


def _root_error(root: Path, reason: str, detail: str, **extra: object) -> SkillError:
    """A :class:`SkillError` for a root that cannot be scanned.

    The path in the message goes through ``escape_for_log`` (surrogate code
    points and C0/C1 control characters escaped), as in
    :meth:`Skill.from_directory`; ``context["path"]`` keeps the path as it is.
    """
    context: dict[str, object] = {"path": str(root), "reason": reason, **extra}
    return SkillError(f"Invalid skills root {escape_for_log(str(root))}: {detail}", context=context)


__all__ = ["discover_skills"]
