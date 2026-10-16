"""The :class:`Skill` model and the single-directory loader (FR-004).

A skill is a directory holding a ``SKILL.md`` file: YAML frontmatter with a
``name`` and a ``description``, then Markdown instructions (the body). Other
files in the directory are reference files the instructions may mention;
:class:`~fifty_agent_sdk.skills.SkillSet`'s ``load_skill`` tool reads them on
request.

:meth:`Skill.from_directory` reads and checks ``SKILL.md`` eagerly, so a
malformed file fails when the skill is loaded, not in the middle of a run. A
:class:`Skill` can also be built in code, with no directory (it then has no
reference files).
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from fifty_agent_sdk._model_json import escape_for_log
from fifty_agent_sdk.errors import SkillError
from fifty_agent_sdk.skills._files import (
    MAX_SKILL_FILE_BYTES,
    _FileProblem,
    path_exists,
    read_text_bounded,
)
from fifty_agent_sdk.skills._frontmatter import _FrontmatterError, parse_skill_md

_SKILL_FILE_NAME: Final = "SKILL.md"
"""The file a skill directory must hold."""

_NAME_PATTERN: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
"""A skill name: lowercase letters and digits in hyphen-separated runs (FR-004 D3)."""

_MAX_NAME_CHARS: Final = 64
_MAX_DESCRIPTION_CHARS: Final = 1024

_SURROGATE: Final = re.compile("[\ud800-\udfff]")
_DISALLOWED_CONTROL: Final = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
"""Control characters (Unicode category Cc) other than TAB, LF and CR."""


class Skill(BaseModel):
    """One skill: a name, a one-line description, and instructions (FR-004).

    Frozen and ``extra="forbid"``, like the SDK's other value objects. Build
    one with :meth:`from_directory` (or :func:`~fifty_agent_sdk.skills.
    discover_skills`), or directly in code. A field that breaks the rules
    below raises pydantic's ``ValidationError`` on direct construction and
    :class:`~fifty_agent_sdk.errors.SkillError` from :meth:`from_directory`.

    Attributes:
        name: 1 to 64 characters of lowercase ASCII letters and digits in
            hyphen-separated runs (``pdf-forms``; no leading, trailing or
            doubled hyphen), as the open Agent Skills standard asks. A
            directory-loaded skill's name must equal its directory's name.
        description: What the skill does and when to use it; the only text
            the model sees before it loads the skill. Stripped; 1 to 1024
            characters; no control character other than TAB, LF and CR
            (those are shown as spaces in the catalog); no surrogate code
            point.
        body: The instructions ``load_skill`` returns. Not blank; no
            surrogate code point.
        directory: The skill directory, resolved on construction.
            :meth:`from_directory` sets it; a skill built in code may pass
            ``directory=`` too (its reference files are then read from
            there) or leave it ``None``, the default, and have no
            reference files.
        frontmatter: The raw frontmatter text of a directory-loaded skill
            (``""`` otherwise), for hosts that want keys the SDK does not
            interpret, for example ``yaml.safe_load(skill.frontmatter)``.
            The SDK interprets only ``name`` and ``description``; in
            particular ``allowed-tools`` is NOT enforced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    body: str
    directory: Path | None = None
    frontmatter: str = ""

    @field_validator("name")
    @classmethod
    def _name_follows_the_standard(cls, value: str) -> str:
        if not 1 <= len(value) <= _MAX_NAME_CHARS or _NAME_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "a skill name is 1 to 64 lowercase letters and digits in hyphen-separated "
                "runs, such as pdf-forms"
            )
        return value

    @field_validator("description")
    @classmethod
    def _description_is_one_short_text(cls, value: str) -> str:
        if _SURROGATE.search(value):
            raise ValueError("a skill description may not hold a surrogate code point")
        # Checked before stripping: str.strip() also removes some control
        # characters (U+001C-U+001F, U+0085), which must not slip through.
        if _DISALLOWED_CONTROL.search(value):
            raise ValueError(
                "a skill description may not hold a control character other than TAB, LF and CR"
            )
        value = value.strip()
        if not value:
            raise ValueError("a skill description may not be blank")
        if len(value) > _MAX_DESCRIPTION_CHARS:
            raise ValueError("a skill description is at most 1024 characters")
        return value

    @field_validator("body")
    @classmethod
    def _body_is_text(cls, value: str) -> str:
        if _SURROGATE.search(value):
            raise ValueError("a skill body may not hold a surrogate code point")
        if not value.strip():
            raise ValueError("a skill body may not be blank")
        return value

    @field_validator("directory")
    @classmethod
    def _directory_is_resolved(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        try:
            return value.resolve()
        except (OSError, RuntimeError):
            raise ValueError("the skill directory could not be resolved") from None

    @classmethod
    def from_directory(cls, path: str | os.PathLike[str]) -> Skill:
        """Load the skill in directory ``path`` from its ``SKILL.md`` (FR-004).

        ``SKILL.md`` is read now, as strict UTF-8 of at most 256 KiB, and its
        frontmatter is checked against the SDK's YAML subset (see
        :mod:`fifty_agent_sdk.skills`). This is blocking file I/O, meant for
        setup time. The ``name`` in the frontmatter must equal the
        directory's name: the last component of ``path`` as given, with
        symlinks not resolved, so a symlink named ``pdf`` may point at any
        directory whose ``SKILL.md`` says ``name: pdf``.

        Args:
            path: The skill directory.

        Returns:
            The loaded skill, with :attr:`directory` set to the resolved path.

        Raises:
            fifty_agent_sdk.errors.SkillError: For any problem with the
                directory or the file. The message names the path, and
                ``context["reason"]`` is a fixed code (see
                :class:`~fifty_agent_sdk.errors.SkillError`).
        """
        given = Path(path)
        try:
            root = given.resolve(strict=True)
        except FileNotFoundError:
            raise _skill_error(given, "not_found", "the directory does not exist") from None
        except (OSError, RuntimeError, ValueError) as exc:
            # RuntimeError: a symlink loop before CPython 3.13; ValueError: a NUL.
            raise _skill_error(
                given,
                "unreadable",
                f"the directory could not be resolved ({type(exc).__name__})",
                error_type=type(exc).__name__,
            ) from None
        skill_file = root / _SKILL_FILE_NAME
        try:
            # Explicit stat() calls: Path.exists() raises PermissionError on
            # CPython 3.11-3.13 and returns False on 3.14 (measured; see
            # _files.path_exists), so only "not found" counts as absent here and
            # every other OSError is reported the same way on every interpreter.
            if not stat.S_ISDIR(root.stat().st_mode):
                raise _skill_error(given, "not_a_directory", "the path is not a directory")
            if not path_exists(skill_file):
                raise _skill_error(
                    skill_file, "missing_skill_md", f"the directory has no {_SKILL_FILE_NAME}"
                )
            if not stat.S_ISREG(skill_file.stat().st_mode):
                raise _skill_error(skill_file, "not_a_file", "it is not a regular file")
        except OSError as exc:
            raise _skill_error(
                skill_file,
                "unreadable",
                f"it could not be checked ({type(exc).__name__})",
                error_type=type(exc).__name__,
            ) from None
        try:
            text = read_text_bounded(skill_file, MAX_SKILL_FILE_BYTES)
        except _FileProblem as problem:
            if problem.reason == "too_large":
                raise _skill_error(
                    skill_file,
                    "too_large",
                    f"it is larger than the {MAX_SKILL_FILE_BYTES}-byte limit",
                ) from None
            raise _skill_error(
                skill_file,
                "not_utf8",
                f"it is not UTF-8 text (byte offset {problem.offset})",
                offset=problem.offset,
            ) from None
        except OSError as exc:
            raise _skill_error(
                skill_file,
                "unreadable",
                f"it could not be read ({type(exc).__name__})",
                error_type=type(exc).__name__,
            ) from None
        try:
            parsed = parse_skill_md(text)
        except _FrontmatterError as problem:
            raise _skill_error(
                skill_file, problem.reason, problem.detail, line=problem.line, field=problem.field
            ) from None
        try:
            skill = cls(
                name=parsed.name,
                description=parsed.description,
                body=parsed.body,
                directory=root,
                frontmatter=parsed.frontmatter,
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            field = str(first["loc"][0]) if first["loc"] else None
            line = {"name": parsed.name_line, "description": parsed.description_line}.get(
                field or ""
            )
            # The pydantic message for a ValueError is "Value error, <ours>"; ours
            # quotes nothing from the file.
            detail = str(first["msg"]).removeprefix("Value error, ")
            raise _skill_error(
                skill_file, "invalid_field", f"{field}: {detail}", line=line, field=field
            ) from exc
        directory_name = given.absolute().name
        if skill.name != directory_name:
            raise _skill_error(
                skill_file,
                "name_mismatch",
                # repr() escapes every unprintable character, surrogates included.
                f"name: the skill is named {skill.name!r} but its directory is named "
                f"{directory_name!r}; they must be equal",
                line=parsed.name_line,
                field="name",
                directory_name=directory_name,
            )
        return skill


def _skill_error(
    path: Path,
    reason: str,
    detail: str,
    *,
    line: int | None = None,
    field: str | None = None,
    **extra: object,
) -> SkillError:
    """Build the :class:`SkillError` for ``path``: the message names it, ``context`` holds the codes.

    The path in the message goes through
    :func:`fifty_agent_sdk._model_json.escape_for_log`: each surrogate code
    point (a POSIX file name that is not UTF-8 decodes to them) and each C0/C1
    control character is written as a six-character escape, so printing the
    message cannot raise ``UnicodeEncodeError`` or write a terminal control
    sequence (aegis L3). ``context["path"]`` keeps the path as it is.
    """
    kind = "skill file" if path.name == _SKILL_FILE_NAME else "skill directory"
    where = f"frontmatter line {line}: " if line is not None else ""
    message = f"Invalid {kind} {escape_for_log(str(path))}: {where}{detail}"
    context: dict[str, object] = {"path": str(path), "reason": reason}
    if line is not None:
        context["line"] = line
    if field is not None:
        context["field"] = field
    context.update(extra)
    return SkillError(message, context=context)


__all__ = ["Skill"]
