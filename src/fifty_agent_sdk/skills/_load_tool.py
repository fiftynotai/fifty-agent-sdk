"""Private ``load_skill`` tool: a skill's instructions, or a file inside the skill (FR-004).

:class:`_LoadSkillTool` is an ordinary :class:`~fifty_agent_sdk.tools.protocol.Tool`,
so it works in every tool mode: under ``ToolMode.NATIVE`` it is declared in
the ``tools`` request parameter, under ``JSON`` and ``PROSE`` (and the legacy
path) it is listed in the prompt's ``# Tools`` block.

Every outcome is a :class:`~fifty_agent_sdk.tools.protocol.ToolResult`;
:meth:`_LoadSkillTool.invoke` raises nothing but
:class:`asyncio.CancelledError` (and process-fatal ``BaseException``\\ s). In
particular it catches ``OSError`` itself, because
:meth:`fifty_agent_sdk.tools.registry.Registry.invoke`'s generic arm would
quote ``str(exc)``, which holds the absolute path, to the model. Its error
texts are fixed; the only values in them are the model's own ``name`` and
``file`` arguments, cut to 200 characters and written with ``repr``. No host
path ever reaches the model.

Logging (coding_guidelines §10): DEBUG ``skill_loaded`` (``skill_name``,
``has_file``, ``output_length``) and ``skill_file_refused`` (``skill_name``,
``reason``, ``requested_length``). The requested path and the file content are
never logged; the skill name goes through
:func:`fifty_agent_sdk._model_json.escape_for_log`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import structlog

from fifty_agent_sdk._model_json import escape_for_log
from fifty_agent_sdk.skills._files import (
    MAX_REFERENCE_FILE_BYTES,
    _FileProblem,
    read_reference,
    resolve_reference,
    split_reference,
)
from fifty_agent_sdk.skills.model import Skill
from fifty_agent_sdk.tools.protocol import ToolResult, ToolSchema

_log: Final = structlog.get_logger(__name__)
"""DEBUG ``skill_loaded`` and ``skill_file_refused``."""

_MAX_SHOWN_ARGUMENT_CHARS: Final = 200
"""How much of a model-supplied argument an error text quotes."""

LOAD_TOOL_DESCRIPTION: Final = (
    "Load a skill's instructions by name, or a file inside that skill. "
    "The available skills are listed in the Skills section."
)
"""The tool description the model sees."""

_NAME_PROPERTY: Final[dict[str, str]] = {
    "type": "string",
    "description": "The skill's name, exactly as listed in the Skills section.",
}
_FILE_PROPERTY: Final[dict[str, str]] = {
    "type": "string",
    "description": (
        "Optional. A file the skill's instructions mention, as a path relative to "
        'the skill (for example "reference/forms.md"). Omit it to load the '
        "skill's instructions."
    ),
}


def _shown(value: str) -> str:
    """A model-supplied argument as an error text quotes it: cut to 200 characters, then ``repr``."""
    return repr(value[:_MAX_SHOWN_ARGUMENT_CHARS])


def _read_reference(root: Path, parts: tuple[str, ...]) -> str:
    """Resolve, check and read a reference file. Runs in a worker thread.

    Raises:
        _FileProblem: For every way it can fail, ``OSError`` included
            (``os_error`` with the class name only).
    """
    try:
        resolved = resolve_reference(root, parts)
        return read_reference(root, resolved, MAX_REFERENCE_FILE_BYTES)
    except OSError as exc:
        raise _FileProblem("os_error", error_type=type(exc).__name__) from None
    except ValueError:
        # Defence in depth: the lexical check already refuses what makes path
        # functions raise ValueError (NUL, surrogate code points).
        raise _FileProblem("refused") from None


class _LoadSkillTool:
    """The ``load_skill`` tool of one :class:`~fifty_agent_sdk.skills.SkillSet`.

    Args:
        name: The tool name (``SkillSet(tool_name=...)``).
        skills: The set's skills by name.
    """

    def __init__(self, name: str, skills: Mapping[str, Skill]) -> None:
        self.name = name
        self.description = LOAD_TOOL_DESCRIPTION
        self.schema = ToolSchema(
            properties={"name": dict(_NAME_PROPERTY), "file": dict(_FILE_PROPERTY)},
            required=["name"],
            additionalProperties=False,
        )
        self._skills = skills

    def _bad_arguments(self) -> ToolResult:
        return ToolResult(
            is_error=True,
            error=f'{self.name} takes "name" (a string) and an optional "file" (a string).',
        )

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Return a skill's body, or a reference file's text, as a ``ToolResult``.

        ``file`` absent or ``None`` (models often send ``null`` for an optional
        argument) loads the instructions. Reference reads run in a worker
        thread; a cancellation propagates.
        """
        if not isinstance(args, dict) or "name" not in args or set(args) - {"name", "file"}:
            return self._bad_arguments()
        requested_name = args["name"]
        requested_file = args.get("file")
        if not isinstance(requested_name, str) or not (
            requested_file is None or isinstance(requested_file, str)
        ):
            return self._bad_arguments()
        skill = self._skills.get(requested_name)
        if skill is None:
            return ToolResult(
                is_error=True,
                error=(
                    f"Unknown skill: {_shown(requested_name)}. "
                    "Use a name listed in the Skills section."
                ),
            )
        if requested_file is None:
            _log.debug(
                "skill_loaded",
                skill_name=escape_for_log(skill.name),
                has_file=False,
                output_length=len(skill.body),
            )
            return ToolResult(output=skill.body)
        return await self._load_reference(skill, requested_file)

    async def _load_reference(self, skill: Skill, requested_file: str) -> ToolResult:
        shown_name = _shown(skill.name)
        shown_file = _shown(requested_file)
        if skill.directory is None:
            return self._refused(
                skill, requested_file, "no_files", f"Skill {shown_name} has no files to read."
            )
        try:
            parts = split_reference(requested_file)
            text = await asyncio.to_thread(_read_reference, skill.directory, parts)
        except _FileProblem as problem:
            return self._refused(
                skill,
                requested_file,
                problem.reason,
                _problem_text(problem, shown_name, shown_file),
            )
        _log.debug(
            "skill_loaded",
            skill_name=escape_for_log(skill.name),
            has_file=True,
            output_length=len(text),
        )
        return ToolResult(output=text)

    def _refused(self, skill: Skill, requested_file: str, reason: str, text: str) -> ToolResult:
        _log.debug(
            "skill_file_refused",
            skill_name=escape_for_log(skill.name),
            reason=reason,
            requested_length=len(requested_file),
        )
        return ToolResult(is_error=True, error=text)


def _problem_text(problem: _FileProblem, shown_name: str, shown_file: str) -> str:
    """The fixed error text for a reference-file problem (plan Phase 3, E3-E8)."""
    if problem.reason == "not_found":
        return f"No file {shown_file} in skill {shown_name}."
    if problem.reason == "not_a_file":
        return f"{shown_file} in skill {shown_name} is not a regular file."
    if problem.reason == "too_large":
        return (
            f"File {shown_file} in skill {shown_name} is larger than the "
            f"{MAX_REFERENCE_FILE_BYTES}-byte limit."
        )
    if problem.reason == "not_utf8":
        return f"File {shown_file} in skill {shown_name} is not UTF-8 text."
    if problem.reason == "os_error":
        return f"Could not read {shown_file} in skill {shown_name} ({problem.error_type})."
    return f"File refused: {shown_file} is not a path inside skill {shown_name}."


__all__ = ["LOAD_TOOL_DESCRIPTION"]
