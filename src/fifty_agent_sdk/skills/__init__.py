"""Agent Skills: ``SKILL.md`` discovery, a catalog prompt section and a ``load_skill`` tool (FR-004).

A skill is a directory with a ``SKILL.md`` file: YAML frontmatter holding a
``name`` and a ``description``, then Markdown instructions. Other files in the
directory are reference files the instructions may point to. This follows
the open Agent Skills layout, so skills written for other runtimes load
here.

Progressive disclosure
    Only each skill's name and description go into the system prompt, in a
    ``# Skills`` section (:meth:`SkillSet.render_catalog`). The model calls
    the load tool (:meth:`SkillSet.load_tool`, ``load_skill`` by default) to
    read a skill's instructions, or a reference file, when a task needs it.
    The returned text is a tool result: it stays in the run's working
    messages and, like every tool roundtrip, is never persisted.

Wiring
    ``AgentLoop(skills=SkillSet(...))`` renders the catalog and adds the load
    tool for you, without changing your ``Registry``. Or wire the two by
    hand: ``PromptSections(skills=skill_set.render_catalog())`` and
    ``registry.register(skill_set.load_tool())``.

Freshness
    A :class:`Skill`'s body is read when it is built, and a :class:`SkillSet`
    is immutable, so the catalog and the instructions are a snapshot.
    Reference files are read when the model asks for one. Rebuild the
    ``SkillSet`` (and the loop) to pick up edited skills.

Frontmatter
    The SDK reads ``name`` and ``description`` with a strict YAML subset and
    no YAML dependency: plain, single- or double-quoted (on one line) and
    ``|`` / ``>`` block values. A plain value YAML would read as something
    other than text (``no``, ``on``, ``123``, ``2024-01-01``) must be
    quoted. Other keys (``license``, ``metadata``, ``allowed-tools``, ...)
    are tolerated and not interpreted; :attr:`Skill.frontmatter` keeps the
    raw text for hosts that want to parse it themselves. Under those keys a
    quoted value or a ``[`` / ``{`` collection must close on its own line
    (YAML would let it swallow a following ``name:`` line). Stricter than
    YAML too: a TAB anywhere in the frontmatter, the characters YAML's reader
    refuses, U+0085, U+2028, U+2029 and U+FEFF, and any column-1 line that is
    not ``key:`` (``<<:`` included) are refused.

Trust model
    A skill's directory contents are trusted: its body is instructions to
    your model, so install only skills you trust. ``allowed-tools`` is NOT
    enforced; the SDK has no permission model. The ``file`` argument of the
    load tool is chosen by the model and is not trusted: it is confined to
    the skill's directory (no absolute paths, ``..``, ``:``, hidden names or
    symlinks out of the directory), only regular files of at most 256 KiB of
    UTF-8 text are returned, and no error text holds a host path. On POSIX
    the file is opened by walking from the skill directory with
    ``O_NOFOLLOW`` at every step, so a symlink swapped in after the
    containment check is refused. Where that walk is not available (Windows)
    the open goes by path, so such a swap, which needs write access to the
    skill directory (inside the host's trust domain), is not caught there;
    containment on Windows is unmeasured.
"""

from __future__ import annotations

from fifty_agent_sdk.errors import SkillError
from fifty_agent_sdk.skills.discovery import discover_skills
from fifty_agent_sdk.skills.model import Skill
from fifty_agent_sdk.skills.skill_set import SkillSet

__all__ = ["Skill", "SkillError", "SkillSet", "discover_skills"]
