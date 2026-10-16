"""Shared helpers for the skills tests (FR-004).

* :func:`skill_md` renders a ``SKILL.md`` text from a name, a description and
  a body (or a raw frontmatter).
* :func:`make_skill_dir` writes a skill directory, with optional reference
  files, under a root.
* :func:`skill` builds a :class:`~fifty_agent_sdk.skills.Skill` in code.
* :func:`release_fifo_readers` and :func:`call_with_fifo_deadline` keep a FIFO
  test from hanging: a reader blocked in ``open()`` on a FIFO with no writer
  never returns, so a regression that opens one must turn the test red within
  seconds and leave the process able to exit (sentinel advisory, FR-004).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from fifty_agent_sdk.skills import Skill

_T = TypeVar("_T")

FIFO_DEADLINE_SECONDS = 2.0
"""How long a FIFO test lets the code under test run before it counts as blocked."""

DEFAULT_BODY = "# Instructions\n\nFollow these steps carefully."
"""The body :func:`make_skill_dir` writes unless told otherwise."""


def skill_md(
    name: str | None = None,
    description: str = "Use this skill for test tasks.",
    *,
    frontmatter: str | None = None,
    body: str = DEFAULT_BODY,
) -> str:
    """A ``SKILL.md`` text: ``frontmatter`` verbatim, or ``name``/``description`` lines."""
    if frontmatter is None:
        frontmatter = f"name: {name}\ndescription: {description}\n"
    return f"---\n{frontmatter}---\n{body}\n"


def make_skill_dir(
    root: Path,
    name: str,
    *,
    frontmatter: str | None = None,
    description: str = "Use this skill for test tasks.",
    body: str = DEFAULT_BODY,
    files: dict[str, str | bytes] | None = None,
) -> Path:
    """Write ``root/name/SKILL.md`` (and ``files``, relative to the skill) and return the directory."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        skill_md(name, description, frontmatter=frontmatter, body=body), encoding="utf-8"
    )
    for relative, content in (files or {}).items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    return directory


def skill(
    name: str = "pdf-forms",
    description: str = "Fill in PDF forms.",
    body: str = "Use the form filler.",
) -> Skill:
    """A :class:`Skill` built in code (no directory, so no reference files)."""
    return Skill(name=name, description=description, body=body)


def release_fifo_readers(fifo: Path, *, attempts: int = 20, pause: float = 0.05) -> bool:
    """Unblock any reader stuck in ``open()`` on ``fifo``: open the write end without blocking.

    ``O_WRONLY | O_NONBLOCK`` succeeds only while a reader has the FIFO open
    or is waiting in ``open()``; the write end is closed at once, so that
    reader then reads end-of-file and returns. With no reader the open fails
    with ``ENXIO`` (the expected case), so it is retried ``attempts`` times in
    case the reader has not reached ``open()`` yet.

    Returns:
        Whether a blocked reader was released.
    """
    for _ in range(attempts):
        try:
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            time.sleep(pause)
            continue
        return True
    return False


def call_with_fifo_deadline(call: Callable[[], _T], fifo: Path) -> _T:
    """Run a blocking ``call`` that touches ``fifo`` in a daemon thread, failing if it blocks.

    The call gets :data:`FIFO_DEADLINE_SECONDS`. If it is still running then,
    the reader is released (:func:`release_fifo_readers`) so the thread can
    finish and the process can exit, and the test fails. Otherwise its result
    is returned or its exception re-raised in the test's thread.
    """
    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # re-raised below, in the test's thread
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(FIFO_DEADLINE_SECONDS)
    if worker.is_alive():
        release_fifo_readers(fifo)
        worker.join(FIFO_DEADLINE_SECONDS)
        pytest.fail(f"the call blocked on a FIFO for more than {FIFO_DEADLINE_SECONDS} s")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["value"]  # type: ignore[return-value]
