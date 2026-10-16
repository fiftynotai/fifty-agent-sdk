"""Private file helpers shared by the skill loader and the ``load_skill`` tool (FR-004 D5).

Two jobs:

* :func:`read_text_bounded` reads a file as strict UTF-8 text with a size
  cap, for ``SKILL.md`` (at construction).
* :func:`split_reference`, :func:`resolve_reference` and
  :func:`read_reference` turn a path the MODEL chose into the text of a
  regular file inside a skill's directory, or refuse it.

Trust model: a skill directory's contents are chosen by the host and are
trusted. The ``file`` argument of ``load_skill`` is chosen by the model and
may come from prompt injection, so it is checked in this order:

1. Lexically, before any filesystem access (:func:`split_reference`): not
   empty, at most :data:`MAX_REQUESTED_PATH_CHARS` characters, no NUL, no
   surrogate code point, no leading ``/`` or ``\\``, no drive prefix and no
   ``:`` anywhere, and after
   splitting on BOTH ``/`` and ``\\`` (on every OS) no ``..`` component and
   no component starting with ``.``.
2. ``root.joinpath(*parts).resolve(strict=False)``, so symlinks are followed.
   Resolving before any existence check means a link planted to an outside
   path is refused without telling the model whether that path exists.
3. Containment: the resolved path is inside the resolved skill directory,
   and no component below the directory starts with ``.`` (an in-directory
   link to ``.env`` is refused too).
4. Only an existing regular file is read, so a FIFO or a device cannot
   block the read.
5. The file is opened by walking from the skill directory one component at a
   time (:func:`read_reference`): ``os.open`` with ``dir_fd`` and
   ``O_NOFOLLOW`` at every step (``O_DIRECTORY`` for directories,
   ``O_CLOEXEC`` throughout, ``O_NONBLOCK`` for the file), then ``fstat`` must
   say regular file. The resolved path has no symlink in it, so a symlink met
   during the walk was swapped in after step 3 and the read is refused
   (``ELOOP``/``ENOTDIR``). This closes the resolve-then-open race.
6. At most ``cap + 1`` bytes are read; more than ``cap`` is refused.

Residual, documented: where ``os.open`` takes no ``dir_fd`` or there is no
``O_NOFOLLOW`` (Windows), step 5 falls back to opening the resolved path, and
a symlink swapped between step 2 and the read can still lead outside the
directory. That needs write access to the skill directory, which is inside
the host's trust domain. Path behaviour, including the race probe, was
measured on macOS; the same tests run on Ubuntu in CI. Containment on
Windows is unmeasured.

Nothing here returns or raises text holding a host path: problems are
reason codes (:class:`_FileProblem`), and the caller writes the message.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from pathlib import Path
from typing import Final

from fifty_agent_sdk.skills._frontmatter import normalise_text

MAX_SKILL_FILE_BYTES: Final = 262_144
"""The largest ``SKILL.md`` read (256 KiB)."""

MAX_REFERENCE_FILE_BYTES: Final = 262_144
"""The largest reference file ``load_skill`` returns (256 KiB)."""

MAX_REQUESTED_PATH_CHARS: Final = 1024
"""The longest ``file`` argument ``load_skill`` looks at."""

_UTF8_BOM: Final = b"\xef\xbb\xbf"

_SURROGATE: Final = re.compile("[\ud800-\udfff]")

FD_WALK_SUPPORTED: Final = (
    os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
)
"""Whether :func:`read_reference` opens by walking directory fds (POSIX), not by path."""

_WALK_FLAGS: Final = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
"""Flags for every ``os.open`` of the walk; ``O_NOFOLLOW`` refuses a symlink at each step."""

_SWAPPED_IN_ERRNOS: Final = frozenset({errno.ELOOP, errno.ENOTDIR, errno.EMLINK})
"""``O_NOFOLLOW`` met a symlink (``ELOOP``; ``EMLINK`` on FreeBSD), or a directory became a file."""

_READ_CHUNK_BYTES: Final = 65_536


class _FileProblem(Exception):
    """A file that could not be read, as a reason code with no path in it.

    Attributes:
        reason: ``refused``, ``not_found``, ``not_a_file``, ``too_large``,
            ``not_utf8`` or ``os_error``.
        offset: For ``not_utf8``, the byte offset of the first byte that is
            not UTF-8 (counted after a stripped BOM), else ``None``.
        error_type: For ``os_error``, the ``OSError`` subclass name, else ``None``.
    """

    def __init__(
        self, reason: str, *, offset: int | None = None, error_type: str | None = None
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.offset = offset
        self.error_type = error_type


def path_exists(path: Path) -> bool:
    """Whether ``path`` exists (following symlinks); ``False`` only for "not found".

    Unlike :meth:`Path.exists`, whose handling of other ``OSError`` kinds differs
    between CPython releases (measured with a mode-000 parent directory:
    3.11.15, 3.12.15 and 3.13.2 raise ``PermissionError``, 3.14.3 returns
    ``False``; FR-004 evidence), a permission problem or a symlink loop is
    raised here on every interpreter.

    Raises:
        OSError: Any error other than the path (or a parent) not existing.
    """
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def read_text_bounded(path: Path, cap: int) -> str:
    """Read ``path`` as UTF-8 text of at most ``cap`` bytes.

    The file is opened in binary mode and at most ``cap + 1`` bytes are read,
    so a larger file costs no more than that. One leading UTF-8 BOM is
    stripped and the rest is decoded as strict UTF-8, and ``\\r\\n`` and
    lone ``\\r`` become ``\\n``. Strict decoding never yields a surrogate code
    point.

    Args:
        path: The file to read. The caller has already checked what it is.
        cap: The largest size accepted, in bytes.

    Returns:
        The decoded, newline-normalised text.

    Raises:
        _FileProblem: ``too_large`` over the cap, ``not_utf8`` for bytes that
            are not UTF-8.
        OSError: When opening or reading fails; the caller classifies it.
    """
    with path.open("rb") as handle:
        data = handle.read(cap + 1)
    return _decode_bounded(data, cap)


def _decode_bounded(data: bytes, cap: int) -> str:
    """Refuse more than ``cap`` bytes, strip one BOM, decode strict UTF-8, normalise newlines."""
    if len(data) > cap:
        raise _FileProblem("too_large")
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM) :]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _FileProblem("not_utf8", offset=exc.start) from None
    return normalise_text(text)


def split_reference(requested: object) -> tuple[str, ...]:
    """Check a model-chosen reference path lexically and split it into components.

    No filesystem call is made. Both ``/`` and ``\\`` separate components, on
    every OS. Empty and ``.`` components are dropped.

    Args:
        requested: The ``file`` argument as the model sent it.

    Returns:
        The path's components, at least one.

    Raises:
        _FileProblem: ``refused`` for anything the module docstring's step 1
            does not allow.
    """
    if not isinstance(requested, str) or not requested:
        raise _FileProblem("refused")
    if len(requested) > MAX_REQUESTED_PATH_CHARS:
        raise _FileProblem("refused")
    if "\x00" in requested or ":" in requested or requested[0] in "/\\":
        # ":" covers drive prefixes ("C:x", "C:\\x") and NTFS stream names.
        raise _FileProblem("refused")
    if _SURROGATE.search(requested):
        # A lone surrogate code point cannot be encoded for the OS, so the path
        # functions would raise UnicodeEncodeError; no file can have such a name.
        raise _FileProblem("refused")
    parts = tuple(part for part in requested.replace("\\", "/").split("/") if part and part != ".")
    if not parts:
        raise _FileProblem("refused")
    for part in parts:
        # ".." and every hidden name (".env", ".git") start with ".".
        if part.startswith("."):
            raise _FileProblem("refused")
    return parts


def resolve_reference(root: Path, parts: tuple[str, ...]) -> Path:
    """Resolve checked components to an existing regular file inside ``root``.

    Blocking: call it from a worker thread (``asyncio.to_thread``).

    Args:
        root: The skill directory, already resolved (strictly) when the skill
            was loaded.
        parts: Components from :func:`split_reference`.

    Returns:
        The resolved path of a regular file inside ``root``.

    Raises:
        _FileProblem: ``refused`` when resolving fails (a symlink loop, say)
            or the result is outside ``root`` or under a hidden component;
            ``not_found`` when nothing exists there; ``not_a_file`` for a
            directory, FIFO, socket or device; ``os_error`` (with the class
            name) for any other ``OSError`` from ``stat``.
    """
    try:
        resolved = root.joinpath(*parts).resolve(strict=False)
    except (OSError, RuntimeError):
        # RuntimeError: a symlink loop on CPython before 3.13; OSError after.
        raise _FileProblem("refused") from None
    if not resolved.is_relative_to(root):
        raise _FileProblem("refused")
    if any(part.startswith(".") for part in resolved.relative_to(root).parts):
        raise _FileProblem("refused")
    try:
        # An explicit stat(): Path.exists() raises PermissionError on CPython
        # 3.11-3.13 and returns False on 3.14 (measured; see path_exists), so
        # the same file would get a different answer on different interpreters.
        mode = resolved.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        raise _FileProblem("not_found") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            # A symlink loop: CPython 3.13+ resolve() returns the looping path.
            raise _FileProblem("refused") from None
        raise _FileProblem("os_error", error_type=type(exc).__name__) from None
    if not stat.S_ISREG(mode):
        raise _FileProblem("not_a_file")
    return resolved


def read_reference(root: Path, resolved: Path, cap: int) -> str:
    """Read a reference file that :func:`resolve_reference` accepted, refusing a swapped-in symlink.

    Blocking: call it from a worker thread. On POSIX the file is opened by
    walking from ``root`` with ``O_NOFOLLOW`` at every component (module
    docstring, step 5), so a symlink planted after the containment check is
    refused rather than followed out of the directory. Elsewhere it is
    :func:`read_text_bounded` on ``resolved``.

    Args:
        root: The resolved skill directory.
        resolved: The path :func:`resolve_reference` returned (inside ``root``).
        cap: The largest size accepted, in bytes.

    Returns:
        The decoded, newline-normalised text.

    Raises:
        _FileProblem: ``refused`` when the walk meets a symlink or a
            non-directory; ``not_found`` when a component was removed after
            the check; ``not_a_file`` when the opened file is not regular;
            ``too_large`` and ``not_utf8`` as :func:`read_text_bounded`.
        OSError: Any other failure; the caller classifies it.
    """
    if not FD_WALK_SUPPORTED:
        return read_text_bounded(resolved, cap)
    parts = resolved.relative_to(root).parts
    fds: list[int] = []
    try:
        try:
            fds.append(os.open(root, _WALK_FLAGS | os.O_DIRECTORY))
            for part in parts[:-1]:
                fds.append(os.open(part, _WALK_FLAGS | os.O_DIRECTORY, dir_fd=fds[-1]))
            fds.append(os.open(parts[-1], _WALK_FLAGS | os.O_NONBLOCK, dir_fd=fds[-1]))
        except OSError as exc:
            if exc.errno in _SWAPPED_IN_ERRNOS:
                raise _FileProblem("refused") from None
            if exc.errno == errno.ENOENT:
                # Removed between the check and the open: the same E4 as a
                # file that was never there (the path is inside the directory).
                raise _FileProblem("not_found") from None
            raise
        file_fd = fds[-1]
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise _FileProblem("not_a_file")
        chunks: list[bytes] = []
        remaining = cap + 1
        while remaining > 0:
            chunk = os.read(file_fd, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        for fd in fds:
            os.close(fd)
    return _decode_bounded(b"".join(chunks), cap)


__all__ = [
    "MAX_REFERENCE_FILE_BYTES",
    "MAX_REQUESTED_PATH_CHARS",
    "MAX_SKILL_FILE_BYTES",
    "FD_WALK_SUPPORTED",
    "path_exists",
    "read_reference",
    "read_text_bounded",
    "resolve_reference",
    "split_reference",
]
