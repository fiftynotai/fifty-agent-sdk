"""Direct tests of the private file helpers (FR-004 D5).

``test_load_tool.py`` pins what the model sees. These pin the helpers' own
contracts, which the tool's defence-in-depth ``ValueError`` arm would
otherwise hide: :func:`resolve_reference` must raise ``_FileProblem`` for an
outside path itself (mutant M1 removed its containment check and every tool
row stayed green, because ``Path.relative_to`` then raised ``ValueError``,
which the tool also maps to a refusal).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fifty_agent_sdk.skills._files import (
    FD_WALK_SUPPORTED,
    _FileProblem,
    read_reference,
    read_text_bounded,
    resolve_reference,
    split_reference,
)
from tests.skills.conftest import call_with_fifo_deadline


@pytest.fixture
def root(tmp_path: Path) -> Path:
    directory = tmp_path / "skill"
    (directory / "reference").mkdir(parents=True)
    (directory / "reference" / "a.md").write_text("A", encoding="utf-8")
    (directory / ".hidden").write_text("H", encoding="utf-8")
    (tmp_path / "outside.md").write_text("O", encoding="utf-8")
    (directory / "out.md").symlink_to(tmp_path / "outside.md")
    (directory / "secret.md").symlink_to(directory / ".hidden")
    return directory.resolve()


def test_split_reference_drops_empty_and_dot_components_on_both_separators() -> None:
    """``a//./b\\c`` is three components."""
    assert split_reference("a//./b\\c") == ("a", "b", "c")


@pytest.mark.parametrize(
    "requested", [None, 3, "", "/a", "\\a", "C:a", "a:b", "a\x00b", "..", "a/.b", "a\ud800", "."]
)
def test_split_reference_refuses_without_touching_the_filesystem(requested: object) -> None:
    """Step 1 of D5: each of these is ``refused``."""
    with pytest.raises(_FileProblem) as caught:
        split_reference(requested)
    assert caught.value.reason == "refused"


def test_resolve_reference_returns_a_contained_regular_file(root: Path) -> None:
    """A path inside the directory resolves to itself."""
    assert resolve_reference(root, ("reference", "a.md")) == root / "reference" / "a.md"


@pytest.mark.parametrize(
    ("parts", "reason"),
    [
        (("out.md",), "refused"),
        (("secret.md",), "refused"),
        (("..", "outside.md"), "refused"),
        (("reference", "missing.md"), "not_found"),
        (("reference",), "not_a_file"),
    ],
    ids=["link_outside", "link_to_hidden", "dotdot_outside", "missing", "directory"],
)
def test_resolve_reference_raises_its_own_problem_codes(
    root: Path, parts: tuple[str, ...], reason: str
) -> None:
    """Containment, the hidden check and the file checks raise ``_FileProblem``, nothing else."""
    with pytest.raises(_FileProblem) as caught:
        resolve_reference(root, parts)
    assert caught.value.reason == reason


def test_read_text_bounded_reads_at_most_cap_plus_one_bytes(tmp_path: Path) -> None:
    """At the cap the text is returned; one byte over is ``too_large``."""
    path = tmp_path / "f"
    path.write_bytes(b"x" * 10)
    assert read_text_bounded(path, 10) == "x" * 10
    with pytest.raises(_FileProblem) as caught:
        read_text_bounded(path, 9)
    assert caught.value.reason == "too_large"


def test_read_text_bounded_reports_the_offset_after_a_bom(tmp_path: Path) -> None:
    """``not_utf8`` with the offset counted after one stripped BOM."""
    path = tmp_path / "f"
    path.write_bytes(b"\xef\xbb\xbfab\xff")
    with pytest.raises(_FileProblem) as caught:
        read_text_bounded(path, 100)
    assert (caught.value.reason, caught.value.offset) == ("not_utf8", 2)


@pytest.mark.skipif(not FD_WALK_SUPPORTED, reason="the fd walk needs dir_fd and O_NOFOLLOW (POSIX)")
def test_read_reference_refuses_a_link_it_meets_on_the_walk(root: Path) -> None:
    """L1: given a path ``resolve_reference`` never returns (an outward link), the walk refuses it."""
    with pytest.raises(_FileProblem) as caught:
        read_reference(root, root / "out.md", 100)
    assert caught.value.reason == "refused"


@pytest.mark.skipif(not FD_WALK_SUPPORTED, reason="the fd walk needs dir_fd and O_NOFOLLOW (POSIX)")
def test_read_reference_opens_a_fifo_without_blocking(root: Path) -> None:
    """L1: the final open is ``O_NONBLOCK`` and ``fstat`` must say regular file.

    The call runs under :func:`call_with_fifo_deadline`, so a regression that
    opens the FIFO in blocking mode (mutants M17-M19) fails this test within
    seconds and releases the stuck reader instead of hanging the run.
    """
    pipe = root / "pipe"
    os.mkfifo(pipe)
    with pytest.raises(_FileProblem) as caught:
        call_with_fifo_deadline(lambda: read_reference(root, pipe, 100), pipe)
    assert caught.value.reason == "not_a_file"


def test_read_reference_reads_a_contained_file(root: Path) -> None:
    """L1: the walk (or the fallback) returns the file's text."""
    assert read_reference(root, root / "reference" / "a.md", 100) == "A"
