"""Tests for the ``load_skill`` tool, the reference-file containment battery (FR-004 D5; plan T1-T16).

Every row runs the tool twice, directly and through
:meth:`fifty_agent_sdk.tools.registry.Registry.invoke`, and asserts that both
return the same :class:`ToolResult` and that neither raises (AC: "never a raw
exception"). T11 runs every row again and asserts no host path appears in any
text the model would see.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from fifty_agent_sdk.skills import Skill, SkillSet
from fifty_agent_sdk.skills import _load_tool as load_tool_module
from fifty_agent_sdk.skills._files import FD_WALK_SUPPORTED
from fifty_agent_sdk.tools.protocol import Tool, ToolResult
from fifty_agent_sdk.tools.registry import Registry
from tests.skills.conftest import (
    FIFO_DEADLINE_SECONDS,
    make_skill_dir,
    release_fifo_readers,
    skill,
)

_CAP = 262_144
_BODY = "# PDF\n\nRead reference/forms.md before filling a form."
_OUTSIDE = "OUTSIDE-SECRET"


def _refused(requested: str) -> str:
    return f"File refused: {requested!r} is not a path inside skill 'pdf'."


def _not_found(requested: str) -> str:
    return f"No file {requested!r} in skill 'pdf'."


def _not_a_file(requested: str) -> str:
    return f"{requested!r} in skill 'pdf' is not a regular file."


@pytest.fixture
def layout(tmp_path: Path) -> Path:
    """A ``pdf`` skill with reference files, hidden files and links into and out of it."""
    directory = make_skill_dir(
        tmp_path / "skills",
        "pdf",
        description="Fill PDF forms.",
        body=_BODY,
        files={
            "reference/forms.md": "Forms: use field ids.",
            "reference/nested/deep.md": "Deep text.",
            "crlf.md": b"one\r\ntwo\rthree",
            "bom.md": b"\xef\xbb\xbfwith bom",
            "empty.md": "",
            ".secret": "hidden",
            ".env": "TOKEN=1",
            ".git/config": "[core]",
            "big.md": b"x" * _CAP,
            "too_big.md": b"x" * (_CAP + 1),
            "latin1.md": b"caf\xe9",
        },
    )
    (tmp_path / "outside.txt").write_text(_OUTSIDE, encoding="utf-8")
    (tmp_path / "outside_dir").mkdir()
    (tmp_path / "outside_dir" / "x.txt").write_text(_OUTSIDE, encoding="utf-8")
    (directory / "subdir").mkdir()
    (directory / "link_out.md").symlink_to(tmp_path / "outside.txt")
    (directory / "link_in.md").symlink_to(directory / "reference" / "forms.md")
    (directory / "link_secret.md").symlink_to(directory / ".secret")
    (directory / "dir_out").symlink_to(tmp_path / "outside_dir", target_is_directory=True)
    (directory / "loop.md").symlink_to(directory / "loop.md")
    (directory / "dangling_out.md").symlink_to(tmp_path / "no_such_outside_file")
    (directory / "dangling_in.md").symlink_to(directory / "no_such_inside_file")
    return directory


@pytest.fixture
def tool(layout: Path) -> Tool:
    """The load tool of a set holding the directory skill ``pdf`` and the code skill ``notes``."""
    return SkillSet(
        [Skill.from_directory(layout), skill("notes", "Take notes.", "Note body.")]
    ).load_tool()


async def _both(tool: Tool, args: Any) -> ToolResult:
    """Invoke directly and through a Registry; both must return the same ToolResult."""
    direct = await tool.invoke(args)
    registry = Registry()
    registry.register(tool)
    routed = await registry.invoke(tool.name, args, timeout=5.0)
    assert isinstance(direct, ToolResult)
    assert routed == direct
    return direct


def _text(result: ToolResult) -> str:
    return str(result.output) if not result.is_error else str(result.error)


# --- T1, T2, T3: names and arguments --------------------------------------------------------


async def test_known_name_returns_the_body(tool: Tool) -> None:
    """T1: the full SKILL.md body, as a ``str``."""
    result = await _both(tool, {"name": "pdf"})
    assert result == ToolResult(output=_BODY)


async def test_file_none_behaves_like_no_file(tool: Tool) -> None:
    """T3: ``file: null`` loads the instructions (models often send null for optional args)."""
    assert await _both(tool, {"name": "pdf", "file": None}) == ToolResult(output=_BODY)


async def test_unknown_name_is_a_tool_error(tool: Tool) -> None:
    """T2: E0, exact."""
    result = await _both(tool, {"name": "nope"})
    assert result == ToolResult(
        is_error=True, error="Unknown skill: 'nope'. Use a name listed in the Skills section."
    )


async def test_unknown_name_is_quoted_cut_to_200_characters(tool: Tool) -> None:
    """T2: a long model-supplied name is cut before it is quoted."""
    result = await _both(tool, {"name": "x" * 5000})
    assert result.error == f"Unknown skill: {'x' * 200!r}. Use a name listed in the Skills section."


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"file": "a.md"},
        {"name": 1},
        {"name": None},
        {"name": "pdf", "extra": 1},
        {"name": "pdf", "file": 3},
        {"name": "pdf", "file": ["a"]},
    ],
    ids=["empty", "no_name", "int_name", "null_name", "extra_key", "int_file", "list_file"],
)
async def test_bad_arguments_are_a_tool_error(tool: Tool, args: dict[str, Any]) -> None:
    """T3: E1, exact."""
    result = await _both(tool, args)
    assert result == ToolResult(
        is_error=True,
        error='load_skill takes "name" (a string) and an optional "file" (a string).',
    )


# --- T4: reference files ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("reference/forms.md", "Forms: use field ids."),
        ("reference/nested/deep.md", "Deep text."),
        ("./reference//forms.md", "Forms: use field ids."),
        ("reference\\forms.md", "Forms: use field ids."),
        ("crlf.md", "one\ntwo\nthree"),
        ("bom.md", "with bom"),
        ("empty.md", ""),
        ("link_in.md", "Forms: use field ids."),
    ],
)
async def test_reference_files_inside_the_skill_are_returned(
    tool: Tool, requested: str, expected: str
) -> None:
    """T4: nested paths, ``\\`` separators, CRLF/CR normalised, BOM stripped, inside links allowed."""
    assert await _both(tool, {"name": "pdf", "file": requested}) == ToolResult(output=expected)


# --- T5: lexical refusals ---------------------------------------------------------------------

_LEXICAL_ROWS: list[str] = [
    "../x",
    "a/../../x",
    "reference/../forms.md",
    "/etc/passwd",
    "C:\\x",
    "C:x",
    "\\\\srv\\x",
    "a\\..\\..\\x",
    ".env",
    ".git/config",
    "reference/.hidden",
    "x.md:stream",
    "a\x00b",
    "",
    ".",
    "./",
    "a" * 1025,
    "bad\ud800name.md",
]


@pytest.mark.parametrize("requested", _LEXICAL_ROWS)
async def test_lexically_invalid_paths_are_refused(tool: Tool, requested: str) -> None:
    """T5: E3 before any filesystem access, for every traversal, absolute, drive or hidden form."""
    result = await _both(tool, {"name": "pdf", "file": requested})
    assert result == ToolResult(is_error=True, error=_refused(requested[:200]))


@pytest.mark.parametrize("requested", _LEXICAL_ROWS)
async def test_lexical_refusals_make_no_filesystem_call(
    tool: Tool, requested: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T5 (D5 step 1): a lexically refused path never reaches the filesystem step.

    The post-resolve containment and dot checks would refuse most of these
    rows too, with the same text, so without this spy removing the lexical
    check would leave every other row green.
    """
    reached: list[tuple[Any, ...]] = []

    def no_filesystem(*args: Any) -> str:
        reached.append(args)
        raise AssertionError("the filesystem step ran")

    monkeypatch.setattr(load_tool_module, "_read_reference", no_filesystem)
    result = await tool.invoke({"name": "pdf", "file": requested})
    assert result == ToolResult(is_error=True, error=_refused(requested[:200]))
    assert reached == []


# --- T6: symlinks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested",
    ["link_out.md", "link_secret.md", "dir_out/x.txt", "dangling_out.md"],
    ids=["file_outside", "inside_hidden", "dir_outside", "dangling_outside"],
)
async def test_links_leading_out_or_to_hidden_files_are_refused(tool: Tool, requested: str) -> None:
    """T6: E3. The dangling outside link is refused, not "not found": no existence oracle."""
    result = await _both(tool, {"name": "pdf", "file": requested})
    assert result == ToolResult(is_error=True, error=_refused(requested))
    assert _OUTSIDE not in _text(result)


async def test_a_self_loop_link_is_refused_or_not_found_never_raised(tool: Tool) -> None:
    """T6: a symlink loop ends as E3 (3.11/3.12 raise in resolve) or E4 (3.13+ resolve returns)."""
    result = await _both(tool, {"name": "pdf", "file": "loop.md"})
    assert result.error in {_refused("loop.md"), _not_found("loop.md")}


async def test_a_dangling_link_inside_is_not_found(tool: Tool) -> None:
    """T6: a link to a missing inside path is E4 (the target is inside, so existence may be told)."""
    result = await _both(tool, {"name": "pdf", "file": "dangling_in.md"})
    assert result == ToolResult(is_error=True, error=_not_found("dangling_in.md"))


async def test_a_missing_file_is_not_found(tool: Tool) -> None:
    """T6/E4: a plain missing name inside the skill."""
    result = await _both(tool, {"name": "pdf", "file": "reference/missing.md"})
    assert result == ToolResult(is_error=True, error=_not_found("reference/missing.md"))


# --- T7, T8, T9, T10: what is read ---------------------------------------------------------------


async def test_a_file_at_the_cap_is_returned_and_one_over_is_refused(tool: Tool) -> None:
    """T7: 262,144 bytes is allowed; 262,145 is E6."""
    at_cap = await _both(tool, {"name": "pdf", "file": "big.md"})
    assert at_cap.output == "x" * _CAP
    over = await _both(tool, {"name": "pdf", "file": "too_big.md"})
    assert over == ToolResult(
        is_error=True,
        error="File 'too_big.md' in skill 'pdf' is larger than the 262144-byte limit.",
    )


async def test_a_file_that_is_not_utf8_is_refused(tool: Tool) -> None:
    """T8: E7."""
    result = await _both(tool, {"name": "pdf", "file": "latin1.md"})
    assert result == ToolResult(
        is_error=True, error="File 'latin1.md' in skill 'pdf' is not UTF-8 text."
    )


async def test_a_directory_is_not_a_regular_file(tool: Tool) -> None:
    """T9: E5 for a directory."""
    result = await _both(tool, {"name": "pdf", "file": "subdir"})
    assert result == ToolResult(is_error=True, error=_not_a_file("subdir"))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs only")
async def test_a_fifo_is_refused_without_blocking(layout: Path, tool: Tool) -> None:
    """T9: E5 within 2 s; opening a FIFO with no writer would block forever.

    Two checks make sure it is never opened in blocking mode:
    ``resolve_reference``'s ``stat`` says it is not a regular file, and on
    POSIX the fd walk opens the final component with ``O_NONBLOCK`` and checks
    ``fstat``. If a regression removes both (mutant M19), ``asyncio.timeout``
    fails the test after 2 s, and the ``finally`` releases the worker thread
    still blocked in ``open()``, so the process can exit.
    """
    pipe = layout / "pipe"
    os.mkfifo(pipe)
    try:
        async with asyncio.timeout(FIFO_DEADLINE_SECONDS):
            result = await _both(tool, {"name": "pdf", "file": "pipe"})
    finally:
        release_fifo_readers(pipe, attempts=1)
    assert result == ToolResult(is_error=True, error=_not_a_file("pipe"))


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can read a mode-000 file"
)
async def test_an_unreadable_file_names_only_the_error_class(layout: Path, tool: Tool) -> None:
    """T10: E8 with the ``OSError`` class name; never ``str(exc)``, which holds the path."""
    locked = layout / "locked.md"
    locked.write_text("locked", encoding="utf-8")
    locked.chmod(0)
    try:
        result = await _both(tool, {"name": "pdf", "file": "locked.md"})
    finally:
        locked.chmod(0o600)
    assert result == ToolResult(
        is_error=True, error="Could not read 'locked.md' in skill 'pdf' (PermissionError)."
    )


# --- T11: no host path anywhere ---------------------------------------------------------------


async def test_no_text_the_model_sees_holds_a_host_path(
    tmp_path: Path, layout: Path, tool: Tool
) -> None:
    """T11: across every row above (and E2, E8), no output or error contains the tmp path."""
    requests: list[Any] = [
        {"name": "pdf"},
        {"name": "nope"},
        {},
        {"name": "notes", "file": "x.md"},
        *({"name": "pdf", "file": requested} for requested in _LEXICAL_ROWS),
        *(
            {"name": "pdf", "file": requested}
            for requested in (
                "reference/forms.md",
                "link_out.md",
                "link_secret.md",
                "dir_out/x.txt",
                "dangling_out.md",
                "dangling_in.md",
                "loop.md",
                "reference/missing.md",
                "too_big.md",
                "latin1.md",
                "subdir",
            )
        ),
    ]
    locked = layout / "locked.md"
    locked.write_text("locked", encoding="utf-8")
    locked.chmod(0)
    requests.append({"name": "pdf", "file": "locked.md"})
    try:
        texts = [_text(await _both(tool, args)) for args in requests]
    finally:
        locked.chmod(0o600)
    host_paths = {str(tmp_path), str(tmp_path.resolve()), str(layout), str(layout.resolve())}
    leaks = [text for text in texts for host_path in host_paths if host_path in text]
    assert leaks == []
    assert all(_OUTSIDE not in text for text in texts)


# --- T12: a skill built in code ------------------------------------------------------------------


async def test_file_on_a_code_built_skill_is_a_tool_error(tool: Tool) -> None:
    """T12: E2; its body still loads."""
    assert await _both(tool, {"name": "notes"}) == ToolResult(output="Note body.")
    result = await _both(tool, {"name": "notes", "file": "x.md"})
    assert result == ToolResult(is_error=True, error="Skill 'notes' has no files to read.")


# --- T13: log lines ----------------------------------------------------------------------------


async def test_log_lines_carry_metadata_only(tool: Tool, monkeypatch: pytest.MonkeyPatch) -> None:
    """T13: exact key sets; the requested path is never logged; the name goes through escape_for_log."""
    escaped: list[str] = []
    real_escape = load_tool_module.escape_for_log

    def spy(text: str) -> str:
        escaped.append(text)
        return real_escape(text)

    monkeypatch.setattr(load_tool_module, "escape_for_log", spy)
    secret_path = "reference/../SECRET-PATH.md"
    with capture_logs() as logs:
        await tool.invoke({"name": "pdf"})
        await tool.invoke({"name": "pdf", "file": "reference/forms.md"})
        await tool.invoke({"name": "pdf", "file": secret_path})
        await tool.invoke({"name": "pdf", "file": "reference/missing.md"})
    events = [(entry["event"], entry) for entry in logs]
    loaded = [entry for event, entry in events if event == "skill_loaded"]
    refused = [entry for event, entry in events if event == "skill_file_refused"]
    assert [set(entry) for entry in loaded] == [
        {"event", "log_level", "skill_name", "has_file", "output_length"}
    ] * 2
    assert [(entry["has_file"], entry["output_length"]) for entry in loaded] == [
        (False, len(_BODY)),
        (True, len("Forms: use field ids.")),
    ]
    assert [set(entry) for entry in refused] == [
        {"event", "log_level", "skill_name", "reason", "requested_length"}
    ] * 2
    assert [(entry["reason"], entry["requested_length"]) for entry in refused] == [
        ("refused", len(secret_path)),
        ("not_found", len("reference/missing.md")),
    ]
    assert all(entry["log_level"] == "debug" for entry in logs)
    assert "SECRET-PATH" not in repr(logs)
    assert escaped == ["pdf"] * 4


# --- T14, T15, T16: concurrency, threads, cancellation ---------------------------------------------


async def test_concurrent_reads_all_succeed(tool: Tool) -> None:
    """T14: ten reads gathered together each return their file."""
    results = await asyncio.gather(
        *(tool.invoke({"name": "pdf", "file": "reference/forms.md"}) for _ in range(10))
    )
    assert results == [ToolResult(output="Forms: use field ids.")] * 10


async def test_reference_reads_run_in_a_worker_thread(
    tool: Tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T15: the filesystem work goes through ``asyncio.to_thread``; the body load does not."""
    calls: list[Callable[..., Any]] = []
    real_to_thread = asyncio.to_thread

    async def spy(func: Callable[..., Any], /, *args: Any) -> Any:
        calls.append(func)
        return await real_to_thread(func, *args)

    monkeypatch.setattr(load_tool_module.asyncio, "to_thread", spy)
    await tool.invoke({"name": "pdf"})
    assert calls == []
    await tool.invoke({"name": "pdf", "file": "reference/forms.md"})
    assert calls == [load_tool_module._read_reference]


async def test_cancellation_during_a_reference_read_propagates(
    tool: Tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T16: cancelling the call while the read is in its thread raises ``CancelledError``."""
    started = threading.Event()
    release = threading.Event()
    real_read = load_tool_module._read_reference

    def slow_read(root: Path, parts: tuple[str, ...]) -> str:
        started.set()
        release.wait(5)
        return real_read(root, parts)

    monkeypatch.setattr(load_tool_module, "_read_reference", slow_read)
    task = asyncio.create_task(tool.invoke({"name": "pdf", "file": "reference/forms.md"}))
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()


# --- L1: a symlink swapped in after the containment check (aegis round 2) --------------------


@pytest.mark.skipif(not FD_WALK_SUPPORTED, reason="the fd walk needs dir_fd and O_NOFOLLOW (POSIX)")
@pytest.mark.parametrize(
    "requested", ["reference/forms.md", "reference/nested/deep.md"], ids=["file", "directory"]
)
async def test_a_symlink_swapped_in_after_the_check_is_refused(
    tmp_path: Path, layout: Path, tool: Tool, requested: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L1: replacing the checked file (or a directory on its path) by an outward link gets E3.

    Deterministic: the swap happens inside a wrapper around ``resolve_reference``,
    after the containment check and before the open, which is the window the
    race used (aegis measured 236 of 3000 wins before the fd walk, 0 after).
    """
    real_resolve = load_tool_module.resolve_reference

    def resolve_then_swap(root: Path, parts: tuple[str, ...]) -> Path:
        resolved = real_resolve(root, parts)
        if requested == "reference/forms.md":
            resolved.unlink()
            resolved.symlink_to(tmp_path / "outside.txt")
        else:
            nested = layout / "reference" / "nested"
            nested.rename(layout / "reference" / "moved")
            nested.symlink_to(tmp_path / "outside_dir", target_is_directory=True)
            (tmp_path / "outside_dir" / "deep.md").write_text(_OUTSIDE, encoding="utf-8")
        return resolved

    monkeypatch.setattr(load_tool_module, "resolve_reference", resolve_then_swap)
    result = await tool.invoke({"name": "pdf", "file": requested})
    assert result == ToolResult(is_error=True, error=_refused(requested))
    assert _OUTSIDE not in _text(result)


@pytest.mark.skipif(not FD_WALK_SUPPORTED, reason="the fd walk needs dir_fd and O_NOFOLLOW (POSIX)")
async def test_the_fd_walk_closes_every_descriptor(layout: Path, tool: Tool) -> None:
    """L1: successful, refused and not-a-file reads leave no descriptor open."""
    before = len(os.listdir("/dev/fd"))
    for requested in ("reference/nested/deep.md", "link_out.md", "subdir", "too_big.md"):
        await tool.invoke({"name": "pdf", "file": requested})
    assert len(os.listdir("/dev/fd")) == before


@pytest.mark.skipif(not FD_WALK_SUPPORTED, reason="the fd walk needs dir_fd and O_NOFOLLOW (POSIX)")
async def test_a_file_removed_after_the_check_is_not_found(
    layout: Path, tool: Tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sentinel advisory 2: a file deleted between the check and the open gets E4, not E8."""
    real_resolve = load_tool_module.resolve_reference

    def resolve_then_remove(root: Path, parts: tuple[str, ...]) -> Path:
        resolved = real_resolve(root, parts)
        resolved.unlink()
        return resolved

    monkeypatch.setattr(load_tool_module, "resolve_reference", resolve_then_remove)
    result = await tool.invoke({"name": "pdf", "file": "reference/forms.md"})
    assert result == ToolResult(is_error=True, error=_not_found("reference/forms.md"))
