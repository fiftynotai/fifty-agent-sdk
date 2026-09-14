"""Executable guards for the tag-triggered release workflow."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"
RELEASE_WORKFLOW: Final[Path] = REPO_ROOT / ".github" / "workflows" / "release.yml"


def _tag_version_guard() -> str:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"      - name: Verify tag matches pyproject version\n"
        r"        if: [^\n]+\n"
        r"        run: \|\n"
        r"(?P<script>(?:          .*\n)+)",
        workflow,
    )
    assert match is not None, "release tag-version guard is missing or has an unknown shape"
    return textwrap.dedent(match.group("script"))


def _project_version() -> str:
    with PYPROJECT.open("rb") as pyproject:
        return str(tomllib.load(pyproject)["project"]["version"])


def _run_guard(tag: str) -> subprocess.CompletedProcess[str]:
    executable_dir = str(Path(sys.executable).parent)
    return subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", _tag_version_guard()],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "GITHUB_REF_NAME": tag,
            "PATH": os.pathsep.join((executable_dir, os.environ.get("PATH", ""))),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_tag_guard_accepts_matching_project_version() -> None:
    result = _run_guard(f"v{_project_version()}")
    assert result.returncode == 0, result.stderr


def test_release_tag_guard_rejects_mismatched_project_version() -> None:
    result = _run_guard("v0.0.0")
    assert result.returncode != 0
    assert "does not match pyproject version" in result.stdout
