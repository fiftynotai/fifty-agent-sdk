"""Regression tests for the package version attribute (BR-001, TD-001)."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version

import fifty_agent_sdk


def test_version_is_pep440_shaped() -> None:
    assert isinstance(fifty_agent_sdk.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", fifty_agent_sdk.__version__)


def test_version_derives_from_installed_metadata() -> None:
    """``__version__`` is read from installed distribution metadata (TD-001), so
    it cannot drift from the published version the way a hardcoded string did
    (BR-001). In a source checkout with no dist-info it falls back cleanly."""
    try:
        assert fifty_agent_sdk.__version__ == version("fifty-agent-sdk")
    except PackageNotFoundError:
        assert fifty_agent_sdk.__version__ == "0.0.0+unknown"
