"""Golden tests: the skills request shape, a FORWARD PIN (FR-004, plan D14c).

**This is not a compatibility baseline.** The fixture
``golden/requests_skills_1_12_0.json`` was captured from FR-004's own tree
(on top of ``4d4fda6``, after the loop's ``skills=`` wiring was green; 37736
bytes, sha256
``5e0a970f9da5a1fcfb609f8b6aac0ce03a32bb530359e6640714880c43516560``), because
no earlier release had skills. A diff means the skills request shape moved:
the ``# Skills`` section, the ``load_skill`` declaration (prompt-side tool
block under JSON and PROSE, the ``tools`` parameter under NATIVE and the
legacy flag), or the tool replies for a body, a reference file and an unknown
name. Whether such a move is allowed is a release decision. Never regenerate
the fixture to make a failing diff pass.

The compatibility promise for runs WITHOUT skills is pinned by the four other
goldens (``test_legacy_golden.py``, ``test_golden_1_8_0.py``,
``test_golden_1_9_0.py``, ``test_golden_1_11_0.py``).

What this does NOT pin: key order (sorted on both sides), HTTP bytes,
timestamps and raw ids (excluded and normalised), log lines, error paths
other than the unknown-name tool error, the containment refusals (pinned by
``tests/skills/test_load_tool.py``), and non-ASCII skill text (pinned by LS12
in ``test_loop_skills.py``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.loop.golden_capture_skills_1_12_0 import (
    GOLDEN_SKILLS_1_12_0_PATH,
    SCENARIOS_SKILLS_1_12_0,
    capture_scenario_skills_1_12_0,
)


def _fixture() -> dict[str, dict[str, list[dict[str, Any]]]]:
    data: dict[str, dict[str, list[dict[str, Any]]]] = json.loads(
        GOLDEN_SKILLS_1_12_0_PATH.read_text(encoding="utf-8")
    )
    return data


def _canonical(scenario: dict[str, list[dict[str, Any]]]) -> str:
    """Serialise a scenario so the comparison is textual: ``0``, ``0.0`` and ``False`` differ."""
    return json.dumps(scenario, sort_keys=True, ensure_ascii=False)


def test_golden_skills_fixture_covers_every_scenario() -> None:
    """The fixture and the scenario table have exactly the same keys (FR-004 D14c)."""
    assert sorted(_fixture()) == sorted(SCENARIOS_SKILLS_1_12_0)


@pytest.mark.parametrize("key", sorted(SCENARIOS_SKILLS_1_12_0))
async def test_golden_skills_1_12_0(key: str) -> None:
    """With skills, bodies and events equal the forward-pin capture (FR-004 D14c)."""
    assert _canonical(await capture_scenario_skills_1_12_0(key)) == _canonical(_fixture()[key])
