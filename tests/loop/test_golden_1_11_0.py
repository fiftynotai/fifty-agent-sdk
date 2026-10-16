"""Golden tests: with no skills, requests and events keep their 1.11.0 content (FR-004, plan D14a).

Every scenario in :mod:`tests.loop.golden_capture_1_11_0` drives a loop built
WITHOUT ``skills``: all four 1.11.0 prompt slots (``additional_context``
included) under JSON, PROSE and NATIVE; the legacy path with a caller
``tool_descriptions`` the loop replaces; NATIVE with an empty registry; and
two intervention runs (an argument replacement plus an ``after_tool`` note
under JSON, and a NATIVE batch with one denied member). The recorded bodies
(through the real ``OpenAICompatibleClient._build_body``) and the event
stream must equal the checked-in fixture ``golden/requests_1_11_0.json``.

This complements ``test_legacy_golden.py`` (1.7.0), ``test_golden_1_8_0.py``
(1.8.0) and ``test_golden_1_9_0.py`` (1.9.0), which keep running unmodified.

Provenance: the fixture was captured from the UNMODIFIED 1.11.0 tree (commit
``4d4fda6``, the ``v1.11.0`` tag) BEFORE any FR-004 edit under ``src/``
(30903 bytes, sha256
``52ed42e4c8883381830245d9477b5a57a6a0268e28ae0a54641d1b2a9bdab6ce``),
and a capture run on the published 1.11.0 PyPI wheel was ``cmp``-identical.

**Never regenerate the fixture to make a failing diff pass.** There is
deliberately no regenerate flag here. A diff means a no-skills request or
event moved, which FR-004 promises it does not.

What this pins, per scenario:

* every wire body's keys, values and JSON types, the system prompt included
  as each body's first message, so the section order and the
  ``# Additional Context`` adjacency are pinned;
* the event stream: event types, order, ``sequence`` values and payloads;
* id pairing, through ONE first-appearance mapping over bodies then events.

The comparison is canonical JSON text, so ``0``, ``0.0`` and ``False`` differ.

What this does NOT pin:

* Key order. The fixture is written with ``sort_keys=True`` and
  :func:`_canonical` sorts too. Key order holds by construction: FR-004
  touches neither ``_build_body`` nor ``_build_request``'s three returns.
* HTTP bytes. The ``openai`` SDK's serialisation of these bodies was not
  measured.
* Timestamps (excluded) and raw ``uuid4`` values (normalised).
* Any error path: no scenario ends on an ``ErrorEvent``.
* Non-ASCII text: every string in the scenarios is ASCII.
* Log lines: the golden tests capture none.
* Any skills path. No scenario passes ``skills``; an empty ``SkillSet`` is
  pinned against ``skills=None`` by the LS7 differential in
  ``tests/loop/test_loop_skills.py``, and the skills request shape by the
  forward pin ``test_golden_skills_1_12_0.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.loop.golden_capture_1_11_0 import (
    GOLDEN_1_11_0_PATH,
    SCENARIOS_1_11_0,
    capture_scenario_1_11_0,
)


def _fixture() -> dict[str, dict[str, list[dict[str, Any]]]]:
    data: dict[str, dict[str, list[dict[str, Any]]]] = json.loads(
        GOLDEN_1_11_0_PATH.read_text(encoding="utf-8")
    )
    return data


def _canonical(scenario: dict[str, list[dict[str, Any]]]) -> str:
    """Serialise a scenario so the comparison is textual: ``0``, ``0.0`` and ``False`` differ."""
    return json.dumps(scenario, sort_keys=True, ensure_ascii=False)


def test_golden_1_11_0_fixture_covers_every_scenario() -> None:
    """The fixture and the scenario table have exactly the same keys (FR-004 D14a)."""
    assert sorted(_fixture()) == sorted(SCENARIOS_1_11_0)


@pytest.mark.parametrize("key", sorted(SCENARIOS_1_11_0))
async def test_golden_1_11_0(key: str) -> None:
    """With no skills, bodies and events equal the 1.11.0 capture (FR-004 D14a)."""
    assert _canonical(await capture_scenario_1_11_0(key)) == _canonical(_fixture()[key])
