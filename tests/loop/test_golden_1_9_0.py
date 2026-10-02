"""Golden tests: with no interventions, requests and events keep their 1.9.0 content (FR-003 AC-4).

Every scenario in :mod:`tests.loop.golden_capture_1_9_0` drives a loop built
WITHOUT ``interventions``: four observation kinds (success, ``is_error``,
``ToolNotFound``, ``ToolTimeout``) through the JSON (``assistant`` and
``user`` role), PROSE, NATIVE single-call, NATIVE batch, legacy ``"tool"``,
legacy ``"assistant"`` and legacy ``native_tools_enabled`` batch paths, plus a
streamed JSON run and the two 1.9.0 request-option shapes. The recorded
bodies (through the real ``OpenAICompatibleClient._build_body``) and the
event stream must equal the checked-in fixture ``golden/requests_1_9_0.json``.

This complements ``test_legacy_golden.py`` (13 ``tool_mode``-omitted shapes,
captured from 1.7.0) and ``test_golden_1_8_0.py`` (22 scenarios captured from
1.8.0), which both keep running unmodified.

Provenance: the fixture was captured from the UNMODIFIED 1.9.0 tree (commit
``50560a3``) BEFORE any FR-003 edit under ``src/``, cross-checked against a
capture run on the published 1.9.0 PyPI wheel (identical file), and its size
and sha256 are recorded in the FR-003 evidence file.

**Never regenerate the fixture to make a failing diff pass.** There is
deliberately no regenerate flag here. A diff means a no-intervention request
or event moved, which FR-003 promises it does not.

What this pins, per scenario:

* every wire body's keys, values and JSON types, the system prompt included
  as each body's first message;
* the event stream: event types, order, ``sequence`` values and payloads
  (``args``, ``result``, ``error``, ``text``, ``raw_completion``);
* id pairing: call ids are normalised with ONE first-appearance mapping over
  the bodies and then the events, so on the ``"tool"``-role paths an event's
  ``call_id`` and the wire ``tool_call_id`` must share a placeholder.

The comparison is canonical JSON text, so ``0``, ``0.0`` and ``False`` differ.

What this does NOT pin:

* Key order. The fixture is written with ``sort_keys=True`` and
  :func:`_canonical` sorts too. Key order holds by construction: FR-003
  touches neither ``_build_body`` nor ``_build_request``.
* HTTP bytes. The ``openai`` SDK's serialisation of these bodies was not
  measured.
* Timestamps (excluded) and raw ``uuid4`` values (normalised), so it cannot
  tell WHEN an id is minted, only how ids pair.
* Any intervention path. No scenario passes ``interventions``; the
  configured-but-passive cases are pinned by
  ``test_passive_interventions_leave_requests_and_events_unchanged``.
* Non-ASCII text or U+007F in the JSON the SDK writes for the model, which
  1.10.2 changed (BR-020). No scenario's tool results, tool schemas or
  replayed arguments hold any.
* Any error path. No scenario ends on an ``ErrorEvent``, so the error-path
  final text, message and context that 1.10.2 changed (BR-021) are outside it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.loop.golden_capture_1_9_0 import (
    GOLDEN_1_9_0_PATH,
    SCENARIOS_1_9_0,
    capture_scenario_1_9_0,
)


def _fixture() -> dict[str, dict[str, list[dict[str, Any]]]]:
    data: dict[str, dict[str, list[dict[str, Any]]]] = json.loads(
        GOLDEN_1_9_0_PATH.read_text(encoding="utf-8")
    )
    return data


def _canonical(scenario: dict[str, list[dict[str, Any]]]) -> str:
    """Serialise a scenario so the comparison is textual: ``0``, ``0.0`` and ``False`` differ."""
    return json.dumps(scenario, sort_keys=True, ensure_ascii=False)


def test_golden_1_9_0_fixture_covers_every_scenario() -> None:
    """The fixture and the scenario table have exactly the same keys (FR-003 AC-4)."""
    assert sorted(_fixture()) == sorted(SCENARIOS_1_9_0)


@pytest.mark.parametrize("key", sorted(SCENARIOS_1_9_0))
async def test_golden_1_9_0(key: str) -> None:
    """With no interventions, bodies and events equal the 1.9.0 capture (FR-003 AC-4)."""
    assert _canonical(await capture_scenario_1_9_0(key)) == _canonical(_fixture()[key])
