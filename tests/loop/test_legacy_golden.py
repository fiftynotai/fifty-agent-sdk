"""AC-3 golden tests: omitting ``tool_mode`` sends the same request bodies as 1.7.0 (FR-001).

Every scenario in :data:`tests.loop.golden_capture.SCENARIOS` builds a loop
with a legacy 1.7.0 call shape (no ``tool_mode``), drives it to completion,
and serialises every request through the real OpenAI wire translator. The
result must equal the checked-in fixture ``golden/legacy_1_7_0.json``.

Provenance: the fixture was captured from the UNMODIFIED 1.7.0 tree (commit
``728ea59``) BEFORE any FR-001 edit under ``src/``, cross-checked
byte-for-byte against a capture run on the published 1.7.0 PyPI wheel, and
its sha256 is recorded in the FR-001 brief.

**Never regenerate the fixture to make a failing diff pass.** There is
deliberately no regenerate flag here. A diff means the legacy path moved,
which FR-001 promises it does not.

What this does NOT pin: the event stream. FR-001 D12 intentionally changes
``ThoughtEvent.text`` on native tool turns (content is now carried as the
thought), which is an event change, not a request change. Legacy event
behaviour is pinned by the unmodified ``tests/loop`` and ``tests/runner``
suites instead. Nor key order: the fixture is written with ``sort_keys=True``
and :func:`_canonical` sorts too, so this pins keys, values and JSON types
only. Key order is preserved by construction
(``OpenAICompatibleClient._build_body`` is unchanged since 1.7.0).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.loop.golden_capture import GOLDEN_PATH, SCENARIOS, capture_scenario


def _fixture() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data


def _canonical(bodies: list[dict[str, Any]]) -> str:
    """Serialise bodies so the comparison is textual: ``0``, ``0.0`` and ``False`` differ."""
    return json.dumps(bodies, sort_keys=True, ensure_ascii=False)


def test_legacy_golden_fixture_covers_every_scenario() -> None:
    """The fixture and the scenario table have exactly the same keys (FR-001 AC-3)."""
    assert sorted(_fixture()) == sorted(SCENARIOS)


@pytest.mark.parametrize("key", sorted(SCENARIOS))
async def test_legacy_golden(key: str) -> None:
    """A legacy-shaped loop's wire bodies equal the 1.7.0 fixture, system prompt included (FR-001 AC-3)."""
    assert _canonical(await capture_scenario(key)) == _canonical(_fixture()[key])
