"""Golden tests: with the FR-002 options omitted, requests keep their 1.8.0 content (FR-002 AC-1, AC-5).

Every scenario in :mod:`tests.loop.golden_capture_1_8_0` either drives a loop
built with an explicit 1.8.0 ``tool_mode`` (no ``reasoning_effort``, no
``temperature`` kwarg), or serialises a fixed direct-client ``ChatRequest``
with ``reasoning_effort`` unset. Each recorded request goes through the real
OpenAI wire translator (``OpenAICompatibleClient._build_body``), and the
result must equal the checked-in fixture ``golden/requests_1_8_0.json``.

This complements ``test_legacy_golden.py`` (the 13 ``tool_mode``-omitted
shapes, captured from 1.7.0). Together they pin:

* AC-1: omitted ``reasoning_effort`` adds no key to any body;
* AC-5 "omitted": omitted ``AgentLoop(temperature=)`` keeps sending
  ``"temperature": 0.0`` as a JSON float on every loop body.

Provenance: the fixture was captured from the UNMODIFIED 1.8.0 tree (commit
``bfdf27f``) BEFORE any FR-002 edit under ``src/``, cross-checked against a
capture run on the published 1.8.0 PyPI wheel, and its sha256 is recorded in
the FR-002 brief.

**Never regenerate the fixture to make a failing diff pass.** There is
deliberately no regenerate flag here. A diff means an unset-path request
moved, which FR-002 promises it does not.

What this pins: for each scenario, every body's keys, values and JSON types,
and (for loop scenarios) the system prompt as the first message. The
comparison is canonical JSON text, so ``0``, ``0.0`` and ``False`` differ.

What this does NOT pin:

* Key order. The fixture is written with ``sort_keys=True`` and
  :func:`_canonical` sorts too. Key order holds by construction: FR-002 adds
  one trailing ``if`` to ``_build_body`` that adds nothing when unset.
* HTTP bytes. The ``openai`` SDK's serialisation of these bodies was not
  measured.
* The event stream. Event behaviour is pinned by the ``tests/loop`` and
  ``tests/runner`` suites.
* ``ChatRequest.model_dump()``. It gains a ``reasoning_effort: None`` key in
  1.9.0; the fixture pins wire bodies, not SDK model dumps.
* Non-ASCII text or U+007F in the JSON the SDK writes for the model, which
  1.10.2 changed (BR-020). No scenario's tool results, tool schemas or
  replayed arguments hold any.
* Any text the 64-level nesting check refuses (BR-019). It refuses no text
  in any scenario, whose tool arguments nest at most 1 level.
* Tool-result text holding a surrogate code point before the escape, or a
  non-string tool result that cannot be rendered (BR-022). No scenario's
  tool-result text holds one, and every successful tool result in the
  scenarios is a non-string value that renders as JSON.
* Model-written text the client sends (an assistant message's content, a
  tool call's name, a tool reply's name) holding a surrogate code point
  (BR-024). No scenario's text in those fields holds one (BR-024 evidence,
  P5).
* A run that dispatches a tool call while structlog passes the SDK's log keys
  to stdlib ``logging`` as ``extra`` with DEBUG enabled (BR-026). No scenario
  routes structlog through stdlib, and the fixtures hold no log lines (by
  reading).
* A run in which a value the SDK escapes for a log line held, before the
  escape, a character that the stream structlog writes that line to cannot
  encode (BR-028). No scenario uses MCP, the fixtures hold no log lines, and
  they hold no control character other than line feed, none in a tool name,
  and no surrogate code point (BR-028 evidence, P5).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.loop.golden_capture_1_8_0 import (
    GOLDEN_1_8_0_PATH,
    SCENARIOS_1_8_0,
    capture_scenario_1_8_0,
)


def _fixture() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = json.loads(
        GOLDEN_1_8_0_PATH.read_text(encoding="utf-8")
    )
    return data


def _canonical(bodies: list[dict[str, Any]]) -> str:
    """Serialise bodies so the comparison is textual: ``0``, ``0.0`` and ``False`` differ."""
    return json.dumps(bodies, sort_keys=True, ensure_ascii=False)


def test_golden_1_8_0_fixture_covers_every_scenario() -> None:
    """The fixture and the scenario table have exactly the same keys (FR-002 AC-1)."""
    assert sorted(_fixture()) == sorted(SCENARIOS_1_8_0)


@pytest.mark.parametrize("key", sorted(SCENARIOS_1_8_0))
async def test_golden_1_8_0(key: str) -> None:
    """With the FR-002 options omitted, the wire bodies equal the 1.8.0 capture (FR-002 AC-1, AC-5)."""
    assert _canonical(await capture_scenario_1_8_0(key)) == _canonical(_fixture()[key])
