"""Golden capture of the skills request shape (FR-004, plan D14c): a FORWARD PIN.

Unlike the other golden harnesses, this one is NOT a compatibility baseline:
it was captured from FR-004's own tree, after the loop's ``skills=`` wiring
was green, because no earlier release had skills. It pins what a
skills-enabled loop sends and emits (the ``# Skills`` section, the
``load_skill`` declaration in the prompt-side tool block and in the NATIVE
``tools`` parameter, the tool replies carrying a body, a reference file and
an unknown-name error) so that a later change to that shape shows up as a
diff instead of passing silently. A diff means the skills request shape
moved; whether that is allowed is a release decision, not a reason to
regenerate.

* :data:`SCENARIOS_SKILLS_1_12_0` maps a stable key to a builder. Skills are
  built in code, except ``skills_native_reference_file``, whose skill
  directory is written into a :class:`tempfile.TemporaryDirectory` with
  fixed contents. No text the model sees holds a host path (the load tool's
  contract, ``tests/skills/test_load_tool.py`` T11), so the fixture does not
  depend on where it is captured.
* Bodies go through the real wire translator and events are serialised and
  id-normalised as in :mod:`tests.loop.golden_capture_1_9_0`.

It does not pin key order (sorted on both sides), HTTP bytes, timestamps, raw
ids, log lines, error paths other than the unknown-name tool error, or
anything about runs without skills (the four compatibility goldens do).

Running ``python -m tests.loop.golden_capture_skills_1_12_0`` from the repo
root writes ``tests/loop/golden/requests_skills_1_12_0.json``. Never run it to
make a failing diff pass.

Not named ``test_*.py``, so pytest does not collect it;
:mod:`tests.loop.test_golden_skills_1_12_0` imports it.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    ChatMessage,
    JsonModeParser,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    ToolResult,
)
from fifty_agent_sdk.skills import Skill, SkillSet
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response
from tests.loop.golden_capture import render, wire_bodies
from tests.loop.golden_capture_1_9_0 import _normalise_scenario, serialise_events

GOLDEN_SKILLS_1_12_0_PATH = Path(__file__).parent / "golden" / "requests_skills_1_12_0.json"
"""Where the fixture lives. Written once, from the FR-004 tree, by ``__main__`` below."""

_PERSONA = "You are a document assistant."
_MODEL = "golden-model"
_USER_INPUT = [ChatMessage(role="user", content="Fill in the tax form.")]
_PDF_BODY = "# PDF forms\n\nRead reference/forms.md, then fill each field by id."

LoopScenario = tuple[AgentLoop, FakeLLMClient, list[ChatMessage]]


def _code_skills() -> SkillSet:
    return SkillSet(
        [
            Skill(name="pdf-forms", description="Fill in PDF forms.", body=_PDF_BODY),
            Skill(name="docx", description="Edit Word documents.", body="Use tracked changes."),
        ]
    )


def _registry() -> Registry:
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"hits": 2})))
    return registry


def _json_tool(name: str, args: dict[str, Any]) -> str:
    return json.dumps(
        {
            "thought": f"calling {name}",
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def _json_final(answer: str) -> str:
    return json.dumps(
        {
            "thought": "done",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def _mode_loop(
    llm: FakeLLMClient, tool_mode: ToolMode, skills: SkillSet, **extra: Any
) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        registry=_registry(),
        prompts=PromptSections(persona=_PERSONA),
        safety=SafetyConfig(),
        model=_MODEL,
        tool_mode=tool_mode,
        skills=skills,
        **extra,
    )


def _skills_json_load_then_final() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("load_skill", {"name": "pdf-forms"})),
            make_response(_json_final("Filled.")),
        ]
    )
    return _mode_loop(llm, ToolMode.JSON, _code_skills()), llm, list(_USER_INPUT)


def _skills_prose_load_then_final() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(
                'Thought: load it\nAction: load_skill\nAction Input: {"name": "pdf-forms"}'
            ),
            make_response("Thought: done\nFinal Answer: Filled."),
        ]
    )
    return _mode_loop(llm, ToolMode.PROSE, _code_skills()), llm, list(_USER_INPUT)


def _skills_native_load_then_final() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("load_skill", {"name": "pdf-forms"})]),
            make_response("Filled."),
        ]
    )
    return _mode_loop(llm, ToolMode.NATIVE, _code_skills()), llm, list(_USER_INPUT)


_TEMPORARY_DIRECTORIES: list[tempfile.TemporaryDirectory[str]] = []
"""Skill directories written by builders; :func:`capture_scenario_skills_1_12_0` removes them."""


def _skills_native_reference_file() -> LoopScenario:
    holder = tempfile.TemporaryDirectory()
    _TEMPORARY_DIRECTORIES.append(holder)
    directory = Path(holder.name) / "pdf-forms"
    (directory / "reference").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: pdf-forms\ndescription: Fill in PDF forms.\n---\n{_PDF_BODY}\n",
        encoding="utf-8",
    )
    (directory / "reference" / "forms.md").write_text(
        "Field ids: name, tax_id, total.\n", encoding="utf-8"
    )
    skills = SkillSet([Skill.from_directory(directory)])
    llm = FakeLLMClient(
        [
            make_multi_tool_response(
                [("load_skill", {"name": "pdf-forms", "file": "reference/forms.md"})]
            ),
            make_response("Filled."),
        ]
    )
    return _mode_loop(llm, ToolMode.NATIVE, skills), llm, list(_USER_INPUT)


def _skills_json_unknown_name() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("load_skill", {"name": "spreadsheets"})),
            make_response(_json_final("No such skill.")),
        ]
    )
    return _mode_loop(llm, ToolMode.JSON, _code_skills()), llm, list(_USER_INPUT)


def _skills_legacy_native_tools_enabled() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("load_skill", {"name": "docx"})]),
            make_response(_json_final("Edited.")),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        registry=_registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=_PERSONA),
        safety=SafetyConfig(native_tools_enabled=True),
        model=_MODEL,
        output_format=JSON_MODE_OUTPUT_FORMAT,
        skills=_code_skills(),
    )
    return loop, llm, list(_USER_INPUT)


SCENARIOS_SKILLS_1_12_0: dict[str, Callable[[], LoopScenario]] = {
    "skills_json_load_then_final": _skills_json_load_then_final,
    "skills_prose_load_then_final": _skills_prose_load_then_final,
    "skills_native_load_then_final": _skills_native_load_then_final,
    "skills_native_reference_file": _skills_native_reference_file,
    "skills_json_unknown_name": _skills_json_unknown_name,
    "skills_legacy_native_tools_enabled": _skills_legacy_native_tools_enabled,
}
"""Scenario key -> builder. These are the fixture's top-level keys."""


async def capture_scenario_skills_1_12_0(key: str) -> dict[str, list[dict[str, Any]]]:
    """Run scenario ``key`` and return its id-normalised wire bodies and events."""
    try:
        loop, llm, messages = SCENARIOS_SKILLS_1_12_0[key]()
        events = [event async for event in loop.run(messages)]
        bodies = await wire_bodies(llm, stream=loop._stream)
    finally:
        while _TEMPORARY_DIRECTORIES:
            _TEMPORARY_DIRECTORIES.pop().cleanup()
    return _normalise_scenario(bodies, serialise_events(events))


async def capture_all_skills_1_12_0() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Capture every scenario, keyed as in :data:`SCENARIOS_SKILLS_1_12_0`."""
    return {
        key: await capture_scenario_skills_1_12_0(key) for key in sorted(SCENARIOS_SKILLS_1_12_0)
    }


if __name__ == "__main__":
    GOLDEN_SKILLS_1_12_0_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_SKILLS_1_12_0_PATH.write_text(
        render(asyncio.run(capture_all_skills_1_12_0())), encoding="utf-8"
    )
    print(f"wrote {GOLDEN_SKILLS_1_12_0_PATH}")
