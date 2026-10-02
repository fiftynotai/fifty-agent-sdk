"""Golden wire-body capture for the legacy (``tool_mode`` omitted) loop path (FR-001 AC-3).

FR-001 adds ``AgentLoop(tool_mode=...)``. Its compatibility promise is that a
loop built WITHOUT ``tool_mode`` sends the same request bodies as 1.7.0 (keys,
values, JSON types). This module is the harness that pins that promise:

* :data:`SCENARIOS` maps a stable key to a builder. Each builder uses ONLY the
  legacy 1.7.0 call shapes (no ``tool_mode``) and scripts a fresh
  :class:`tests.loop.conftest.FakeLLMClient`.
* :func:`capture_scenario` drives one scenario to completion and serialises
  every recorded :class:`~fifty_agent_sdk.llm.types.ChatRequest` through the
  REAL wire translator (``OpenAICompatibleClient._build_body``, which does no
  I/O), so the fixture pins the provider-facing JSON body, system prompt
  included, rather than an SDK-internal model dump.
* :func:`_normalise_ids` rewrites the loop's ``uuid4`` call ids to
  ``<id-N>`` in first-appearance order, keeping the id-PAIRING structure
  (assistant ``tool_calls[].id`` ↔ ``role="tool"`` ``tool_call_id``) while
  staying independent of how many times ``uuid4`` is called.

It does NOT pin non-ASCII text or U+007F in the JSON the SDK writes for the
model, which 1.10.2 changed (BR-020): no scenario's tool results, tool schemas
or replayed arguments hold any.

Running ``python -m tests.loop.golden_capture`` from the repo root writes
``tests/loop/golden/legacy_1_7_0.json``. The fixture in the tree was written
from the UNMODIFIED 1.7.0 source (commit ``728ea59``) before any FR-001 edit
under ``src/``; its sha256 is recorded in the FR-001 brief. It must never be
regenerated to make a failing diff pass — a diff means the legacy path moved.

This file is deliberately NOT named ``test_*.py`` so pytest does not collect
it; :mod:`tests.loop.test_legacy_golden` imports it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    AgentLoop,
    ChatMessage,
    JsonModeParser,
    OpenAICompatibleClient,
    PromptSections,
    ProseModeParser,
    Registry,
    SafetyConfig,
    ToolResult,
    tool,
)
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "legacy_1_7_0.json"
"""Where the fixture lives. Written once, from 1.7.0, by ``__main__`` below."""

_PERSONA = "You are a careful research assistant."
_MODEL = "golden-model"
_USER_INPUT = [ChatMessage(role="user", content="Find the FR-001 design notes.")]

Scenario = tuple[AgentLoop, FakeLLMClient, list[ChatMessage]]
"""A built loop, the fake it records requests on, and the caller's input messages."""


# ---------------------------------------------------------------------------
# Deterministic registry and scripted replies
# ---------------------------------------------------------------------------


def _registry() -> Registry:
    """A registry with one ``@tool`` (real properties + required) and one ``FakeTool``."""

    @tool(description="Search the design corpus.")
    async def search(query: str, limit: int = 10) -> list[str]:
        return [f"{query}-hit-{i}" for i in range(min(limit, 2))]

    registry = Registry()
    registry.register(search)
    registry.register(FakeTool("lookup", result=ToolResult(output={"id": 7, "ok": True})))
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


def _prose_tool(name: str, args: dict[str, Any]) -> str:
    return f"Thought: calling {name}\nAction: {name}\nAction Input: {json.dumps(args)}"


def _prose_final(answer: str) -> str:
    return f"Thought: done\nFinal Answer: {answer}"


def _loop(
    llm: FakeLLMClient,
    *,
    registry: Registry | None = None,
    parser: JsonModeParser | ProseModeParser | None = None,
    output_format: str = JSON_MODE_OUTPUT_FORMAT,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    **extra: Any,
) -> AgentLoop:
    """Build a loop with the legacy 1.7.0 call shape (``parser=`` required, no ``tool_mode``)."""
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _registry(),
        parser=parser if parser is not None else JsonModeParser(),
        prompts=PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else SafetyConfig(),
        model=_MODEL,
        stream=stream,
        output_format=output_format,
        **extra,
    )


# ---------------------------------------------------------------------------
# Scenarios — legacy call shapes ONLY
# ---------------------------------------------------------------------------


def _json_default_role() -> Scenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("search", {"query": "fr-001"})),
            make_response(_json_final("found it")),
        ]
    )
    return _loop(llm), llm, list(_USER_INPUT)


def _json_assistant_role_require_tool() -> Scenario:
    # Consumer A shape: assistant-role results + require_tool_before_final +
    # custom reminders.
    llm = FakeLLMClient(
        [
            make_response(_json_final("premature")),
            make_response(_json_tool("lookup", {"id": 7})),
            make_response(_json_final("grounded")),
        ]
    )
    safety = SafetyConfig(
        require_tool_before_final=True,
        tool_required_reminder="Call a tool first if the task needs one.",
        parser_retry_reminder="Reply with the JSON envelope only.",
    )
    loop = _loop(llm, safety=safety, tool_message_role="assistant")
    return loop, llm, list(_USER_INPUT)


def _json_user_role() -> Scenario:
    # Consumer B shape: user-role results.
    llm = FakeLLMClient(
        [
            make_response(_json_tool("search", {"query": "gdc", "limit": 1})),
            make_response(_json_final("gdc answer")),
        ]
    )
    return _loop(llm, tool_message_role="user"), llm, list(_USER_INPUT)


def _json_no_output_format() -> Scenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("lookup", {"id": 1})),
            make_response(_json_final("bare")),
        ]
    )
    return _loop(llm, output_format=""), llm, list(_USER_INPUT)


def _json_parser_retry() -> Scenario:
    llm = FakeLLMClient(
        [
            make_response("- item one\n- item two"),
            make_response(_json_final("recovered")),
        ]
    )
    return _loop(llm), llm, list(_USER_INPUT)


def _json_parser_retry_blank() -> Scenario:
    llm = FakeLLMClient([make_response(""), make_response(_json_final("after blank"))])
    return _loop(llm), llm, list(_USER_INPUT)


def _json_stream() -> Scenario:
    final = _json_final("streamed")
    llm = FakeLLMClient([make_stream_chunks([final[:10], final[10:30], final[30:]])])
    return _loop(llm, stream=True), llm, list(_USER_INPUT)


def _prose_default() -> Scenario:
    llm = FakeLLMClient(
        [
            make_response(_prose_tool("search", {"query": "prose"})),
            make_response(_prose_final("prose answer")),
        ]
    )
    loop = _loop(llm, parser=ProseModeParser(), output_format=PROSE_MODE_OUTPUT_FORMAT)
    return loop, llm, list(_USER_INPUT)


def _prose_parser_retry() -> Scenario:
    # Pins the JSON-worded default reminder under a legacy prose loop.
    llm = FakeLLMClient(
        [make_response("garbage with no headers"), make_response(_prose_final("ok"))]
    )
    loop = _loop(llm, parser=ProseModeParser(), output_format=PROSE_MODE_OUTPUT_FORMAT)
    return loop, llm, list(_USER_INPUT)


def _native_single_with_content() -> Scenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "native"})], content="checking"),
            make_response(_json_final("native answer")),
        ]
    )
    loop = _loop(llm, safety=SafetyConfig(native_tools_enabled=True))
    return loop, llm, list(_USER_INPUT)


def _native_multi() -> Scenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "a"}), ("lookup", {"id": 2})]),
            make_response(_json_final("multi answer")),
        ]
    )
    safety = SafetyConfig(native_tools_enabled=True, max_concurrent_tool_calls=2)
    return _loop(llm, safety=safety), llm, list(_USER_INPUT)


def _native_text_tool_call() -> Scenario:
    # The 1.7.0 half-native shape: a TEXT tool call under native_tools_enabled
    # is dispatched and replied with role="tool" after an assistant turn with
    # no tool_calls. Intentionally preserved on the legacy path (FR-001 D2).
    llm = FakeLLMClient(
        [
            make_response(_json_tool("lookup", {"id": 3})),
            make_response(_json_final("half native")),
        ]
    )
    loop = _loop(llm, safety=SafetyConfig(native_tools_enabled=True))
    return loop, llm, list(_USER_INPUT)


def _native_empty_registry() -> Scenario:
    llm = FakeLLMClient([make_response(_json_final("no tools"))])
    loop = _loop(llm, registry=Registry(), safety=SafetyConfig(native_tools_enabled=True))
    return loop, llm, list(_USER_INPUT)


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "json_default_role": _json_default_role,
    "json_assistant_role_require_tool": _json_assistant_role_require_tool,
    "json_user_role": _json_user_role,
    "json_no_output_format": _json_no_output_format,
    "json_parser_retry": _json_parser_retry,
    "json_parser_retry_blank": _json_parser_retry_blank,
    "json_stream": _json_stream,
    "prose_default": _prose_default,
    "prose_parser_retry": _prose_parser_retry,
    "native_single_with_content": _native_single_with_content,
    "native_multi": _native_multi,
    "native_text_tool_call": _native_text_tool_call,
    "native_empty_registry": _native_empty_registry,
}
"""Scenario key → builder. Keys are the fixture's top-level keys."""


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _normalise_ids(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace every call id with ``<id-N>`` in first-appearance order across ``bodies``.

    Covers ``tool_calls[].id`` on assistant entries and ``tool_call_id`` on
    reply entries. The same raw id always maps to the same placeholder, so the
    pairing structure survives normalisation.
    """
    mapping: dict[str, str] = {}

    def _placeholder(raw: str) -> str:
        if raw not in mapping:
            mapping[raw] = f"<id-{len(mapping) + 1}>"
        return mapping[raw]

    normalised: list[dict[str, Any]] = json.loads(json.dumps(bodies))
    for body in normalised:
        for message in body["messages"]:
            for entry in message.get("tool_calls") or []:
                if entry.get("id") is not None:
                    entry["id"] = _placeholder(entry["id"])
            if message.get("tool_call_id") is not None:
                message["tool_call_id"] = _placeholder(message["tool_call_id"])
    return normalised


async def wire_bodies(llm: FakeLLMClient, *, stream: bool) -> list[dict[str, Any]]:
    """Serialise every request ``llm`` recorded through the real OpenAI wire translator.

    ``_build_body`` is a pure function of the request (no I/O); a throwaway
    client is built only to reach it and is closed before returning.
    """
    client = OpenAICompatibleClient(
        api_key="golden", base_url="http://golden.invalid", max_retries=0
    )
    try:
        return [client._build_body(req, model=req.model, stream=stream) for req in llm.calls]
    finally:
        await client.aclose()


async def capture_scenario(key: str) -> list[dict[str, Any]]:
    """Run scenario ``key`` to completion and return its id-normalised wire bodies."""
    loop, llm, messages = SCENARIOS[key]()
    async for _event in loop.run(messages):
        pass
    return _normalise_ids(await wire_bodies(llm, stream=loop._stream))


async def capture_all() -> dict[str, list[dict[str, Any]]]:
    """Capture every scenario, keyed as in :data:`SCENARIOS`."""
    return {key: await capture_scenario(key) for key in SCENARIOS}


def render(captured: dict[str, list[dict[str, Any]]]) -> str:
    """The exact fixture text: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(captured, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(render(asyncio.run(capture_all())), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")
