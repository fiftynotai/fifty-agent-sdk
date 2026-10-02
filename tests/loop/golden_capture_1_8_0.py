"""Golden wire-body capture for the 1.8.0 request shapes (FR-002 AC-1, AC-5).

FR-002 adds ``ChatRequest.reasoning_effort`` and the ``AgentLoop`` keyword
arguments ``reasoning_effort=`` and ``temperature=``. Its compatibility promise
is that with both kwargs omitted (and ``reasoning_effort`` unset on a direct
``ChatRequest``) the SDK sends the same request bodies as 1.8.0 (same keys,
values and JSON types). The legacy fixture ``legacy_1_7_0.json`` already pins
the 13 ``tool_mode``-omitted loop shapes. This harness pins what that fixture
does not cover:

* the explicit ``tool_mode`` loop shapes that 1.8.0 introduced
  (``ToolMode.JSON`` / ``PROSE`` / ``NATIVE``), including parser-retry,
  require-tool-before-final (BR-036) re-asks, streaming, native multi-call,
  the native empty-registry and empty-completion paths, and ``ToolNotFound``;
* direct-client shapes that the loop never produces (``temperature=None``,
  ``max_tokens`` on a regular and on a reasoning-family model name,
  ``response_format``, a dict ``tool_choice``, ``stream=True``). These run no
  loop: a fixed :class:`~fifty_agent_sdk.llm.types.ChatRequest` goes straight
  through the wire translator.

Every recorded request is serialised through the REAL wire translator
(``OpenAICompatibleClient._build_body``, which does no I/O), and call ids are
normalised with :func:`tests.loop.golden_capture._normalise_ids`, exactly as
the legacy harness does. For loop scenarios the system prompt is pinned
through the first message of every body.

Running ``python -m tests.loop.golden_capture_1_8_0`` from the repo root writes
``tests/loop/golden/requests_1_8_0.json``. The fixture in the tree was written
from the UNMODIFIED 1.8.0 source (commit ``bfdf27f``) before any FR-002 edit
under ``src/``; its sha256 and capture order are recorded in the FR-002 brief.
It must never be regenerated to make a failing diff pass: a diff means an
unset-path request moved. Never run the legacy writer
(``python -m tests.loop.golden_capture``) either: it would rewrite the 1.7.0
fixture from newer source and destroy its provenance.

No scenario here passes ``reasoning_effort`` or ``temperature`` to
``AgentLoop``: neither kwarg existed when the fixture was captured.

It does NOT pin non-ASCII text or U+007F in the JSON the SDK writes for the
model, which 1.10.2 changed (BR-020): no scenario's tool results, tool schemas
or replayed arguments hold any. Nor any text the 64-level nesting check
refuses (BR-019): it refuses no text in any scenario, whose tool arguments
nest at most 1 level. Nor tool-result text holding a surrogate code point
before the escape, or a non-string tool result that cannot be rendered
(BR-022): no scenario's tool-result text holds one, and every successful
tool result in the scenarios is a non-string value that renders as JSON.

This file is deliberately NOT named ``test_*.py`` so pytest does not collect
it; :mod:`tests.loop.test_golden_1_8_0` imports it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fifty_agent_sdk import (
    AgentLoop,
    ChatMessage,
    ChatRequest,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolCall,
    ToolMode,
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
from tests.loop.golden_capture import _normalise_ids, render, wire_bodies

GOLDEN_1_8_0_PATH = Path(__file__).parent / "golden" / "requests_1_8_0.json"
"""Where the fixture lives. Written once, from 1.8.0, by ``__main__`` below."""

_PERSONA = "You are a careful research assistant."
_MODEL = "golden-model"
_USER_INPUT = [ChatMessage(role="user", content="Find the FR-002 design notes.")]

LoopScenario = tuple[AgentLoop, FakeLLMClient, list[ChatMessage]]
"""A built loop, the fake it records requests on, and the caller's input messages."""

DirectScenario = tuple[ChatRequest, bool]
"""A fixed request and the ``stream`` flag it is serialised with (no loop runs)."""


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
    tool_mode: ToolMode,
    *,
    registry: Registry | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    **extra: Any,
) -> AgentLoop:
    """Build a loop with an explicit 1.8.0 ``tool_mode`` and no FR-002 kwargs."""
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _registry(),
        prompts=PromptSections(persona=_PERSONA),
        safety=safety if safety is not None else SafetyConfig(),
        model=_MODEL,
        stream=stream,
        tool_mode=tool_mode,
        **extra,
    )


def _require_tool_safety() -> SafetyConfig:
    return SafetyConfig(
        require_tool_before_final=True,
        tool_required_reminder="Call a tool first if the task needs one.",
    )


# ---------------------------------------------------------------------------
# Loop scenarios — explicit tool_mode, FR-002 kwargs omitted
# ---------------------------------------------------------------------------


def _json_tool_then_final() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("search", {"query": "fr-002"})),
            make_response(_json_final("found it")),
        ]
    )
    return _loop(llm, ToolMode.JSON), llm, list(_USER_INPUT)


def _json_parser_retry() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response("- not an envelope"),
            make_response(_json_final("recovered")),
        ]
    )
    return _loop(llm, ToolMode.JSON), llm, list(_USER_INPUT)


def _json_blank_not_echoed() -> LoopScenario:
    llm = FakeLLMClient([make_response("   "), make_response(_json_final("fixed"))])
    return _loop(llm, ToolMode.JSON), llm, list(_USER_INPUT)


def _json_require_tool_reask() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_final("premature")),
            make_response(_json_tool("lookup", {"id": 7})),
            make_response(_json_final("grounded")),
        ]
    )
    return _loop(llm, ToolMode.JSON, safety=_require_tool_safety()), llm, list(_USER_INPUT)


def _json_user_role() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_json_tool("search", {"query": "gdc", "limit": 1})),
            make_response(_json_final("gdc answer")),
        ]
    )
    return _loop(llm, ToolMode.JSON, tool_message_role="user"), llm, list(_USER_INPUT)


def _json_stream() -> LoopScenario:
    final = _json_final("streamed")
    llm = FakeLLMClient(
        [
            make_stream_chunks([_json_tool("search", {"query": "s"})]),
            make_stream_chunks([final[:20], final[20:]]),
        ]
    )
    return _loop(llm, ToolMode.JSON, stream=True), llm, list(_USER_INPUT)


def _prose_tool_then_final() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response(_prose_tool("search", {"query": "prose"})),
            make_response(_prose_final("prose answer")),
        ]
    )
    return _loop(llm, ToolMode.PROSE), llm, list(_USER_INPUT)


def _prose_parser_retry() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response("I will just ramble without the format."),
            make_response(_prose_final("recovered")),
        ]
    )
    return _loop(llm, ToolMode.PROSE), llm, list(_USER_INPUT)


def _prose_stream() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_stream_chunks([_prose_tool("lookup", {"id": 4})]),
            make_stream_chunks(["Thought: done\n", "Final Answer: streamed prose"]),
        ]
    )
    return _loop(llm, ToolMode.PROSE, stream=True), llm, list(_USER_INPUT)


def _native_single() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "native"})], content="checking"),
            make_response("native answer"),
        ]
    )
    return _loop(llm, ToolMode.NATIVE), llm, list(_USER_INPUT)


def _native_multi() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"query": "a"}), ("lookup", {"id": 2})]),
            make_response("multi answer"),
        ]
    )
    safety = SafetyConfig(max_concurrent_tool_calls=2)
    return _loop(llm, ToolMode.NATIVE, safety=safety), llm, list(_USER_INPUT)


def _native_empty_registry() -> LoopScenario:
    llm = FakeLLMClient([make_response("no tools")])
    return _loop(llm, ToolMode.NATIVE, registry=Registry()), llm, list(_USER_INPUT)


def _native_empty_final_retry() -> LoopScenario:
    llm = FakeLLMClient([make_response(""), make_response("answer after retry")])
    return _loop(llm, ToolMode.NATIVE), llm, list(_USER_INPUT)


def _native_require_tool_reask() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_response("premature"),
            make_multi_tool_response([("lookup", {"id": 9})]),
            make_response("grounded"),
        ]
    )
    return _loop(llm, ToolMode.NATIVE, safety=_require_tool_safety()), llm, list(_USER_INPUT)


def _native_tool_not_found() -> LoopScenario:
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("missing_tool", {"x": 1})]),
            make_response("recovered after not found"),
        ]
    )
    return _loop(llm, ToolMode.NATIVE), llm, list(_USER_INPUT)


LOOP_SCENARIOS_1_8_0: dict[str, Callable[[], LoopScenario]] = {
    "json_tool_then_final": _json_tool_then_final,
    "json_parser_retry": _json_parser_retry,
    "json_blank_not_echoed": _json_blank_not_echoed,
    "json_require_tool_reask": _json_require_tool_reask,
    "json_user_role": _json_user_role,
    "json_stream": _json_stream,
    "prose_tool_then_final": _prose_tool_then_final,
    "prose_parser_retry": _prose_parser_retry,
    "prose_stream": _prose_stream,
    "native_single": _native_single,
    "native_multi": _native_multi,
    "native_empty_registry": _native_empty_registry,
    "native_empty_final_retry": _native_empty_final_retry,
    "native_require_tool_reask": _native_require_tool_reask,
    "native_tool_not_found": _native_tool_not_found,
}
"""Loop scenario key → builder. Every builder passes an explicit ``tool_mode``."""


# ---------------------------------------------------------------------------
# Direct-client scenarios — a fixed ChatRequest, no loop
# ---------------------------------------------------------------------------

_DIRECT_MESSAGES = [
    ChatMessage(role="system", content="Direct system prompt."),
    ChatMessage(role="user", content="Direct question."),
]

_DIRECT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the design corpus.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _direct_basic_gpt4o() -> DirectScenario:
    return ChatRequest(messages=list(_DIRECT_MESSAGES), model="gpt-4o"), False


def _direct_temperature_none() -> DirectScenario:
    return ChatRequest(messages=list(_DIRECT_MESSAGES), model="gpt-5.1", temperature=None), False


def _direct_max_tokens_gpt4o() -> DirectScenario:
    return ChatRequest(messages=list(_DIRECT_MESSAGES), model="gpt-4o", max_tokens=256), False


def _direct_max_tokens_gpt5_1() -> DirectScenario:
    # BR-018: a reasoning-family model name sends max_completion_tokens.
    request = ChatRequest(
        messages=list(_DIRECT_MESSAGES), model="gpt-5.1", temperature=None, max_tokens=256
    )
    return request, False


def _direct_response_format() -> DirectScenario:
    request = ChatRequest(
        messages=list(_DIRECT_MESSAGES),
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return request, False


def _direct_tools_tool_choice_dict() -> DirectScenario:
    # A native history (assistant tool_calls + paired tool reply) plus a
    # forced-function tool_choice object.
    messages = [
        *_DIRECT_MESSAGES,
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="search", args={"query": "q"}, id="call-direct-1")],
        ),
        ChatMessage(
            role="tool", content='{"hits": 1}', name="search", tool_call_id="call-direct-1"
        ),
    ]
    request = ChatRequest(
        messages=messages,
        model="gpt-4o",
        tools=list(_DIRECT_TOOLS),
        tool_choice={"type": "function", "function": {"name": "search"}},
    )
    return request, False


def _direct_stream_true() -> DirectScenario:
    return ChatRequest(messages=list(_DIRECT_MESSAGES), model="gpt-4o", temperature=1.0), True


DIRECT_SCENARIOS_1_8_0: dict[str, Callable[[], DirectScenario]] = {
    "direct_basic_gpt4o": _direct_basic_gpt4o,
    "direct_temperature_none": _direct_temperature_none,
    "direct_max_tokens_gpt4o": _direct_max_tokens_gpt4o,
    "direct_max_tokens_gpt5_1": _direct_max_tokens_gpt5_1,
    "direct_response_format": _direct_response_format,
    "direct_tools_tool_choice_dict": _direct_tools_tool_choice_dict,
    "direct_stream_true": _direct_stream_true,
}
"""Direct-client scenario key → builder of a fixed request and its ``stream`` flag."""

SCENARIOS_1_8_0: frozenset[str] = frozenset(LOOP_SCENARIOS_1_8_0) | frozenset(
    DIRECT_SCENARIOS_1_8_0
)
"""Every scenario key. These are the fixture's top-level keys."""


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


async def _direct_body(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
    """Serialise one fixed request through the real wire translator (no I/O)."""
    client = OpenAICompatibleClient(
        api_key="golden", base_url="http://golden.invalid", max_retries=0
    )
    try:
        return client._build_body(request, model=request.model, stream=stream)
    finally:
        await client.aclose()


async def capture_scenario_1_8_0(key: str) -> list[dict[str, Any]]:
    """Run scenario ``key`` and return its id-normalised wire bodies."""
    if key in DIRECT_SCENARIOS_1_8_0:
        request, stream = DIRECT_SCENARIOS_1_8_0[key]()
        return _normalise_ids([await _direct_body(request, stream=stream)])
    loop, llm, messages = LOOP_SCENARIOS_1_8_0[key]()
    async for _event in loop.run(messages):
        pass
    return _normalise_ids(await wire_bodies(llm, stream=loop._stream))


async def capture_all_1_8_0() -> dict[str, list[dict[str, Any]]]:
    """Capture every scenario, keyed as in :data:`SCENARIOS_1_8_0`."""
    return {key: await capture_scenario_1_8_0(key) for key in sorted(SCENARIOS_1_8_0)}


if __name__ == "__main__":
    GOLDEN_1_8_0_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_1_8_0_PATH.write_text(render(asyncio.run(capture_all_1_8_0())), encoding="utf-8")
    print(f"wrote {GOLDEN_1_8_0_PATH}")
