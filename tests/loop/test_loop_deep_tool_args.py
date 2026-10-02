"""Tool arguments nested deeper than the limit end a run through the typed error path (BR-019).

Before BR-019, a native tool call whose ``arguments`` ``json.loads`` decoded
but the SDK could not re-encode for the next request ran the tool, and
building that request raised a raw ``RecursionError`` out of
``AgentLoop.run()``, with no ``ErrorEvent`` or ``FinalEvent`` (reproduced on
CPython 3.14.3, and on 3.11.15 in a 3-level band; BR-019 evidence P1).
Arguments deeper than ``json.loads`` itself reaches raised it from the decode
instead. Since BR-019 the shipped client refuses arguments nested deeper than
``MAX_TOOL_ARGS_DEPTH`` before decoding them: the run ends on that turn (in
these tests, after one request) with an ``LLMError`` ``ErrorEvent`` and
``SafetyConfig.error_fallback_message``, and no tool or hook runs. The
shipped text parsers raise ``ParserError`` instead, which takes the loop's
parser retry.

The measured-limit test derives its depth from the running interpreter. The
helper it calls (``_measured_codec_limit_depth``) finds, in its own frame,
the deepest arguments ``dumps_for_model`` re-encodes, and returns one level
more if that still decodes there, else the deepest that decode there. In
BR-019's runs (evidence P2 and "L1 on each interpreter"): on CPython 3.14.3
the returned depth decodes and does not re-encode, with a gap of about
11,600 levels; on 3.11.15 the gap is one level, made by
``dumps_for_model``'s own frame, and it closes one frame shallower; on
3.13.2 there was no gap, and the helper returned the decode maximum. What
the depth exercises without the limit therefore differs by interpreter
(BR-019 evidence, mutant M1): on 3.14.3 the client decodes it and the
re-encode for the next request raises, which is the brief's crash; on
3.11.15 and 3.13.2 the client's frames are deeper than the helper's, so the
decode raises first. Either way the depth is far past the limit, so the
refusal the test asserts does not depend on which case applied.

What these do NOT pin: the re-encode crash on 3.11.15, which needs
arguments inside the 3-level band the client decodes but cannot re-encode;
that band moves with the call stack, so no test targets it (BR-019 evidence
P1(d)). Nor custom ``LLMClient`` or ``Parser`` implementations, which decode
their own arguments and are not checked.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk import (
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ErrorEvent,
    FinalEvent,
    Interventions,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
)
from fifty_agent_sdk._json_depth import MAX_TOOL_ARGS_DEPTH
from fifty_agent_sdk._model_json import dumps_for_model
from fifty_agent_sdk.tool_mode import _PROSE_PARSER_RETRY_REMINDER
from tests.loop.conftest import FakeLLMClient, FakeTool, make_response

_USER = [ChatMessage(role="user", content="Run the search.")]
_ENDPOINT = "https://example.com/v1/chat/completions"
_DEPTH_MESSAGE = f"provider tool_call arguments nest deeper than {MAX_TOOL_ARGS_DEPTH} levels"


# --- Helpers -----------------------------------------------------------------------


def _nested_args(depth: int) -> str:
    """Arguments text nested ``depth`` levels: ``{"b":"[","a":[[...0...]]}`` (BR-019).

    The ``"["`` string gives the text one more ``[`` than levels, so the depth
    check reads past its ``str.count`` shortcut at the limit too.
    """
    return '{"b":"[","a":' + "[" * (depth - 1) + "0" + "]" * (depth - 1) + "}"


def _nested_value(depth: int) -> dict[str, Any]:
    """The value ``_nested_args(depth)`` decodes to, built without recursion."""
    chain: Any = 0
    for _ in range(depth - 1):
        chain = [chain]
    return {"b": "[", "a": chain}


def _measured_codec_limit_depth() -> int:
    """The running interpreter's codec limit for ``_nested_args``, measured in this frame.

    ``E`` is the deepest ``_nested_value`` that ``dumps_for_model`` (the
    replay encoder) encodes, found by doubling from 128 and then a binary
    search, every attempt made directly from this frame. If
    ``json.loads(_nested_args(E + 1))`` succeeds here, re-encoding that
    decoded value must raise ``RecursionError`` here (precondition B), and
    ``E + 1`` is returned: arguments that decode but do not re-encode at
    this frame (BR-019 evidence: CPython 3.14.3 and 3.11.15).

    Otherwise this frame has no such gap (BR-019 evidence P2: 3.13.2 inside
    a pytest test), and the deepest arguments that decode here are
    returned; they must be past the limit (precondition A). Deep values
    built here never reach an ``assert``.
    """
    lo, hi = 1, 128
    while True:
        value = _nested_value(hi)
        try:
            dumps_for_model(value)
        except RecursionError:
            break
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        value = _nested_value(mid)
        try:
            dumps_for_model(value)
        except RecursionError:
            hi = mid
        else:
            lo = mid
    del value
    depth = lo + 1
    try:
        decoded = json.loads(_nested_args(depth))
    except RecursionError:
        decoded = None
    if decoded is not None:
        try:
            dumps_for_model(decoded)
        except RecursionError:
            return depth
        pytest.fail(
            f"precondition B failed: arguments nested {depth} levels decode and also re-encode "
            "at this frame although a built value of that depth did not; see BR-019 plan P2"
        )
    lo, hi = 1, depth
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            json.loads(_nested_args(mid))
        except RecursionError:
            hi = mid
        else:
            lo = mid
    if lo <= MAX_TOOL_ARGS_DEPTH:
        pytest.fail(
            f"precondition A failed: json.loads decodes only {lo} levels at this frame, "
            "no deeper than the limit; see BR-019 plan P2"
        )
    return lo


def _tool_calls_body(calls: list[tuple[str, str]]) -> dict[str, Any]:
    """A chat completion whose message carries native ``tool_calls`` (id, arguments)."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "search", "arguments": arguments},
                        }
                        for call_id, arguments in calls
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _final_body(text: str) -> dict[str, Any]:
    body = _tool_calls_body([])
    body["choices"][0]["finish_reason"] = "stop"
    body["choices"][0]["message"] = {"role": "assistant", "content": text}
    return body


def _native_loop(tool: FakeTool, *, interventions: Interventions | None = None) -> AgentLoop:
    registry = Registry()
    registry.register(tool)
    return AgentLoop(
        llm=OpenAICompatibleClient(
            api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
        ),
        registry=registry,
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=ToolMode.NATIVE,
        interventions=interventions,
    )


async def _run(loop: AgentLoop) -> list[AgentEvent]:
    return [event async for event in loop.run(_USER)]


def _assert_refused_after_one_request(
    events: list[AgentEvent], httpx_mock: HTTPXMock, *, call_id: str, arguments: str
) -> None:
    """The run ended with the depth LLMError and the error text, after one request."""
    assert [type(e) for e in events] == [ErrorEvent, FinalEvent]
    error, final = events
    assert isinstance(error, ErrorEvent) and isinstance(final, FinalEvent)
    assert error.error_type == "LLMError"
    assert error.message == _DEPTH_MESSAGE
    assert error.context == {
        "model": "test-model",
        "type": "MalformedResponse",
        "tool_call_id": call_id,
        "arguments_excerpt": arguments[:200],
        "max_tool_args_depth": MAX_TOOL_ARGS_DEPTH,
    }
    assert final.text == SafetyConfig().error_fallback_message
    assert len(httpx_mock.get_requests()) == 1


# --- NATIVE through the real client -----------------------------------------------


@pytest.mark.parametrize("hooks", [False, True], ids=["no_hooks", "hooks"])
@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
async def test_native_run_with_arguments_at_the_measured_codec_limit_ends_with_a_typed_error(
    httpx_mock: HTTPXMock, batch: bool, hooks: bool
) -> None:
    """Arguments at the interpreter's measured codec limit end the run with an LLMError, not a raw RecursionError (BR-019).

    The depth comes from ``_measured_codec_limit_depth()`` (see the module
    docstring). Without the limit, this run raised a raw ``RecursionError``
    at the re-encode on CPython 3.14.3, and at the decode on 3.11.15 and
    3.13.2 (BR-019 evidence, mutant M1). The batch's first call is valid and
    still does not run. The hooks case counts both intervention hooks.
    """
    arguments = _nested_args(_measured_codec_limit_depth())
    calls = [("call_deep", arguments)]
    if batch:
        calls.insert(0, ("call_ok", '{"q": "x"}'))
    httpx_mock.add_response(method="POST", url=_ENDPOINT, json=_tool_calls_body(calls))
    seen = {"before_tool": 0, "after_tool": 0}

    def before_tool(*_: Any) -> None:
        seen["before_tool"] += 1

    def after_tool(*_: Any) -> None:
        seen["after_tool"] += 1

    tool = FakeTool("search")
    interventions = Interventions(before_tool=before_tool, after_tool=after_tool) if hooks else None

    events = await _run(_native_loop(tool, interventions=interventions))

    _assert_refused_after_one_request(events, httpx_mock, call_id="call_deep", arguments=arguments)
    assert tool.call_count == 0
    assert seen == {"before_tool": 0, "after_tool": 0}


async def test_native_run_at_the_limit_round_trips_the_arguments(httpx_mock: HTTPXMock) -> None:
    """Arguments at the limit run the tool and are replayed intact in the next request (BR-019).

    This pins that 64 levels re-encode at the real replay site on whichever
    interpreter runs it.
    """
    arguments = _nested_args(MAX_TOOL_ARGS_DEPTH)
    httpx_mock.add_response(
        method="POST", url=_ENDPOINT, json=_tool_calls_body([("call_1", arguments)])
    )
    httpx_mock.add_response(method="POST", url=_ENDPOINT, json=_final_body("done"))
    tool = FakeTool("search")

    events = await _run(_native_loop(tool))

    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"
    assert not any(isinstance(e, ErrorEvent) for e in events)
    assert tool.call_count == 1
    assert tool.last_args == json.loads(arguments)
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    replayed = [
        m
        for m in json.loads(requests[1].content)["messages"]
        if m["role"] == "assistant" and m.get("tool_calls")
    ]
    assert len(replayed) == 1
    replayed_arguments = replayed[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(replayed_arguments) == json.loads(arguments)


async def test_native_run_one_past_the_limit_ends_after_one_request(
    httpx_mock: HTTPXMock,
) -> None:
    """Arguments one level past the limit end the run with the typed error after one request (BR-019)."""
    arguments = _nested_args(MAX_TOOL_ARGS_DEPTH + 1)
    httpx_mock.add_response(
        method="POST", url=_ENDPOINT, json=_tool_calls_body([("call_deep", arguments)])
    )
    tool = FakeTool("search")

    events = await _run(_native_loop(tool))

    _assert_refused_after_one_request(events, httpx_mock, call_id="call_deep", arguments=arguments)
    assert tool.call_count == 0


# --- Text modes: the parser retry ----------------------------------------------------


def _json_completion(args_text: str) -> str:
    return (
        '{"thought":"t","action":"tool","tool_name":"search","tool_args":'
        + args_text
        + ',"answer":null}'
    )


_TEXT_CASES: dict[str, dict[str, Any]] = {
    "json": {
        "mode": ToolMode.JSON,
        "deep": _json_completion(_nested_args(MAX_TOOL_ARGS_DEPTH + 1)),
        "final": json.dumps(
            {
                "thought": "done",
                "action": "final",
                "tool_name": None,
                "tool_args": None,
                "answer": "fixed",
            }
        ),
        "reminder": SafetyConfig().parser_retry_reminder,
    },
    "prose": {
        "mode": ToolMode.PROSE,
        "deep": "Thought: T\nAction: search\nAction Input: "
        + _nested_args(MAX_TOOL_ARGS_DEPTH + 1),
        "final": "Thought: ok\nFinal Answer: fixed",
        "reminder": _PROSE_PARSER_RETRY_REMINDER,
    },
}


@pytest.mark.parametrize("case", list(_TEXT_CASES))
async def test_text_mode_too_deep_tool_args_take_the_parser_retry(case: str) -> None:
    """Too-deep tool arguments in JSON or PROSE mode take the parser retry; no tool runs (BR-019)."""
    spec = _TEXT_CASES[case]
    llm = FakeLLMClient([make_response(spec["deep"]), make_response(spec["final"])])
    registry = Registry()
    tool = FakeTool("search")
    registry.register(tool)
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        prompts=PromptSections(persona="You are a test agent."),
        safety=SafetyConfig(),
        model="test-model",
        tool_mode=spec["mode"],
    )

    events = await _run(loop)

    assert isinstance(events[-1], FinalEvent) and events[-1].text == "fixed"
    assert not any(isinstance(e, ErrorEvent) for e in events)
    assert tool.call_count == 0
    assert len(llm.calls) == 2
    assert llm.calls[1].messages[-1] == ChatMessage(role="user", content=spec["reminder"])
