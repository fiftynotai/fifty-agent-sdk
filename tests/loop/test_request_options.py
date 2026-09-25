"""Loop request options: ``AgentLoop(reasoning_effort=..., temperature=...)`` (FR-002).

What these tests pin, and at which layer:

* Every :class:`~fifty_agent_sdk.llm.types.ChatRequest` the loop hands to its
  ``LLMClient`` (``FakeLLMClient.calls``) carries the option values, and so
  does every ``OpenAICompatibleClient._build_body`` output for those requests
  (:func:`tests.loop.golden_capture.wire_bodies`). The HTTP JSON the
  ``openai`` SDK sends for a set ``reasoning_effort`` is pinned separately in
  ``tests/llm/test_openai_compat.py`` (pytest-httpx).
* Each path test uses an exact request count, so a path that short-circuits
  cannot pass vacuously.
* AC-3's "grounding reminder re-asks" have no SDK code path: grounding lives in
  the consuming application (``safety.py``). The SDK paths pinned instead are
  tool steps, ``ToolNotFound`` continues, parser-retry re-asks (including the
  NATIVE empty-completion retry), require-tool-before-final (BR-036) re-asks,
  native single / multi-call / empty-registry turns, the legacy
  ``native_tools_enabled`` flag, streamed turns, and a consumer re-running the
  same loop (directly or through ``AgentRunner``), which is how a
  consumer-side grounding re-ask reaches the SDK.

What these tests do NOT pin: whether any real provider accepts the values.
In particular, whether gpt-5.1 (on core42 or elsewhere) accepts
``temperature`` alongside ``reasoning_effort`` is unverified in this repo; the
default suite makes no live call.

The "options omitted" case is also pinned at full-body level by
``test_golden_1_8_0.py`` and ``test_legacy_golden.py``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, NamedTuple

import pytest
import structlog

import fifty_agent_sdk
from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    AgentRunner,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinalEvent,
    Hooks,
    JsonModeParser,
    MemoryStateStore,
    ObservationEvent,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolFailedEvent,
    ToolMode,
)
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)
from tests.loop.golden_capture import _normalise_ids, wire_bodies

_MODEL = "gpt-5.1"
_USER = [ChatMessage(role="user", content="What is 17 * 23?")]


# --- Option cases -------------------------------------------------------------


class _Case(NamedTuple):
    """Loop kwargs and the values every request must then carry."""

    kwargs: dict[str, Any]
    reasoning_effort: str | None
    temperature: float | None  # None: the key is absent from the wire body


_OPTION_CASES = [
    pytest.param(_Case({"reasoning_effort": "medium"}, "medium", 0.0), id="re_medium"),
    pytest.param(_Case({"temperature": 0.7}, None, 0.7), id="temp_0_7"),
    pytest.param(_Case({"temperature": None}, None, None), id="temp_none"),
    pytest.param(
        _Case({"reasoning_effort": "medium", "temperature": None}, "medium", None), id="both"
    ),
]

_ALL_MODES = [
    pytest.param(None, id="legacy"),
    pytest.param(ToolMode.JSON, id="json"),
    pytest.param(ToolMode.PROSE, id="prose"),
    pytest.param(ToolMode.NATIVE, id="native"),
]

_TEXT_MODES = [
    pytest.param(None, id="legacy_json"),
    pytest.param(ToolMode.JSON, id="json"),
    pytest.param(ToolMode.PROSE, id="prose"),
]


# --- Helpers ------------------------------------------------------------------


def _registry() -> Registry:
    registry = Registry()
    registry.register(FakeTool("search"))
    registry.register(FakeTool("lookup"))
    return registry


def _loop(
    llm: FakeLLMClient,
    mode: ToolMode | None,
    *,
    registry: Registry | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    **extra: Any,
) -> AgentLoop:
    """Build a loop; ``mode=None`` is the legacy JSON call shape (``parser=`` required)."""
    legacy: dict[str, Any] = {}
    if mode is None:
        legacy = {"parser": JsonModeParser(), "output_format": JSON_MODE_OUTPUT_FORMAT}
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _registry(),
        prompts=PromptSections(persona="You are a calculator."),
        safety=safety if safety is not None else SafetyConfig(),
        model=_MODEL,
        stream=stream,
        tool_mode=mode,
        **legacy,
        **extra,
    )


def _tool_text(mode: ToolMode | None, name: str, args: dict[str, Any]) -> str:
    if mode is ToolMode.PROSE:
        return f"Thought: calling {name}\nAction: {name}\nAction Input: {json.dumps(args)}"
    return json.dumps(
        {
            "thought": f"calling {name}",
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def _final_text(mode: ToolMode | None, answer: str) -> str:
    if mode is ToolMode.PROSE:
        return f"Thought: done\nFinal Answer: {answer}"
    if mode is ToolMode.NATIVE:
        return answer
    return json.dumps(
        {
            "thought": "done",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


_DRIFT = "- 17 * 23 is a product, let me think out loud without the format"


def _tool_step(mode: ToolMode | None, name: str, args: dict[str, Any]) -> ChatResponse:
    if mode is ToolMode.NATIVE:
        return make_multi_tool_response([(name, args)])
    return make_response(_tool_text(mode, name, args))


async def _drain(loop: AgentLoop, messages: list[ChatMessage] | None = None) -> None:
    async for _event in loop.run(list(messages if messages is not None else _USER)):
        pass


async def _assert_every_request(
    calls: list[ChatRequest], case: _Case, *, stream: bool = False
) -> None:
    """Every recorded request, and its ``_build_body`` output, carries the case's options."""
    assert calls, "no request was recorded"
    for request in calls:
        assert request.reasoning_effort == case.reasoning_effort
        assert request.temperature == case.temperature
    fake = FakeLLMClient([])
    fake.calls = list(calls)
    for body in await wire_bodies(fake, stream=stream):
        if case.reasoning_effort is None:
            assert "extra_body" not in body
        else:
            assert body["extra_body"] == {"reasoning_effort": case.reasoning_effort}
        if case.temperature is None:
            assert "temperature" not in body
        else:
            assert type(body["temperature"]) is float
            assert body["temperature"] == case.temperature


def _unpaired_tool_replies(body: dict[str, Any]) -> list[int]:
    """Indices of ``role="tool"`` entries whose id is not on the nearest preceding assistant turn."""
    unpaired: list[int] = []
    last_assistant: dict[str, Any] = {}
    for index, message in enumerate(body["messages"]):
        if message["role"] == "assistant":
            last_assistant = message
        elif message["role"] == "tool":
            ids = [tc["id"] for tc in last_assistant.get("tool_calls") or []]
            if message.get("tool_call_id") not in ids:
                unpaired.append(index)
    return unpaired


# --- AC-1 / AC-5 "omitted": unchanged requests ---------------------------------


@pytest.mark.parametrize("mode", _ALL_MODES)
async def test_loop_request_options_unset_leave_requests_unchanged(mode: ToolMode | None) -> None:
    """With both options omitted no option is passed to ``ChatRequest`` (FR-002 AC-1, AC-5).

    Pins field values and ``model_fields_set`` per request, plus
    ``"temperature": 0.0`` as a JSON float and no ``extra_body`` in every
    ``_build_body`` output. Full-body equality with 1.8.0 is the goldens' job.
    """
    llm = FakeLLMClient(
        [_tool_step(mode, "search", {"q": "x"}), make_response(_final_text(mode, "391"))]
    )
    await _drain(_loop(llm, mode))

    assert len(llm.calls) == 2
    for request in llm.calls:
        assert request.reasoning_effort is None
        assert request.temperature == 0.0
        assert "reasoning_effort" not in request.model_fields_set
        assert "temperature" not in request.model_fields_set
    for body in await wire_bodies(llm, stream=False):
        assert "extra_body" not in body
        assert '"temperature": 0.0' in json.dumps(body, sort_keys=True)


# --- AC-3 / AC-5: every request path -------------------------------------------


@pytest.mark.parametrize("mode", _TEXT_MODES)
@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_on_every_text_mode_request(
    case: _Case, mode: ToolMode | None
) -> None:
    """Re-ask, parser retry, ToolNotFound and tool-step requests all carry the options (FR-002 AC-3, AC-5).

    Script: premature final (BR-036 re-ask) → drift (parser retry) → unknown
    tool (``ToolNotFound`` continue) → valid tool → final: five requests.
    """
    llm = FakeLLMClient(
        [
            make_response(_final_text(mode, "premature")),
            make_response(_DRIFT),
            make_response(_tool_text(mode, "missing_tool", {"x": 1})),
            make_response(_tool_text(mode, "search", {"q": "17*23"})),
            make_response(_final_text(mode, "391")),
        ]
    )
    safety = SafetyConfig(require_tool_before_final=True)
    loop = _loop(llm, mode, safety=safety, **case.kwargs)
    with structlog.testing.capture_logs() as logs:
        events = [type(e) async for e in loop.run(list(_USER))]

    assert len(llm.calls) == 5
    triggered = [e["event"] for e in logs]
    assert triggered.count("tool_required_force_triggered") == 1
    assert triggered.count("parser_retry_triggered") == 1
    assert events.count(ToolFailedEvent) == 1
    assert events.count(ObservationEvent) == 1
    assert events[-1] is FinalEvent
    await _assert_every_request(llm.calls, case)


@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_on_every_native_request(case: _Case) -> None:
    """Native multi-call, single call and empty-completion retry all carry the options (FR-002 AC-3, AC-5).

    Also checks that ``tools``/``tool_choice`` are still sent and that every
    ``role="tool"`` reply stays paired to its assistant ``tool_calls`` id (L-934).
    """
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"q": "a"}), ("lookup", {"id": 1})]),
            make_multi_tool_response([("search", {"q": "b"})]),
            make_response(""),
            make_response("391"),
        ]
    )
    safety = SafetyConfig(max_concurrent_tool_calls=2)
    await _drain(_loop(llm, ToolMode.NATIVE, safety=safety, **case.kwargs))

    assert len(llm.calls) == 4
    await _assert_every_request(llm.calls, case)
    bodies = await wire_bodies(llm, stream=False)
    for body in bodies:
        assert [t["function"]["name"] for t in body["tools"]] == ["search", "lookup"]
        assert body["tool_choice"] == "auto"
        assert _unpaired_tool_replies(body) == []
    assert [m["role"] for m in bodies[-1]["messages"]].count("tool") == 3


@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_with_empty_native_registry(case: _Case) -> None:
    """An empty NATIVE registry (no ``tools`` sent) still carries the options (FR-002 AC-3, AC-5)."""
    llm = FakeLLMClient([make_response(""), make_response("391")])
    await _drain(_loop(llm, ToolMode.NATIVE, registry=Registry(), **case.kwargs))

    assert len(llm.calls) == 2
    await _assert_every_request(llm.calls, case)
    for request in llm.calls:
        assert request.tools is None
        assert request.tool_choice is None


@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_on_legacy_native_flag(case: _Case) -> None:
    """The legacy ``native_tools_enabled`` path carries the options (FR-002 AC-3, AC-5)."""
    llm = FakeLLMClient(
        [
            make_multi_tool_response([("search", {"q": "x"})]),
            make_response(_final_text(None, "391")),
        ]
    )
    safety = SafetyConfig(native_tools_enabled=True)
    await _drain(_loop(llm, None, safety=safety, **case.kwargs))

    assert len(llm.calls) == 2
    await _assert_every_request(llm.calls, case)
    for request in llm.calls:
        assert request.tools is not None
        assert request.tool_choice == "auto"


@pytest.mark.parametrize(
    "mode", [pytest.param(ToolMode.JSON, id="json"), pytest.param(ToolMode.PROSE, id="prose")]
)
@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_when_streaming(case: _Case, mode: ToolMode) -> None:
    """Streamed turns, including a parser retry, carry the options (FR-002 AC-3, AC-5)."""
    final = _final_text(mode, "391")
    llm = FakeLLMClient(
        [
            make_stream_chunks([_DRIFT]),
            make_stream_chunks([_tool_text(mode, "search", {"q": "s"})]),
            make_stream_chunks([final[:10], final[10:]]),
        ]
    )
    await _drain(_loop(llm, mode, stream=True, **case.kwargs))

    assert len(llm.calls) == 3
    await _assert_every_request(llm.calls, case, stream=True)
    for body in await wire_bodies(llm, stream=True):
        assert body["stream"] is True


@pytest.mark.parametrize("case", _OPTION_CASES)
async def test_loop_stamps_options_on_consumer_rerun(case: _Case) -> None:
    """A consumer re-running the same loop, directly or via ``AgentRunner``, gets the options (FR-002 AC-3, AC-5).

    This is the shape a consumer-side grounding re-ask takes: another run on
    the same loop. The SDK itself has no grounding-reminder path.
    """
    llm = FakeLLMClient(
        [
            _tool_step(ToolMode.JSON, "search", {"q": "x"}),
            make_response(_final_text(ToolMode.JSON, "391")),
            make_response(_final_text(ToolMode.JSON, "391, checked")),
        ]
    )
    loop = _loop(llm, ToolMode.JSON, **case.kwargs)
    await _drain(loop)
    await _drain(loop, [ChatMessage(role="user", content="Double-check that answer.")])
    assert len(llm.calls) == 3
    await _assert_every_request(llm.calls, case)

    runner_llm = FakeLLMClient(
        [
            make_response(_final_text(ToolMode.JSON, "first")),
            make_response(_final_text(ToolMode.JSON, "second")),
        ]
    )
    runner = AgentRunner(
        loop=_loop(runner_llm, ToolMode.JSON, **case.kwargs), state=MemoryStateStore()
    )
    for turn in ("hello", "again"):
        async for _event in runner.run("session-1", turn):
            pass
    assert len(runner_llm.calls) == 2
    await _assert_every_request(runner_llm.calls, case)


async def test_on_llm_call_hook_sees_request_options() -> None:
    """The ``on_llm_call`` hook receives requests carrying both options (FR-002, consumer observability)."""
    seen: list[ChatRequest] = []

    def on_llm_call(
        session_id: str | None, request: ChatRequest, response: ChatResponse, duration_ms: float
    ) -> None:
        seen.append(request)

    llm = FakeLLMClient([make_multi_tool_response([("search", {"q": "x"})]), make_response("391")])
    loop = _loop(
        llm,
        ToolMode.NATIVE,
        hooks=Hooks(on_llm_call=on_llm_call),
        reasoning_effort="medium",
        temperature=None,
    )
    await _drain(loop)

    assert len(seen) == 2
    assert all(r.reasoning_effort == "medium" and r.temperature is None for r in seen)


# --- Construction-time validation ---------------------------------------------


@pytest.mark.parametrize("bad", ["", "High", " low", "low\n", 3])
def test_loop_rejects_malformed_reasoning_effort(bad: Any) -> None:
    """A malformed ``reasoning_effort`` raises ``ValueError`` at construction (FR-002 D4)."""
    llm = FakeLLMClient([])
    with pytest.raises(ValueError, match="reasoning_effort"):
        _loop(llm, ToolMode.JSON, reasoning_effort=bad)
    assert llm.calls == []


@pytest.mark.parametrize("bad", [-0.1, 2.1, float("nan"), float("inf"), True, "0.7"])
def test_loop_rejects_invalid_temperature(bad: Any) -> None:
    """An invalid ``temperature`` raises ``ValueError`` at construction, before any LLM call (FR-002 D10)."""
    llm = FakeLLMClient([])
    with pytest.raises(ValueError, match="temperature"):
        _loop(llm, ToolMode.JSON, temperature=bad)
    assert llm.calls == []


# --- Independence ---------------------------------------------------------------


@pytest.mark.parametrize("mode", _ALL_MODES)
def test_loop_request_options_are_independent_of_tool_mode(mode: ToolMode | None) -> None:
    """The options change none of the values the tool-mode resolver sets (FR-002 D1, D9)."""
    plain = _loop(FakeLLMClient([]), mode)
    with_options = _loop(FakeLLMClient([]), mode, reasoning_effort="high", temperature=None)

    assert type(with_options._parser) is type(plain._parser)
    assert with_options._tool_message_role == plain._tool_message_role
    assert with_options._system_prompt == plain._system_prompt
    assert with_options._declare_native_tools == plain._declare_native_tools


async def test_loop_temperature_is_not_coupled_to_reasoning_effort() -> None:
    """Setting ``reasoning_effort`` never drops or changes ``temperature``, and logs no warning (FR-002 D9)."""
    llm = FakeLLMClient(
        [
            _tool_step(ToolMode.JSON, "search", {"q": "x"}),
            make_response(_final_text(ToolMode.JSON, "391")),
        ]
    )
    with structlog.testing.capture_logs() as logs:
        await _drain(_loop(llm, ToolMode.JSON, reasoning_effort="high"))
    assert len(llm.calls) == 2
    await _assert_every_request(llm.calls, _Case({}, "high", 0.0))
    assert [e for e in logs if e["log_level"] in ("warning", "error")] == []

    both = FakeLLMClient([make_response(_final_text(ToolMode.JSON, "391"))])
    await _drain(_loop(both, ToolMode.JSON, reasoning_effort="high", temperature=0.5))
    await _assert_every_request(both.calls, _Case({}, "high", 0.5))


# --- AC-5: temperature values ---------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [(0.0, 0.0), (0.7, 0.7), (2.0, 2.0), (1, 1.0)],
    ids=["explicit_0_0", "0_7", "2_0", "int_1"],
)
async def test_loop_temperature_float_is_sent_as_given(given: float, expected: float) -> None:
    """A given temperature is on every request and sent as a JSON float (FR-002 AC-5).

    An explicit ``0.0`` sends the same content as omission; it is told apart
    only by ``model_fields_set``. An ``int`` is sent as the validated float,
    as a direct ``ChatRequest(temperature=1)`` would carry it.
    """
    llm = FakeLLMClient(
        [
            _tool_step(ToolMode.JSON, "search", {"q": "x"}),
            make_response(_final_text(ToolMode.JSON, "391")),
        ]
    )
    await _drain(_loop(llm, ToolMode.JSON, temperature=given))

    assert len(llm.calls) == 2
    await _assert_every_request(llm.calls, _Case({}, None, expected))
    for request in llm.calls:
        assert "temperature" in request.model_fields_set


@pytest.mark.parametrize("mode", _ALL_MODES)
async def test_loop_temperature_none_omits_key_from_every_request(mode: ToolMode | None) -> None:
    """``temperature=None`` omits the key from every request body, in every tool mode (FR-002 AC-5)."""
    llm = FakeLLMClient(
        [_tool_step(mode, "search", {"q": "x"}), make_response(_final_text(mode, "391"))]
    )
    await _drain(_loop(llm, mode, temperature=None))

    assert len(llm.calls) == 2
    for request in llm.calls:
        assert request.temperature is None
    for body in await wire_bodies(llm, stream=False):
        assert "temperature" not in body


async def test_loop_explicit_temperature_none_differs_from_omitted() -> None:
    """Explicit ``temperature=None`` and omission differ ONLY by the ``temperature`` key (FR-002 AC-5)."""

    async def bodies(**kwargs: Any) -> list[dict[str, Any]]:
        llm = FakeLLMClient(
            [
                _tool_step(ToolMode.NATIVE, "search", {"q": "x"}),
                make_response(_final_text(ToolMode.NATIVE, "391")),
            ]
        )
        await _drain(_loop(llm, ToolMode.NATIVE, **kwargs))
        return _normalise_ids(await wire_bodies(llm, stream=False))

    omitted = await bodies()
    explicit_none = await bodies(temperature=None)

    assert len(omitted) == len(explicit_none) == 2
    for body in omitted:
        assert type(body.pop("temperature")) is float
    assert json.dumps(omitted, sort_keys=True) == json.dumps(explicit_none, sort_keys=True)


def test_loop_temperature_default_is_private_sentinel() -> None:
    """Both kwargs are keyword-only; ``temperature``'s default is a private marker, not ``None`` or a number (FR-002 D9).

    Does not pin the marker's type: the enum is a mypy-narrowing choice, not a
    contract. Consumers omit the kwarg rather than pass the marker.
    """
    params = inspect.signature(AgentLoop).parameters
    assert params["reasoning_effort"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["reasoning_effort"].default is None
    assert params["temperature"].kind is inspect.Parameter.KEYWORD_ONLY
    default = params["temperature"].default
    assert default is not None
    assert not isinstance(default, (int, float))
    assert "_Unset" not in fifty_agent_sdk.__all__
    assert "_UNSET" not in fifty_agent_sdk.__all__
