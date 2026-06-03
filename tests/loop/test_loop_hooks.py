"""Integration tests for :class:`agent_sdk.loop.AgentLoop` hook dispatch.

Covers BR-012's Loop-integration surface — the two Loop-tier hooks driven
directly through :meth:`AgentLoop.run`, with no :class:`AgentRunner`:

* ``on_iteration`` fires once per ReACT iteration, with the 1-based counter.
* ``on_llm_call`` fires once per SUCCESSFUL LLM call, with a real
  :class:`ChatRequest` / :class:`ChatResponse` and a ``duration_ms``.
* ``on_llm_call`` in stream mode receives the synthesized
  :class:`ChatResponse`.
* ``on_llm_call`` does NOT fire on an :class:`LLMError`.
* ``session_id`` is forwarded verbatim (or ``None`` when unset).
* Sync and async hook variants behave identically.
* A raising Loop hook never aborts the loop.
* ``hooks=None`` is a zero-overhead no-op.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent_sdk import (
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ErrorEvent,
    FinalEvent,
    Hooks,
    JsonModeParser,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolResult,
)
from agent_sdk.errors import LLMError
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_response,
    make_stream_chunks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _final_json(answer: str) -> str:
    return json.dumps(
        {
            "thought": "done",
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def _tool_json(thought: str, name: str, args: dict[str, Any] | None) -> str:
    return json.dumps(
        {
            "thought": thought,
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def _make_loop(
    *,
    llm: FakeLLMClient,
    registry: Registry | None = None,
    safety: SafetyConfig | None = None,
    stream: bool = False,
    hooks: Hooks | None = None,
) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona="You are a helpful agent."),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
        stream=stream,
        hooks=hooks,
    )


async def _collect(it: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in it]


def _user(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


class IterationRecorder:
    """Records ``on_iteration`` calls; sync or async per the flag."""

    def __init__(self, *, sync: bool = True) -> None:
        self.args: list[tuple[Any, ...]] = []
        self._sync = sync

    def hook(self) -> Any:
        def record(*args: Any) -> None:
            self.args.append(args)

        if self._sync:
            return record

        async def arecord(*args: Any) -> None:
            record(*args)

        return arecord


class LlmCallRecorder:
    """Records ``on_llm_call`` calls; sync or async per the flag."""

    def __init__(self, *, sync: bool = True) -> None:
        self.args: list[tuple[Any, ...]] = []
        self._sync = sync

    def hook(self) -> Any:
        def record(*args: Any) -> None:
            self.args.append(args)

        if self._sync:
            return record

        async def arecord(*args: Any) -> None:
            record(*args)

        return arecord


# ---------------------------------------------------------------------------
# on_iteration
# ---------------------------------------------------------------------------


async def test_on_iteration_fires_once_per_iteration() -> None:
    """A 2-step run (one tool, one final) fires ``on_iteration`` exactly twice."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output="ok")))
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("look up", "search", {})),
            make_response(_final_json("done")),
        ]
    )
    rec = IterationRecorder()
    loop = _make_loop(llm=llm, registry=registry, hooks=Hooks(on_iteration=rec.hook()))

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(rec.args) == 2
    assert [a[1] for a in rec.args] == [1, 2]


async def test_on_iteration_receives_session_id() -> None:
    """``on_iteration`` receives the forwarded ``session_id``."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])
    rec = IterationRecorder()
    loop = _make_loop(llm=llm, hooks=Hooks(on_iteration=rec.hook()))

    await _collect(loop.run(_user("Hi"), session_id="sess-xyz"))

    assert rec.args == [("sess-xyz", 1)]


async def test_on_iteration_session_id_is_none_when_unset() -> None:
    """Without a ``session_id`` the Loop-tier hooks receive ``None``."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])
    rec = IterationRecorder()
    loop = _make_loop(llm=llm, hooks=Hooks(on_iteration=rec.hook()))

    await _collect(loop.run(_user("Hi")))

    assert rec.args == [(None, 1)]


# ---------------------------------------------------------------------------
# on_llm_call
# ---------------------------------------------------------------------------


async def test_on_llm_call_fires_per_successful_call() -> None:
    """A 2-step run fires ``on_llm_call`` exactly twice, with real types."""
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output="ok")))
    llm = FakeLLMClient(
        replies=[
            make_response(_tool_json("look up", "search", {})),
            make_response(_final_json("done")),
        ]
    )
    rec = LlmCallRecorder()
    loop = _make_loop(llm=llm, registry=registry, hooks=Hooks(on_llm_call=rec.hook()))

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(rec.args) == 2
    for session_id, request, response, duration_ms in rec.args:
        assert session_id == "s1"
        assert isinstance(request, ChatRequest)
        assert isinstance(response, ChatResponse)
        assert isinstance(duration_ms, float)
        assert duration_ms >= 0.0


async def test_on_llm_call_response_matches_completion_in_complete_mode() -> None:
    """In non-stream mode ``on_llm_call`` gets the real provider ChatResponse."""
    llm = FakeLLMClient(replies=[make_response(_final_json("the answer"))])
    rec = LlmCallRecorder()
    loop = _make_loop(llm=llm, hooks=Hooks(on_llm_call=rec.hook()))

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(rec.args) == 1
    response = rec.args[0][2]
    assert isinstance(response, ChatResponse)
    assert response.message.content == _final_json("the answer")


async def test_on_llm_call_in_stream_mode_passes_synthesized_response() -> None:
    """In stream mode ``on_llm_call`` receives a synthesized ChatResponse."""
    chunks = make_stream_chunks([_final_json("streamed answer")])
    llm = FakeLLMClient(replies=[chunks])
    rec = LlmCallRecorder()
    loop = _make_loop(llm=llm, stream=True, hooks=Hooks(on_llm_call=rec.hook()))

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(rec.args) == 1
    response = rec.args[0][2]
    assert isinstance(response, ChatResponse)
    # The synthesized response carries the accumulated completion.
    assert response.message.content == _final_json("streamed answer")
    assert response.finish_reason == "stop"


async def test_on_llm_call_does_not_fire_on_llm_error() -> None:
    """An ``LLMError`` does NOT fire ``on_llm_call``; the run still terminates."""
    llm = FakeLLMClient(replies=[LLMError("provider down")])
    rec = LlmCallRecorder()
    loop = _make_loop(llm=llm, hooks=Hooks(on_llm_call=rec.hook()))

    events = await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert rec.args == []
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert isinstance(events[-1], FinalEvent)


# ---------------------------------------------------------------------------
# Sync vs async variants
# ---------------------------------------------------------------------------


async def test_loop_hooks_sync_variant() -> None:
    """Loop-tier hooks fire correctly with SYNC callables."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])
    it_rec = IterationRecorder(sync=True)
    llm_rec = LlmCallRecorder(sync=True)
    loop = _make_loop(
        llm=llm,
        hooks=Hooks(on_iteration=it_rec.hook(), on_llm_call=llm_rec.hook()),
    )

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(it_rec.args) == 1
    assert len(llm_rec.args) == 1


async def test_loop_hooks_async_variant() -> None:
    """Loop-tier hooks fire correctly with ASYNC callables."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])
    it_rec = IterationRecorder(sync=False)
    llm_rec = LlmCallRecorder(sync=False)
    loop = _make_loop(
        llm=llm,
        hooks=Hooks(on_iteration=it_rec.hook(), on_llm_call=llm_rec.hook()),
    )

    await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert len(it_rec.args) == 1
    assert len(llm_rec.args) == 1


# ---------------------------------------------------------------------------
# Raising hook is isolated
# ---------------------------------------------------------------------------


async def test_raising_on_iteration_does_not_abort_loop() -> None:
    """A raising ``on_iteration`` never breaks the loop."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])

    def boom(*_args: Any) -> None:
        raise RuntimeError("iteration hook down")

    loop = _make_loop(llm=llm, hooks=Hooks(on_iteration=boom))

    events = await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert isinstance(events[-1], FinalEvent)


async def test_raising_on_llm_call_does_not_abort_loop() -> None:
    """A raising ``on_llm_call`` never breaks the loop."""
    llm = FakeLLMClient(replies=[make_response(_final_json("done"))])

    def boom(*_args: Any) -> None:
        raise RuntimeError("llm-call hook down")

    loop = _make_loop(llm=llm, hooks=Hooks(on_llm_call=boom))

    events = await _collect(loop.run(_user("Hi"), session_id="s1"))

    assert isinstance(events[-1], FinalEvent)


# ---------------------------------------------------------------------------
# Zero-overhead: hooks=None
# ---------------------------------------------------------------------------


async def test_hooks_none_does_not_change_loop_behaviour() -> None:
    """With ``hooks=None`` the loop produces an identical event stream."""

    def build_llm() -> FakeLLMClient:
        return FakeLLMClient(
            replies=[
                make_response(_tool_json("look up", "search", {})),
                make_response(_final_json("done")),
            ]
        )

    registry_a = Registry()
    registry_a.register(FakeTool("search", result=ToolResult(output="ok")))
    loop_a = _make_loop(llm=build_llm(), registry=registry_a, hooks=None)
    events_a = await _collect(loop_a.run(_user("Hi")))

    registry_b = Registry()
    registry_b.register(FakeTool("search", result=ToolResult(output="ok")))
    loop_b = _make_loop(
        llm=build_llm(),
        registry=registry_b,
        hooks=Hooks(on_iteration=lambda *_a: None),
    )
    events_b = await _collect(loop_b.run(_user("Hi")))

    assert [type(e) for e in events_a] == [type(e) for e in events_b]
