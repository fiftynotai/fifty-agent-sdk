"""Shared helpers for the runner integration tests.

Reuses :class:`tests.loop.conftest.FakeLLMClient` and
:class:`tests.loop.conftest.FakeTool` so the runner tests exercise the
SAME plumbing the loop integration tests do — keeping a single source of
truth for LLM/tool stand-ins.

The :func:`_make_runner` helper composes a full
``loop + state + runner`` stack mirroring the shape of
:func:`tests.loop.test_loop._make_loop`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fifty_agent_sdk import (
    AgentEvent,
    AgentLoop,
    AgentRunner,
    JsonModeParser,
    MemoryStateStore,
    PromptSections,
    Registry,
    SafetyConfig,
)
from fifty_agent_sdk.audit.protocol import AuditSink
from fifty_agent_sdk.llm.types import ChatRequest, ChatResponse
from fifty_agent_sdk.observability import Hooks
from fifty_agent_sdk.state.protocol import StateStore
from tests.loop.conftest import FakeLLMClient, make_response


def final_json(answer: str, thought: str = "done") -> str:
    """Render a JSON envelope for a :class:`FinalAnswer` parse."""
    return json.dumps(
        {
            "thought": thought,
            "action": "final",
            "tool_name": None,
            "tool_args": None,
            "answer": answer,
        }
    )


def tool_json(thought: str, name: str, args: dict[str, Any] | None) -> str:
    """Render a JSON envelope for a :class:`ThoughtAction` parse."""
    return json.dumps(
        {
            "thought": thought,
            "action": "tool",
            "tool_name": name,
            "tool_args": args,
            "answer": None,
        }
    )


def make_runner(
    *,
    llm: FakeLLMClient,
    registry: Registry | None = None,
    state: StateStore | None = None,
    safety: SafetyConfig | None = None,
    system_prompt: str | None = None,
    persona: str = "You are a helpful agent.",
    audit: AuditSink | None = None,
    hooks: Hooks | None = None,
) -> tuple[AgentRunner, StateStore]:
    """Build a ready-to-drive :class:`AgentRunner` and return it with its store.

    The store is returned alongside the runner so tests can assert on
    persisted history without reaching into private state.

    Pass ``audit`` to wire an :class:`fifty_agent_sdk.audit.protocol.AuditSink`
    into the runner; left ``None`` (default) the runner emits no audit
    events.

    Pass ``hooks`` to wire an :class:`fifty_agent_sdk.observability.Hooks` into the
    stack. The SAME instance is threaded into BOTH the :class:`AgentLoop`
    and the :class:`AgentRunner` — the consumer-shares-one-instance pattern
    — so the two Loop-tier hooks and the five Runner-tier hooks all fire
    from one wired stack. Left ``None`` (default) no hooks fire.
    """
    loop = AgentLoop(
        llm=llm,
        registry=registry if registry is not None else Registry(),
        parser=JsonModeParser(),
        prompts=PromptSections(persona=persona),
        safety=safety if safety is not None else SafetyConfig(),
        model="test-model",
        hooks=hooks,
    )
    store = state if state is not None else MemoryStateStore()
    runner = AgentRunner(
        loop=loop,
        state=store,
        system_prompt=system_prompt,
        audit=audit,
        hooks=hooks,
    )
    return runner, store


async def collect(iterator: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    """Drain an :class:`AsyncIterator[AgentEvent]` to a list."""
    return [event async for event in iterator]


class FormatAwareFakeLLM:
    """FakeLLM that mirrors real provider behavior: if the most recent assistant
    message in the inbound ChatRequest is a JSON envelope (parses + has `action`
    key), reply with json_reply; otherwise reply with prose_reply.

    Locks BR-016: a runner that persists only the parsed answer feeds prose
    back on turn 2, this fake then replies with prose, JsonModeParser raises
    ParserError. The fix persists the raw envelope so turn 2 stays in format.
    """

    def __init__(self, *, json_reply: str, prose_reply: str) -> None:
        self._json_reply = json_reply
        self._prose_reply = prose_reply
        self.calls: list[ChatRequest] = []

    def _select_reply(self, request: ChatRequest) -> str:
        """Pick ``json_reply`` or ``prose_reply`` based on the prior assistant turn.

        Scans ``request.messages`` in reverse for the most recent
        ``role="assistant"`` message:

        * If its ``content`` parses as JSON AND the resulting dict has an
          ``"action"`` key → format-following provider behavior is the
          ``json_reply``.
        * Otherwise (content is prose, or parses but lacks ``"action"``) →
          the provider has drifted out of format and produces ``prose_reply``,
          which is what JsonModeParser will reject as a ParserError.
        * If no assistant message exists yet (turn 1) → default to
          ``json_reply``: the system prompt is the format-pinning signal on
          the first turn.
        """
        for message in reversed(request.messages):
            if message.role != "assistant":
                continue
            try:
                parsed = json.loads(message.content)
            except (TypeError, ValueError, json.JSONDecodeError):
                return self._prose_reply
            if isinstance(parsed, dict) and "action" in parsed:
                return self._json_reply
            return self._prose_reply
        return self._json_reply

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        return make_response(self._select_reply(request))

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        self.calls.append(request)
        yield make_response(self._select_reply(request))
