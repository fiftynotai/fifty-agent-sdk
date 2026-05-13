"""AgentRunner — the user-facing orchestrator.

:class:`AgentRunner` is the SDK's top-level entry point for running an
agent across multiple conversational turns. It wraps an
:class:`agent_sdk.loop.AgentLoop` with conversation-state persistence
around each ``run()`` call:

1. Load prior messages from a :class:`agent_sdk.state.protocol.StateStore`.
2. Persist any first-turn ``system_prompt`` and the user's new message
   BEFORE driving the loop (durable proof of request).
3. Drive :meth:`AgentLoop.run` and forward every
   :class:`agent_sdk.streaming.AgentEvent` to the caller.
4. Persist the assistant's final answer ONLY on a clean
   :class:`agent_sdk.streaming.FinalEvent` with no preceding
   :class:`agent_sdk.streaming.ErrorEvent`.

System prompt vs. AgentLoop's structured prompt
    :class:`AgentLoop`'s ``prompts: PromptSections`` is the SDK-structured
    prompt rebuilt on every iteration from a tool-registry snapshot. It is
    NEVER persisted — it lives inside the loop's private working list and
    is the model's per-iteration reasoning scaffolding.

    :class:`AgentRunner`'s ``system_prompt: str | None`` is an OPTIONAL
    consumer-supplied kickoff message that, when set, is persisted as
    the first :class:`agent_sdk.llm.types.ChatMessage` with
    ``role="system"`` on the FIRST turn of a session only. It is the
    high-level instruction the consumer wants to ride alongside the
    SDK's structured prompt (for example, "You are a helpful
    customer-support agent"). Both coexist in the loop's prompt; that
    is intentional and supported by every major LLM provider.

Transactional persistence invariants
    * Every successful ``run()`` appends exactly one user message and
      exactly one assistant message to the state store. If
      ``system_prompt`` was set and the session was empty, exactly one
      :class:`ChatMessage` with ``role="system"`` is also appended,
      BEFORE the user message.
    * On any error during loop execution (LLMError, ParserError,
      iteration cap), the assistant message is NOT persisted; the user
      message remains durable. The fallback final answer is yielded to
      the caller but not committed to history.
    * On consumer cancellation, the assistant message is NOT persisted;
      the user message remains durable.
    * Tool roundtrips (``role="tool"`` messages) are NOT persisted to the
      state store. They live in the loop's private working list and are
      deterministically re-derivable from the assistant's final answer
      on the next turn. Tool-level audit provenance is BR-011's
      responsibility.

Logging
    Module-level :mod:`structlog` logger. ``INFO`` on run start and run
    end; ``ERROR`` only when persistence itself fails. Never logs prompt
    or message content — only lengths and counts.

    The ``runner.run_completed`` log carries ``terminated_by`` with one of:

    * ``"final_answer"`` — happy path: a clean :class:`FinalEvent` was
      yielded and the assistant message was persisted.
    * ``"error"`` — loop-internal failure (LLMError, ParserError,
      MaxIterationsExceeded). The loop's fallback FinalEvent is yielded
      but the assistant message is NOT persisted.
    * ``"state_store_error"`` — durability boundary failed. A
      :class:`agent_sdk.errors.StateStoreError` propagated out of one of
      the load/append calls. A companion ``phase`` field names which
      site failed: ``"load"``, ``"persist_system"``, ``"persist_user"``,
      or ``"persist_assistant"``.
    * ``"cancelled"`` — caller cancelled the consumer task or broke out
      of the ``async for`` via ``aclose()``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Final
from uuid import uuid4

import structlog

from agent_sdk.errors import StateStoreError
from agent_sdk.llm.types import ChatMessage
from agent_sdk.loop import AgentLoop
from agent_sdk.state.protocol import StateStore
from agent_sdk.streaming import AgentEvent, ErrorEvent, FinalEvent

_log: Final = structlog.get_logger(__name__)
"""Module-level structured logger. INFO at run boundaries; no content payloads."""


class AgentRunner:
    """User-facing orchestrator that drives an :class:`AgentLoop` with state.

    A typical end-to-end agent fits in roughly fifteen lines::

        from agent_sdk import (
            AgentLoop, AgentRunner, JsonModeParser, MemoryStateStore,
            OpenAICompatibleClient, PromptSections, Registry, SafetyConfig,
        )

        llm = OpenAICompatibleClient(...)
        registry = Registry()
        loop = AgentLoop(
            llm=llm, registry=registry, parser=JsonModeParser(),
            prompts=PromptSections(persona="You are helpful."),
            safety=SafetyConfig(), model="gpt-4o",
        )
        runner = AgentRunner(
            loop=loop, state=MemoryStateStore(),
            system_prompt="You are a helpful customer-support agent.",
        )
        async for event in runner.run("session-abc", "Hello"):
            print(event)

    Args:
        loop: The :class:`AgentLoop` instance to drive on each ``run()``
            call. The same loop is reused across turns.
        state: Any :class:`StateStore` implementation. Use
            :class:`MemoryStateStore` for ephemeral in-memory storage;
            BR-009/BR-010 ship durable backends.
        system_prompt: Optional consumer-supplied kickoff persisted as the
            FIRST :class:`ChatMessage` with ``role="system"`` on the first
            turn of each fresh session. When ``None`` (default), no system
            message is persisted; the loop's structured prompt does the
            entire job. See the module docstring for the precise boundary
            between this and ``AgentLoop.prompts``.

    Invariants:
        * Every ``run()`` either persists exactly one user message and
          exactly one assistant message, OR persists only the user
          message (on error or cancellation).
        * The optional ``system_prompt`` is persisted at most ONCE per
          session — only on the first ``run()`` call for that session.
        * Tool roundtrips are NOT persisted to state; the loop's working
          list carries them. Audit-level tool provenance is BR-011's
          responsibility.
        * A Runner-level ``run_id`` is generated per ``run()`` call for
          log correlation and is SEPARATE from the inner :class:`AgentLoop`
          run id. Neither id is exposed on :class:`AgentEvent` values.
    """

    def __init__(
        self,
        *,
        loop: AgentLoop,
        state: StateStore,
        system_prompt: str | None = None,
    ) -> None:
        self._loop = loop
        self._state = state
        self._system_prompt = system_prompt

    async def run(
        self, session_id: str, user_message: str
    ) -> AsyncIterator[AgentEvent]:
        """Drive a single conversational turn for ``session_id``.

        Algorithm (full edge-case detail in the module docstring):

        1. Load prior messages from the state store.
        2. If the session is empty AND ``system_prompt`` is set, append the
           system message to state BEFORE the user message.
        3. Append the user message to state BEFORE driving the loop. This
           is the load-bearing transactional property — if the loop
           later fails the user message remains durable.
        4. Drive :meth:`AgentLoop.run` with the loaded-plus-user
           conversation. Forward every event to the caller unchanged.
        5. If the run terminates with a :class:`FinalEvent` and no
           preceding :class:`ErrorEvent`, append the assistant message
           to state. Otherwise skip — the fallback final answer is
           yielded but not committed.

        On consumer cancellation (the consumer breaks out of the
        ``async for`` loop): :class:`asyncio.CancelledError` propagates
        untouched. The user message persisted in step 3 survives; no
        assistant message is persisted.

        On :class:`agent_sdk.errors.StateStoreError` raised by the state
        store: the error is logged at ``ERROR`` and re-raised. The Runner
        does NOT swallow state-store failures — the caller decides retry
        policy.

        Args:
            session_id: Opaque session identifier.
            user_message: The new user message text.

        Yields:
            :class:`AgentEvent` values forwarded from the inner
            :class:`AgentLoop` in monotonic ``sequence`` order. The
            terminal event is always a :class:`FinalEvent`.

        Raises:
            agent_sdk.errors.StateStoreError: If any state-store
                operation fails. The error is logged before being
                re-raised.
            asyncio.CancelledError: Propagated untouched from the loop
                or from the consumer's cancellation.
        """
        run_id = uuid4().hex

        # ── PHASE 1: LOAD ──────────────────────────────────────────────
        try:
            history = await self._state.get_messages(session_id)
        except StateStoreError as exc:
            _log.error(
                "runner.persist_failed",
                phase="load",
                session_id=session_id,
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            # No run_completed log here: we never entered the try/finally
            # block below, so there is nothing to summarise. The caller
            # gets the StateStoreError unchanged.
            raise
        is_first_turn = len(history) == 0
        _log.info(
            "runner.run_started",
            session_id=session_id,
            run_id=run_id,
            user_message_len=len(user_message),
            is_first_turn=is_first_turn,
            has_system_prompt=self._system_prompt is not None,
            has_prior_messages=not is_first_turn,
        )

        # Initial value is "interrupted" — neutral and applies to any
        # unexpected exit path (e.g. an exception escaping the loop that
        # we did not catch explicitly). The dedicated
        # ``except asyncio.CancelledError`` branch upgrades this to
        # ``"cancelled"`` ONLY when we can attribute exit to an actual
        # task/consumer cancellation.
        terminated_by = "interrupted"
        state_store_error_phase: str | None = None
        saw_error = False
        final_text: str | None = None
        event_count = 0

        try:
            # ── PHASE 2: PERSIST KICKOFF (FIRST TURN ONLY) ────────────
            if is_first_turn and self._system_prompt is not None:
                sys_msg = ChatMessage(role="system", content=self._system_prompt)
                try:
                    await self._state.append(session_id, sys_msg)
                except StateStoreError as exc:
                    state_store_error_phase = "persist_system"
                    _log.error(
                        "runner.persist_failed",
                        phase="persist_system",
                        session_id=session_id,
                        run_id=run_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise
                # Mirror locally — `history` was a defensive copy.
                history.append(sys_msg)

            # ── PHASE 3: PERSIST USER MESSAGE ─────────────────────────
            user_msg = ChatMessage(role="user", content=user_message)
            try:
                await self._state.append(session_id, user_msg)
            except StateStoreError as exc:
                state_store_error_phase = "persist_user"
                _log.error(
                    "runner.persist_failed",
                    phase="persist_user",
                    session_id=session_id,
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                raise
            history.append(user_msg)

            # ── PHASE 4: DRIVE THE LOOP ───────────────────────────────
            # AgentLoop adds its OWN structured system message to the
            # head of its working list — that is the per-iteration
            # reasoning scaffold (tool descriptions, output format
            # hints) and is separate from any role="system" message we
            # persisted in phase 2 (the consumer's kickoff). Both
            # coexist in the prompt; that is intentional.
            loop_messages = list(history)  # defensive copy for the loop

            async for event in self._loop.run(loop_messages):
                event_count += 1
                if isinstance(event, ErrorEvent):
                    saw_error = True
                elif isinstance(event, FinalEvent):
                    final_text = event.text
                yield event

            # ── PHASE 5: PERSIST ASSISTANT (SUCCESS PATH) ─────────────
            if not saw_error and final_text is not None:
                asst_msg = ChatMessage(role="assistant", content=final_text)
                try:
                    await self._state.append(session_id, asst_msg)
                except StateStoreError as exc:
                    state_store_error_phase = "persist_assistant"
                    _log.error(
                        "runner.persist_failed",
                        phase="persist_assistant",
                        session_id=session_id,
                        run_id=run_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise
                terminated_by = "final_answer"
            else:
                # Error path: an ErrorEvent was emitted; the loop's
                # fallback FinalEvent is yielded but NOT persisted, so
                # the next run() does not see a fake assistant turn.
                terminated_by = "error"
        except asyncio.CancelledError:
            # Caller cancelled the consumer task (or broke out via
            # ``aclose()``). Attribute exit to cancellation and let the
            # exception propagate untouched.
            terminated_by = "cancelled"
            raise
        finally:
            if state_store_error_phase is not None:
                terminated_by = "state_store_error"
            assistant_persisted = terminated_by == "final_answer"
            _log.info(
                "runner.run_completed",
                session_id=session_id,
                run_id=run_id,
                terminated_by=terminated_by,
                assistant_message_persisted=assistant_persisted,
                event_count=event_count,
                final_event_type="final" if final_text is not None else None,
                phase=state_store_error_phase,
            )


__all__ = ["AgentRunner"]
