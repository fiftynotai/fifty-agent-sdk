"""StateStore protocol — pluggable conversation persistence.

The :class:`StateStore` protocol is the SDK's abstraction for durable
conversation history. :class:`agent_sdk.runner.AgentRunner` calls it on
every turn to load prior messages, persist the user message before driving
the loop, and persist the assistant's final answer on a clean termination.

Implementations may be in-memory (see :class:`agent_sdk.state.memory.
MemoryStateStore` — the default for ephemeral use), or backed by SQL
(BR-009), Redis (BR-010), or any other durable store.

Contract invariants
    * **Ordering.** Messages are returned in append order. Implementations
      MUST NOT reorder, deduplicate, or merge messages.
    * **TTL.** Implementations MAY apply TTL eviction. When they do,
      :meth:`get_messages` returns whatever has not yet expired (an empty
      list is a valid result; it MUST NOT raise).
    * **Empty sessions.** :meth:`get_messages` on a never-seen
      ``session_id`` returns ``[]``. It MUST NOT raise.
    * **Defensive copy.** :meth:`get_messages` returns a freshly
      constructed list; callers mutating it MUST NOT see those mutations
      reflected in the store.
    * **Atomicity.** :meth:`append` is atomic with respect to concurrent
      reads on the same session — readers see either pre- or post-append
      state, never a partially-written list.
    * **Idempotent delete.** :meth:`delete` on an unknown ``session_id`` is
      a silent no-op.
    * **Backend failures.** Implementations MUST wrap backend-level
      failures (SQL exceptions, Redis disconnects, IO errors) in
      :class:`agent_sdk.errors.StateStoreError` with ``context`` carrying
      the originating exception type and the ``session_id``. Programmer
      errors (passing ``None``, type violations) raise the appropriate
      built-in exception. :class:`MemoryStateStore` essentially never
      raises :class:`StateStoreError` because dict operations have no
      plausible backend failure mode.
    * **Opaque session ids.** A ``session_id`` is an opaque string; the
      store performs no validation beyond what its backend requires.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_sdk.llm.types import ChatMessage


@runtime_checkable
class StateStore(Protocol):
    """Pluggable conversation-state backend.

    See the module docstring for the full contract. Implementations are
    duck-typed: any class providing the three async methods below with
    matching signatures satisfies :func:`isinstance` against
    :class:`StateStore` thanks to ``@runtime_checkable``.

    Note:
        ``@runtime_checkable`` :class:`Protocol` instances only check for
        method *presence*, not signature compatibility. Mypy ``--strict``
        catches signature mismatches at type-check time; downstream tests
        that pass a structurally-correct fake will also pass
        :func:`isinstance`.
    """

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Load the messages persisted for ``session_id``.

        Args:
            session_id: Opaque session identifier.

        Returns:
            A freshly-constructed list of :class:`ChatMessage` values in
            append order. An empty list for an unknown session is a valid,
            non-error result.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails for an implementation-specific reason (SQL/Redis/IO).
        """
        ...

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's ordered message log.

        Atomic with respect to concurrent reads on the same session: a
        reader either sees the message or not, never a half-written list.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Remove all persisted state for ``session_id``.

        Idempotent — calling on an unknown ``session_id`` is a silent no-op.

        Args:
            session_id: Opaque session identifier.

        Raises:
            agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...


__all__ = ["StateStore"]
