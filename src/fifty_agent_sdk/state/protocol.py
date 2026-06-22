"""StateStore protocol — pluggable conversation persistence with branching.

The :class:`StateStore` protocol is the SDK's abstraction for durable
conversation history. :class:`fifty_agent_sdk.runner.AgentRunner` calls it on
every turn to load prior messages, persist the user message before driving
the loop, and persist the assistant's final answer on a clean termination.

Implementations may be in-memory (see :class:`fifty_agent_sdk.state.memory.
MemoryStateStore` — the default for ephemeral use), or backed by SQL
(BR-009), Redis (BR-010), or any other durable store.

Branching (BR-004)
    A session is a **tree** of branches, not a single linear log. Every
    session starts on an implicit ``"trunk"`` branch (:data:`TRUNK_BRANCH_ID`);
    existing pre-branching sessions are read as trunk with zero data loss.
    :meth:`fork` creates a new branch from a point in the active branch's
    history (the old line stays reachable — this is the "edit a message /
    regenerate" model); :meth:`switch_branch` moves the active head;
    :meth:`list_branches` enumerates the tree; and :meth:`get_messages`
    takes an optional ``branch_id`` for branch-scoped reads. :meth:`append`
    always writes to the **active** branch.

    *Sequence model.* ``sequence`` is a 1-based position in a branch's
    **materialized** history (its lineage from the root concatenated with
    its own messages). It is per-branch monotonic, anchored at the fork
    point: a branch forked from the trunk at sequence ``3`` inherits
    messages ``1..3`` and its own first append is sequence ``4``. A read of
    that branch returns ``trunk[1..3] + branch[4..head]``.

Contract invariants
    * **Ordering.** Messages are returned in append order. Implementations
      MUST NOT reorder, deduplicate, or merge messages.
    * **TTL.** Implementations MAY apply TTL eviction. When they do,
      :meth:`get_messages` returns whatever has not yet expired (an empty
      list is a valid result; it MUST NOT raise).
    * **Empty / unknown sessions.** :meth:`get_messages` and
      :meth:`list_branches` on a never-seen ``session_id`` return ``[]``;
      they MUST NOT raise.
    * **Active branch default.** ``branch_id=None`` resolves to the session's
      active branch (``"trunk"`` until a :meth:`switch_branch`). An *explicit*
      ``branch_id`` that does not exist is a programmer error and raises
      :class:`ValueError`.
    * **Defensive copy.** :meth:`get_messages` returns a freshly
      constructed list; callers mutating it MUST NOT see those mutations
      reflected in the store.
    * **Atomicity.** :meth:`append` is atomic with respect to concurrent
      reads on the same session — readers see either pre- or post-append
      state, never a partially-written list.
    * **Idempotent delete.** :meth:`delete` removes the WHOLE session (all
      branches). On an unknown ``session_id`` it is a silent no-op.
    * **Backend failures.** Implementations MUST wrap backend-level
      failures (SQL exceptions, Redis disconnects, IO errors) in
      :class:`fifty_agent_sdk.errors.StateStoreError` with ``context`` carrying
      the originating exception type and the ``session_id``. Programmer
      errors (passing ``None``, type violations, unknown branch ids) raise
      the appropriate built-in exception. :class:`MemoryStateStore`
      essentially never raises :class:`StateStoreError` because dict
      operations have no plausible backend failure mode.
    * **Opaque session ids.** A ``session_id`` is an opaque string; the
      store performs no validation beyond what its backend requires.

Reserved for BR-003 (not part of this protocol yet)
    The destructive sibling primitive
    ``truncate_after(session_id, sequence, *, branch_id=None)`` (hard-delete
    of ``sequence > N`` on a branch) is intentionally NOT on this protocol
    yet — it lands in BR-003. Its branch-aware signature is fixed here so the
    two features compose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from fifty_agent_sdk.llm.types import ChatMessage

TRUNK_BRANCH_ID = "trunk"
"""The canonical id of the implicit first branch every session starts on.

Pre-branching sessions (and rows migrated from before BR-004) are read as
this branch, so the feature is additive with zero data loss.
"""


@dataclass(frozen=True)
class BranchInfo:
    """Metadata describing one branch of a session's conversation tree.

    Returned by :meth:`StateStore.list_branches`.

    Attributes:
        branch_id: The branch's identifier (``"trunk"`` for the root, an
            opaque generated id for forks).
        parent_branch_id: The branch this one was forked from, or ``None``
            for the trunk.
        forked_from_sequence: The sequence in the parent's materialized
            history at which this branch diverged, or ``None`` for the trunk.
        head_sequence: The sequence of this branch's most recent message in
            its materialized history (equal to ``forked_from_sequence`` for a
            freshly-forked branch with no own messages yet; ``0`` for an empty
            trunk).
        created_at: When the branch was created (timezone-aware). The trunk's
            timestamp is the session's first-append time.
        is_active: Whether this branch is the session's current active head.
    """

    branch_id: str
    parent_branch_id: str | None
    forked_from_sequence: int | None
    head_sequence: int
    created_at: datetime
    is_active: bool


@runtime_checkable
class StateStore(Protocol):
    """Pluggable conversation-state backend with first-class branching.

    See the module docstring for the full contract. Implementations are
    duck-typed: any class providing the async methods below with matching
    signatures satisfies :func:`isinstance` against :class:`StateStore`
    thanks to ``@runtime_checkable``.

    Note:
        ``@runtime_checkable`` :class:`Protocol` instances only check for
        method *presence*, not signature compatibility. Mypy ``--strict``
        catches signature mismatches at type-check time; downstream tests
        that pass a structurally-correct fake will also pass
        :func:`isinstance`. Because BR-004 added the branching methods to
        this protocol, any external implementation MUST provide them — this
        is a deliberate breaking change.
    """

    async def get_messages(
        self, session_id: str, *, branch_id: str | None = None
    ) -> list[ChatMessage]:
        """Load the materialized messages for a branch of ``session_id``.

        Args:
            session_id: Opaque session identifier.
            branch_id: Which branch to read. ``None`` (the default) reads the
                session's active branch. An explicit, unknown ``branch_id``
                raises :class:`ValueError`.

        Returns:
            A freshly-constructed list of :class:`ChatMessage` values in
            append order — the branch's full materialized history (lineage
            from the root concatenated with the branch's own messages). An
            empty list for an unknown session (with ``branch_id=None``) is a
            valid, non-error result.

        Raises:
            ValueError: If an explicit ``branch_id`` does not exist.
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails for an implementation-specific reason (SQL/Redis/IO).
        """
        ...

    async def append(self, session_id: str, message: ChatMessage) -> None:
        """Append ``message`` to the session's **active** branch.

        Atomic with respect to concurrent reads on the same session: a
        reader either sees the message or not, never a half-written list.
        A never-seen session is created implicitly on the ``"trunk"`` branch.

        Args:
            session_id: Opaque session identifier.
            message: The :class:`ChatMessage` to append.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Remove ALL persisted state for ``session_id`` (every branch).

        Idempotent — calling on an unknown ``session_id`` is a silent no-op.

        Args:
            session_id: Opaque session identifier.

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...

    async def fork(self, session_id: str, from_sequence: int) -> str:
        """Fork the active branch at ``from_sequence`` into a new branch.

        The new branch inherits the active branch's materialized history up
        to and including ``from_sequence``; subsequent :meth:`append`\\ s to it
        continue the sequence from there. The original branch is untouched and
        remains reachable (the non-destructive "edit / regenerate" model).
        This does NOT change the active head — call :meth:`switch_branch` to
        move onto the new branch.

        Args:
            session_id: Opaque session identifier (must already exist).
            from_sequence: The sequence in the active branch to fork from
                (``0`` forks from an empty history; must be in
                ``0..head_sequence`` of the active branch).

        Returns:
            The generated ``branch_id`` of the new branch.

        Raises:
            ValueError: If the session is unknown, or ``from_sequence`` is
                outside ``0..head_sequence`` of the active branch.
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...

    async def list_branches(self, session_id: str) -> list[BranchInfo]:
        """Enumerate all branches of ``session_id``.

        Args:
            session_id: Opaque session identifier.

        Returns:
            A list of :class:`BranchInfo` (trunk first, then by creation
            order). An unknown session yields ``[]`` (not an error).

        Raises:
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...

    async def switch_branch(self, session_id: str, branch_id: str) -> None:
        """Set the session's active head to ``branch_id``.

        Subsequent :meth:`append`\\ s target this branch and
        ``get_messages(session_id)`` reads it.

        Args:
            session_id: Opaque session identifier.
            branch_id: The branch to activate.

        Raises:
            ValueError: If ``branch_id`` does not exist for this session.
            fifty_agent_sdk.errors.StateStoreError: If the backend operation
                fails.
        """
        ...


__all__ = ["TRUNK_BRANCH_ID", "BranchInfo", "StateStore"]
