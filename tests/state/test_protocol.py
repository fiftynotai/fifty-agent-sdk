"""Runtime-checkable conformance tests for the :class:`StateStore` protocol.

``@runtime_checkable`` :class:`typing.Protocol` only checks for method
*presence*. These tests pin that behavior so the contract is intentional.
"""

from __future__ import annotations

from fifty_agent_sdk import BranchInfo, ChatMessage, MemoryStateStore, StateStore


def test_memory_store_satisfies_protocol() -> None:
    """The default in-memory implementation passes ``isinstance``."""
    store = MemoryStateStore()
    assert isinstance(store, StateStore)


def test_duck_typed_fake_satisfies_protocol() -> None:
    """A structurally-correct fake (no inheritance) passes ``isinstance``."""

    class DuckStore:
        async def get_messages(
            self, session_id: str, *, branch_id: str | None = None
        ) -> list[ChatMessage]:
            return []

        async def append(self, session_id: str, message: ChatMessage) -> None:
            return None

        async def delete(self, session_id: str) -> None:
            return None

        async def fork(self, session_id: str, from_sequence: int) -> str:
            return ""

        async def list_branches(self, session_id: str) -> list[BranchInfo]:
            return []

        async def switch_branch(self, session_id: str, branch_id: str) -> None:
            return None

        async def truncate_after(
            self, session_id: str, sequence: int, *, branch_id: str | None = None
        ) -> None:
            return None

    duck = DuckStore()
    assert isinstance(duck, StateStore)


def test_missing_method_fails_protocol_check() -> None:
    """A class lacking ``delete`` does NOT pass ``isinstance``."""

    class Incomplete:
        async def get_messages(self, session_id: str) -> list[ChatMessage]:
            return []

        async def append(self, session_id: str, message: ChatMessage) -> None:
            return None

        # Missing `delete`.

    bad = Incomplete()
    assert not isinstance(bad, StateStore)


def test_protocol_is_exported_from_top_level() -> None:
    """``StateStore`` is importable from the package root."""
    from fifty_agent_sdk import StateStore as _StateStore

    assert _StateStore is StateStore
