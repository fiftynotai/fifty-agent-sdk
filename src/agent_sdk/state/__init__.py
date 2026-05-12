"""State subpackage: pluggable conversation persistence.

Re-exports the :class:`StateStore` protocol and the default in-memory
implementation :class:`MemoryStateStore`. Durable backends (SQL, Redis)
ship in subsequent briefs (BR-009, BR-010) and will also be re-exported
here.
"""

from agent_sdk.state.memory import MemoryStateStore
from agent_sdk.state.protocol import StateStore

__all__ = ["MemoryStateStore", "StateStore"]
