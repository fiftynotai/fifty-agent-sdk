"""agent-sdk — Production-grade reusable agent loop SDK.

Foundation layer: typed errors, Pydantic v2 message types, the
:class:`LLMClient` protocol, an OpenAI-compatible adapter, and a system
prompt template builder. Higher layers (parser, tool registry, ReACT loop,
state stores) build on these contracts in subsequent briefs.

Optional ``sql`` extra surface
    :class:`SqlStateStore` and :data:`sql_metadata` are re-exported
    lazily — they require ``pip install 'agent-sdk[sql]'``. Importing
    :mod:`agent_sdk` itself does not pull SQLAlchemy. First access to
    either symbol triggers the import and raises a clear
    :class:`ImportError` if the extra is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_sdk.errors import (
    AgentSdkError,
    LLMError,
    MaxIterationsExceeded,
    ParserError,
    StateStoreError,
    ToolNotFound,
    ToolTimeout,
)
from agent_sdk.llm import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    LLMClient,
    OpenAICompatibleClient,
    Role,
    ToolCall,
    Usage,
)
from agent_sdk.loop import AgentLoop
from agent_sdk.parser import (
    FinalAnswer,
    JsonModeParser,
    NativeToolsParser,
    NativeToolsParserStub,
    Parser,
    ParseResult,
    ProseModeParser,
    ThoughtAction,
)
from agent_sdk.prompts import (
    JSON_MODE_OUTPUT_FORMAT,
    PROSE_MODE_OUTPUT_FORMAT,
    PromptSections,
    json_mode_template,
    prose_mode_template,
    render_system_prompt,
)
from agent_sdk.runner import AgentRunner
from agent_sdk.safety import SafetyConfig
from agent_sdk.state import MemoryStateStore, StateStore
from agent_sdk.streaming import (
    ActionEvent,
    AgentEvent,
    ErrorEvent,
    FinalEvent,
    ObservationEvent,
    ThoughtEvent,
    TokenEvent,
    ToolFailedEvent,
    ToolProgressEvent,
    ToolStartedEvent,
)
from agent_sdk.tools import (
    InProcProvider,
    Registry,
    Tool,
    ToolResult,
    ToolSchema,
    tool,
)

if TYPE_CHECKING:
    from agent_sdk.state.sql import SqlStateStore, sql_metadata

__version__ = "0.0.1"

__all__ = [
    "JSON_MODE_OUTPUT_FORMAT",
    "PROSE_MODE_OUTPUT_FORMAT",
    "ActionEvent",
    "AgentEvent",
    "AgentLoop",
    "AgentRunner",
    "AgentSdkError",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ErrorEvent",
    "FinalAnswer",
    "FinalEvent",
    "FinishReason",
    "InProcProvider",
    "JsonModeParser",
    "LLMClient",
    "LLMError",
    "MaxIterationsExceeded",
    "MemoryStateStore",
    "NativeToolsParser",
    "NativeToolsParserStub",
    "ObservationEvent",
    "OpenAICompatibleClient",
    "ParseResult",
    "Parser",
    "ParserError",
    "PromptSections",
    "ProseModeParser",
    "Registry",
    "Role",
    "SafetyConfig",
    "SqlStateStore",
    "StateStore",
    "StateStoreError",
    "ThoughtAction",
    "ThoughtEvent",
    "TokenEvent",
    "Tool",
    "ToolCall",
    "ToolFailedEvent",
    "ToolNotFound",
    "ToolProgressEvent",
    "ToolResult",
    "ToolSchema",
    "ToolStartedEvent",
    "ToolTimeout",
    "Usage",
    "json_mode_template",
    "prose_mode_template",
    "render_system_prompt",
    "sql_metadata",
    "tool",
]


def __getattr__(name: str) -> Any:
    """Lazily import SQL surface symbols on first access.

    ``SqlStateStore`` and ``sql_metadata`` require the optional ``sql``
    extra. Routing access through this hook keeps ``import agent_sdk``
    free of SQLAlchemy and surfaces a clear :class:`ImportError` only
    when the symbols are actually used without the extra installed.
    """
    if name in {"SqlStateStore", "sql_metadata"}:
        from agent_sdk.state import sql

        return getattr(sql, name)
    raise AttributeError(f"module 'agent_sdk' has no attribute {name!r}")
