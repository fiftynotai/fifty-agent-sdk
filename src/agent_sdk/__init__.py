"""agent-sdk — Production-grade reusable agent loop SDK.

Foundation layer: typed errors, Pydantic v2 message types, the
:class:`LLMClient` protocol, an OpenAI-compatible adapter, and a system
prompt template builder. Higher layers (parser, tool registry, ReACT loop,
state stores) build on these contracts in subsequent briefs.
"""

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
from agent_sdk.tools import (
    InProcProvider,
    Registry,
    Tool,
    ToolResult,
    ToolSchema,
    tool,
)

__version__ = "0.0.1"

__all__ = [
    "JSON_MODE_OUTPUT_FORMAT",
    "PROSE_MODE_OUTPUT_FORMAT",
    "AgentSdkError",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "FinalAnswer",
    "FinishReason",
    "InProcProvider",
    "JsonModeParser",
    "LLMClient",
    "LLMError",
    "MaxIterationsExceeded",
    "NativeToolsParser",
    "NativeToolsParserStub",
    "OpenAICompatibleClient",
    "ParseResult",
    "Parser",
    "ParserError",
    "PromptSections",
    "ProseModeParser",
    "Registry",
    "Role",
    "StateStoreError",
    "ThoughtAction",
    "Tool",
    "ToolCall",
    "ToolNotFound",
    "ToolResult",
    "ToolSchema",
    "ToolTimeout",
    "Usage",
    "json_mode_template",
    "prose_mode_template",
    "render_system_prompt",
    "tool",
]
