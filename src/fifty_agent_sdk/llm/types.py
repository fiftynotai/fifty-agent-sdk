"""Pydantic v2 data models for the LLM contract.

These models are the *provider-agnostic surface* of the SDK. Every
:class:`fifty_agent_sdk.llm.protocol.LLMClient` implementation accepts and returns
exclusively these types — provider-specific envelopes are translated at the
adapter boundary.

All models set ``extra="forbid"`` so unknown fields raise a validation error
instead of silently passing through. This is intentional: the SDK should
catch typos and provider-drift early rather than letting unknown attributes
ride along to consumers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
"""Discriminator for the speaker of a :class:`ChatMessage`."""

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "in_progress"]
"""Standard terminal reason for a chat completion — plus the streaming sentinel.

Mirrors the OpenAI chat-completion ``finish_reason`` field. Adapters MUST map
provider-specific values into one of these literals.

The ``"in_progress"`` value is a streaming-only sentinel emitted on
intermediate chunks where the upstream provider has not yet reported a
terminal reason. Consumers can branch on it without misreading an
intermediate delta as a terminal ``"stop"``.
"""


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Who is speaking. One of :data:`Role`.
        content: The textual content of the message. An empty string is
            permitted (for example, an assistant turn that contains only
            tool calls).
        name: Optional name for a function/tool message or a named speaker.
        tool_call_id: Identifier echoed back on a ``role="tool"`` reply so
            the model can match it to the originating tool call.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """A model-issued tool invocation request.

    The provider-agnostic shape: a tool name plus a dict of arguments. Adapters
    are responsible for translating provider-specific tool-call envelopes into
    this shape and back.

    Attributes:
        name: Name of the tool to invoke. Must match a registered tool.
        args: Arguments to pass to the tool. Defaults to an empty dict.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token accounting for a single LLM call.

    All counts are non-negative integers. When a provider does not return
    usage data (for example, partway through a stream), adapters return zeros
    for the missing fields rather than ``None``.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens emitted in the completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatRequest(BaseModel):
    """Provider-agnostic chat-completion request.

    Attributes:
        messages: Ordered list of conversation messages.
        model: Model identifier. Adapters may override this with a default.
        temperature: Sampling temperature in ``[0.0, 2.0]``. Default ``0.0``.
        max_tokens: Optional cap on completion tokens. Must be ``>= 1`` if set.
        response_format: Optional provider-format hint. Common values are
            ``{"type": "json_object"}`` or ``{"type": "text"}``. Adapters
            pass this through verbatim where supported.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    response_format: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Provider-agnostic chat-completion response.

    For non-streaming responses, ``message.content`` holds the full completion.
    For streaming responses, each yielded :class:`ChatResponse` chunk carries
    only the delta in ``message.content`` (consumers accumulate). Intermediate
    chunks have ``finish_reason='in_progress'``; only the final chunk carries a
    real terminal reason (``stop`` / ``length`` / ``tool_calls`` /
    ``content_filter`` / ``error``). Usage figures may be zero on intermediate
    chunks if the provider omits them.

    Attributes:
        message: The assistant's message for this response (or chunk delta).
        usage: Token accounting. Zero-filled when a provider omits counts.
        finish_reason: Why generation stopped. One of :data:`FinishReason`.
    """

    model_config = ConfigDict(extra="forbid")

    message: ChatMessage
    usage: Usage
    finish_reason: FinishReason


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "FinishReason",
    "Role",
    "ToolCall",
    "Usage",
]
