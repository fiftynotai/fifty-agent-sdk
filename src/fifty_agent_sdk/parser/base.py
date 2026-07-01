"""Parser layer contracts: ``ParseResult`` discriminated union + ``Parser`` Protocol.

A parser converts an LLM completion string into a :data:`ParseResult` — a
Pydantic discriminated union that is either a :class:`ThoughtAction` (the model
chose to invoke a single tool), a :class:`MultiAction` (the model requested
multiple tool calls in one turn, reachable only from the native-tools parser
under BR-006), or a :class:`FinalAnswer` (the model produced a terminal
answer). Implementations are sync because parsing is pure-CPU work.

Implementations MUST raise :class:`fifty_agent_sdk.errors.ParserError` on malformed
input. The error's ``context`` dict carries at minimum a ``parser`` name and an
``error_phase`` string so callers can distinguish syntactic from semantic
failures (see the individual parser modules for the full context schema).

Branch on results with the ``kind`` field:

.. code-block:: python

    result = parser.parse(completion)
    if isinstance(result, ThoughtAction):
        ...  # dispatch result.tool_call
    else:
        ...  # surface result.content
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from fifty_agent_sdk.llm.types import ToolCall


class ThoughtAction(BaseModel):
    """Parser output: the model decided to invoke a tool.

    Attributes:
        kind: Literal discriminator; always ``"thought_action"``.
        thought: The model's reasoning that preceded the tool choice.
        tool_call: The tool invocation request, in the SDK's
            provider-agnostic :class:`fifty_agent_sdk.llm.types.ToolCall` shape.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["thought_action"] = "thought_action"
    thought: str
    tool_call: ToolCall


class FinalAnswer(BaseModel):
    """Parser output: the model produced a terminal answer.

    Attributes:
        kind: Literal discriminator; always ``"final_answer"``.
        thought: The model's reasoning that led to the final answer.
        content: The terminal answer text returned to the caller.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["final_answer"] = "final_answer"
    thought: str
    content: str


class MultiAction(BaseModel):
    """Parser output: the model requested MULTIPLE tool calls in one turn (BR-006).

    Reachable ONLY from :class:`fifty_agent_sdk.parser.native_tools.
    NativeToolsParser` when a provider-native response carries MORE THAN ONE
    ``tool_calls`` entry. A single native call yields
    :class:`ThoughtAction` (byte-identical to the pre-BR-006 path), and the
    text/JSON parsers NEVER produce this type — so the multi-call dispatch
    path is structurally unreachable from any text turn.

    Attributes:
        kind: Literal discriminator; always ``"multi_action"``.
        thought: The model's reasoning that preceded the calls. Today the
            native parser passes ``""`` (the provider's structured
            ``tool_calls`` carries no prose-thought channel); the field exists
            for future parsers and mirrors :class:`ThoughtAction.thought`.
        tool_calls: The list of tool invocation requests, in CALL order
            (the order the model emitted them). The loop dispatches them
            concurrently and feeds observations back in THIS order regardless
            of completion order, so replays are deterministic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["multi_action"] = "multi_action"
    thought: str
    tool_calls: list[ToolCall]


ParseResult = Annotated[
    ThoughtAction | FinalAnswer | MultiAction,
    Field(discriminator="kind"),
]
"""Tagged union of parser outputs.

The discriminator is the literal ``kind`` field. Consumers should branch on
:func:`isinstance` against :class:`ThoughtAction` / :class:`FinalAnswer` /
:class:`MultiAction` for the simplest pattern match. The annotated form is
suitable for use with :class:`pydantic.TypeAdapter` for programmatic
validation / round-tripping.
"""


@runtime_checkable
class Parser(Protocol):
    """Pluggable text-completion parser.

    Implementations consume the *complete* assistant completion string and
    return a :data:`ParseResult`. Parsing is pure-CPU; methods are sync.

    Implementations MUST raise :class:`fifty_agent_sdk.errors.ParserError` on
    malformed input. The error's ``context`` payload SHOULD carry the keys
    ``parser`` (the class name), ``error_phase`` (a short tag describing
    where the failure occurred), and ``completion_excerpt`` (a bounded
    prefix of the offending completion).

    The Protocol is :func:`runtime_checkable` so callers can validate
    structural conformance with :func:`isinstance` — useful for plugging in
    test fakes without subclassing.
    """

    def parse(self, completion: str) -> ParseResult:
        """Parse a completion string into a :data:`ParseResult`.

        Args:
            completion: The full assistant message content.

        Returns:
            Either a :class:`ThoughtAction` or :class:`FinalAnswer`.

        Raises:
            fifty_agent_sdk.errors.ParserError: When the completion cannot be
                parsed into a valid :data:`ParseResult`.
        """
        ...


__all__ = [
    "FinalAnswer",
    "MultiAction",
    "ParseResult",
    "Parser",
    "ThoughtAction",
]
