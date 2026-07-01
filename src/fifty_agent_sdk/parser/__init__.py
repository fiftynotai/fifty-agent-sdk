"""Parser subpackage: pluggable strategies for extracting structured actions.

A parser converts an LLM completion (or, in the native-tools case, a
structured :class:`fifty_agent_sdk.llm.types.ChatResponse`) into a
:data:`ParseResult` — a tagged union of :class:`ThoughtAction` (the model
chose a single tool), :class:`MultiAction` (the model requested multiple
tool calls in one turn, BR-006), or :class:`FinalAnswer` (the model produced
a terminal answer).

Four concrete parsers ship today:

* :class:`JsonModeParser` — strict JSON envelope, the default when the loop
  uses :func:`fifty_agent_sdk.prompts.json_mode_template`.
* :class:`ProseModeParser` — classic ReACT prose, tolerant of whitespace and
  case variants.
* :class:`NativeToolsParser` — provider-native function-calling; consumes the
  structured ``tool_calls`` on a :class:`fifty_agent_sdk.llm.types.ChatResponse`
  (dispatched with precedence over text parsing when the adapter populates it).
  Satisfies the runtime-checkable :class:`NativeToolsParserProtocol`.
"""

from fifty_agent_sdk.parser.base import (
    FinalAnswer,
    MultiAction,
    Parser,
    ParseResult,
    ThoughtAction,
)
from fifty_agent_sdk.parser.json_mode import JsonModeParser
from fifty_agent_sdk.parser.native_tools import NativeToolsParser, NativeToolsParserProtocol
from fifty_agent_sdk.parser.prose_mode import ProseModeParser

__all__ = [
    "FinalAnswer",
    "JsonModeParser",
    "MultiAction",
    "NativeToolsParser",
    "NativeToolsParserProtocol",
    "ParseResult",
    "Parser",
    "ProseModeParser",
    "ThoughtAction",
]
