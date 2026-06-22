"""Parser subpackage: pluggable strategies for extracting structured actions.

A parser converts an LLM completion (or, in the native-tools case, a
structured :class:`fifty_agent_sdk.llm.types.ChatResponse`) into a
:data:`ParseResult` — a tagged union of :class:`ThoughtAction` (the model
chose a tool) or :class:`FinalAnswer` (the model produced a terminal answer).

Three concrete parsers ship today:

* :class:`JsonModeParser` — strict JSON envelope, the default when the loop
  uses :func:`fifty_agent_sdk.prompts.json_mode_template`.
* :class:`ProseModeParser` — classic ReACT prose, tolerant of whitespace and
  case variants.
* :class:`NativeToolsParserStub` — placeholder for provider-native tool
  calling; raises :class:`NotImplementedError` until that integration lands.
"""

from fifty_agent_sdk.parser.base import FinalAnswer, Parser, ParseResult, ThoughtAction
from fifty_agent_sdk.parser.json_mode import JsonModeParser
from fifty_agent_sdk.parser.native_tools import NativeToolsParser, NativeToolsParserStub
from fifty_agent_sdk.parser.prose_mode import ProseModeParser

__all__ = [
    "FinalAnswer",
    "JsonModeParser",
    "NativeToolsParser",
    "NativeToolsParserStub",
    "ParseResult",
    "Parser",
    "ProseModeParser",
    "ThoughtAction",
]
