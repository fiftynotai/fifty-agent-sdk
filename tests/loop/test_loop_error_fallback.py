"""Which final text ends each terminal path of the loop (BR-021).

Only the iteration cap (``error_type="MaxIterationsExceeded"``) ends with
``SafetyConfig.fallback_message``. Every other ``ErrorEvent`` (an LLM error,
the native parser error, a text parser error with the retry disabled or used
up) is followed by ``SafetyConfig.error_fallback_message``, so an end user is
not told the steps ran out when the provider failed.

The three end-to-end tests drive the real ``OpenAICompatibleClient`` through
``pytest-httpx``: a provider answering HTTP 200 with a plain-text over-length
error ends the run after one request (streamed or not) with the classified
type in the ``ErrorEvent`` and without the provider's text in the
``FinalEvent``; on the non-streamed run the classified type is also on the
log line, and no structlog entry carries the provider's text.

What these do NOT pin: the Runner's audit payload and persistence
(``tests/runner/test_runner_audit.py``), and ``stdlib`` logging the ``openai``
client may do itself.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from pytest_httpx import HTTPXMock

from fifty_agent_sdk import (
    AgentEvent,
    AgentLoop,
    ChatMessage,
    ErrorEvent,
    FinalEvent,
    JsonModeParser,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    ToolResult,
)
from fifty_agent_sdk.errors import LLMError
from fifty_agent_sdk.llm.protocol import LLMClient
from tests.loop.conftest import FakeLLMClient, FakeTool, make_multi_tool_response, make_response

_USER = [ChatMessage(role="user", content="How long is my contract?")]
_TOOL_STEP = (
    '{"thought": "look it up", "action": "tool", "tool_name": "lookup", '
    '"tool_args": {}, "answer": null}'
)
# A provider's own over-length text, answered with HTTP 200. "SENTINEL" makes
# a leak easy to find; 145048 is the number the end user must not see.
_PROVIDER_TEXT = "Prompt length 145048 exceeds max_prompt_length 131072 SENTINEL-br021"
_ENDPOINT = "https://example.com/v1/chat/completions"


def _loop(
    llm: LLMClient,
    *,
    safety: SafetyConfig,
    stream: bool = False,
    tool_mode: ToolMode | None = None,
) -> AgentLoop:
    registry = Registry()
    registry.register(FakeTool("lookup", result=ToolResult(output="ok")))
    kwargs: dict[str, Any] = {}
    if tool_mode is None:
        kwargs["parser"] = JsonModeParser()
    else:
        kwargs["tool_mode"] = tool_mode
    return AgentLoop(
        llm=llm,
        registry=registry,
        prompts=PromptSections(persona="You are a helpful agent."),
        safety=safety,
        model="test-model",
        stream=stream,
        **kwargs,
    )


async def _run(loop: AgentLoop) -> list[AgentEvent]:
    return [event async for event in loop.run(_USER)]


def _real_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key", base_url="https://example.com/v1", timeout=5.0, max_retries=0
    )


# --- Every path that ends with an ErrorEvent, with both texts customised -----------


_PATHS: dict[str, dict[str, Any]] = {
    "llm_error": {
        "replies": [LLMError("provider down", context={"type": "BadRequestError"})],
        "error_type": "LLMError",
    },
    "llm_error_stream": {
        "replies": [LLMError("provider down mid-stream", context={"type": "APIError"})],
        "stream": True,
        "error_type": "LLMError",
    },
    "native_parser_error": {
        "replies": [make_multi_tool_response([("", {})])],
        "tool_mode": ToolMode.NATIVE,
        "error_type": "ParserError",
    },
    "text_parser_retry_exhausted": {
        "replies": [make_response("drift one"), make_response("drift two")],
        "error_type": "ParserError",
    },
    "text_parser_retry_disabled": {
        "replies": [make_response("drift")],
        "parser_retry_enabled": False,
        "error_type": "ParserError",
    },
    "iteration_cap": {
        "replies": [make_response(_TOOL_STEP)],
        "max_iterations": 1,
        "error_type": "MaxIterationsExceeded",
    },
}


@pytest.mark.parametrize("path", list(_PATHS))
async def test_each_terminal_path_ends_with_its_own_final_text(path: str) -> None:
    """Only the iteration cap ends with fallback_message; every other ErrorEvent with error_fallback_message (BR-021)."""
    spec = _PATHS[path]
    safety = SafetyConfig(
        fallback_message="CAP-TEXT",
        error_fallback_message="ERROR-TEXT",
        parser_retry_enabled=spec.get("parser_retry_enabled", True),
        max_iterations=spec.get("max_iterations", 10),
    )
    llm = FakeLLMClient(list(spec["replies"]))
    loop = _loop(
        llm, safety=safety, stream=spec.get("stream", False), tool_mode=spec.get("tool_mode")
    )

    events = await _run(loop)

    assert [type(e) for e in events[-2:]] == [ErrorEvent, FinalEvent]
    error, final = events[-2], events[-1]
    assert isinstance(error, ErrorEvent) and isinstance(final, FinalEvent)
    assert error.error_type == spec["error_type"]
    expected = "CAP-TEXT" if path == "iteration_cap" else "ERROR-TEXT"
    assert final.text == expected
    assert final.raw_completion is None


async def test_default_error_text_differs_from_the_step_limit_text() -> None:
    """With default SafetyConfig an LLM error and the cap end with two different texts (BR-021 AC-3 control)."""
    error_run = await _run(_loop(FakeLLMClient([LLMError("down")]), safety=SafetyConfig()))
    cap_run = await _run(
        _loop(FakeLLMClient([make_response(_TOOL_STEP)]), safety=SafetyConfig(max_iterations=1))
    )

    error_final, cap_final = error_run[-1], cap_run[-1]
    assert isinstance(error_final, FinalEvent) and isinstance(cap_final, FinalEvent)
    assert error_final.text == SafetyConfig().error_fallback_message
    assert cap_final.text == SafetyConfig().fallback_message
    assert error_final.text != cap_final.text


async def test_setting_only_fallback_message_leaves_the_error_text_at_its_default() -> None:
    """A consumer that sets only fallback_message still gets the default error text on an LLM error (BR-021)."""
    safety = SafetyConfig(fallback_message="custom cap text")

    events = await _run(_loop(FakeLLMClient([LLMError("down")]), safety=safety))

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.text == SafetyConfig().error_fallback_message
    assert final.text != "custom cap text"


# --- End to end through the real client ---------------------------------------------


async def test_provider_text_body_ends_the_run_with_the_error_text(httpx_mock: HTTPXMock) -> None:
    """A 200 plain-text over-length body ends the run: classified ErrorEvent, error text, one request (BR-021).

    Before BR-021 this was ``MalformedResponse`` and the step-limit text.
    """
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINT,
        content=_PROVIDER_TEXT.encode(),
        headers={"content-type": "text/plain"},
    )

    events = await _run(_loop(_real_client(), safety=SafetyConfig()))

    assert [type(e) for e in events] == [ErrorEvent, FinalEvent]
    error, final = events
    assert isinstance(error, ErrorEvent) and isinstance(final, FinalEvent)
    assert error.error_type == "LLMError"
    assert error.context["type"] == "ContextLengthExceeded"
    assert error.context["classified_from"] == "NonJsonProviderBody"
    assert _PROVIDER_TEXT in error.message
    assert final.text == SafetyConfig().error_fallback_message
    assert "145048" not in final.text
    assert "SENTINEL" not in final.text
    assert len(httpx_mock.get_requests()) == 1


async def test_streamed_provider_text_body_ends_the_run_with_the_error_text(
    httpx_mock: HTTPXMock,
) -> None:
    """The same body on a streamed run is NonStreamProviderBody after ONE request (BR-021).

    Before BR-021 the stream looked empty: the parser retry fired, and the
    run ended ``ParserError`` with the step-limit text after two requests.
    Release-dependent leg: the empty-looking stream is openai 2.43.0's SSE
    decoder dropping a body without SSE fields.
    """
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINT,
        content=_PROVIDER_TEXT.encode(),
        headers={"content-type": "text/plain"},
    )

    events = await _run(_loop(_real_client(), safety=SafetyConfig(), stream=True))

    assert [type(e) for e in events] == [ErrorEvent, FinalEvent]
    error, final = events
    assert isinstance(error, ErrorEvent) and isinstance(final, FinalEvent)
    assert error.error_type == "LLMError"
    assert error.context["type"] == "ContextLengthExceeded"
    assert error.context["classified_from"] == "NonStreamProviderBody"
    assert _PROVIDER_TEXT in error.message
    assert final.text == SafetyConfig().error_fallback_message
    assert "145048" not in final.text
    assert len(httpx_mock.get_requests()) == 1


async def test_llm_error_log_line_names_the_classified_type(httpx_mock: HTTPXMock) -> None:
    """agent_loop_completed carries llm_error_type, and no structlog entry carries the provider text (BR-021).

    Captures every structlog entry of the run (loop and adapter). It does
    not see ``stdlib`` logging.
    """
    httpx_mock.add_response(
        method="POST",
        url=_ENDPOINT,
        content=_PROVIDER_TEXT.encode(),
        headers={"content-type": "text/plain"},
    )

    with structlog.testing.capture_logs() as logs:
        await _run(_loop(_real_client(), safety=SafetyConfig()))

    completed = [entry for entry in logs if entry["event"] == "agent_loop_completed"]
    assert len(completed) == 1
    assert completed[0]["terminated_by"] == "llm_error"
    assert completed[0]["llm_error_type"] == "ContextLengthExceeded"
    assert logs, "capture_logs saw no entries; the leak check below would be vacuous"
    assert all("SENTINEL" not in repr(entry) for entry in logs)
    assert all("145048" not in repr(entry) for entry in logs)


@pytest.mark.parametrize(
    ("context", "expected"),
    [({"type": "BadRequestError"}, "BadRequestError"), ({"type": 42}, None), ({}, None)],
    ids=["str_type", "non_str_type", "no_type"],
)
async def test_llm_error_log_type_is_a_string_code_or_none(
    context: dict[str, Any], expected: str | None
) -> None:
    """llm_error_type is the context's str type, else None; never the message (BR-021, §10)."""
    llm = FakeLLMClient([LLMError("secret provider text", context=context)])

    with structlog.testing.capture_logs() as logs:
        await _run(_loop(llm, safety=SafetyConfig()))

    completed = next(entry for entry in logs if entry["event"] == "agent_loop_completed")
    assert completed["llm_error_type"] == expected
    assert all("secret provider text" not in repr(entry) for entry in logs)
