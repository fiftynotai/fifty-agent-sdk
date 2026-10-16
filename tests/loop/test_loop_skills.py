"""Loop tests for ``AgentLoop(skills=...)`` (FR-004 D2, D4, D9, D12; plan LS1-LS16).

The catalog goes into the system prompt's ``# Skills`` section, the load tool
is offered after the registry's tools in every tool mode, and calls to it are
routed to a private one-tool registry, so the caller's ``Registry`` is never
modified. With no skills, or an empty ``SkillSet``, nothing on the request
path moves: LS7 is the relative pin, and the four golden fixtures
(``test_legacy_golden.py``, ``test_golden_1_8_0.py``, ``test_golden_1_9_0.py``
and ``test_golden_1_11_0.py``) are the absolute one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    ActionEvent,
    AgentEvent,
    AgentLoop,
    AgentRunner,
    ChatMessage,
    DenyToolCall,
    FinalEvent,
    Interventions,
    JsonModeParser,
    MemoryStateStore,
    ObservationEvent,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    TokenEvent,
    ToolFailedEvent,
    ToolMode,
    ToolResult,
)
from fifty_agent_sdk.skills import Skill, SkillSet
from tests.loop.conftest import (
    FakeLLMClient,
    FakeTool,
    make_multi_tool_response,
    make_response,
    make_stream_chunks,
)
from tests.loop.golden_capture import wire_bodies
from tests.loop.golden_capture_1_9_0 import _normalise_scenario, serialise_events
from tests.runner.conftest import collect
from tests.runner.test_runner_audit import SpyAuditSink
from tests.skills.conftest import make_skill_dir

_PERSONA = "You are a document assistant."
_BODY = "# PDF forms\n\nUse the field ids from reference/forms.md."
_USER = [ChatMessage(role="user", content="Fill in the tax form.")]
_LOAD_DESCRIPTION = (
    "Load a skill's instructions by name, or a file inside that skill. "
    "The available skills are listed in the Skills section."
)


def _skills(*extra: Skill) -> SkillSet:
    return SkillSet(
        [
            Skill(name="pdf-forms", description="Fill in PDF forms.", body=_BODY),
            Skill(name="docx", description="Edit Word documents.", body="Docx body."),
            *extra,
        ]
    )


def _json_tool(name: str, args: dict[str, Any]) -> str:
    return json.dumps(
        {"thought": "t", "action": "tool", "tool_name": name, "tool_args": args, "answer": None}
    )


def _json_final(answer: str = "done") -> str:
    return json.dumps(
        {"thought": "t", "action": "final", "tool_name": None, "tool_args": None, "answer": answer}
    )


def _prose_tool(name: str, args: dict[str, Any]) -> str:
    return f"Thought: t\nAction: {name}\nAction Input: {json.dumps(args)}"


def _host_registry() -> Registry:
    registry = Registry()
    registry.register(FakeTool("search", result=ToolResult(output={"hits": 1})))
    return registry


def _loop(
    llm: FakeLLMClient,
    *,
    mode: ToolMode | None,
    skills: SkillSet | None,
    registry: Registry | None = None,
    **extra: Any,
) -> AgentLoop:
    """A loop under ``mode``; ``None`` is the legacy ``JsonModeParser`` path."""
    kwargs: dict[str, Any] = dict(extra)
    if mode is None:
        kwargs.setdefault("parser", JsonModeParser())
        kwargs.setdefault("output_format", JSON_MODE_OUTPUT_FORMAT)
    else:
        kwargs["tool_mode"] = mode
    return AgentLoop(
        llm=llm,
        registry=registry if registry is not None else _host_registry(),
        prompts=kwargs.pop("prompts", PromptSections(persona=_PERSONA)),
        safety=kwargs.pop("safety", SafetyConfig()),
        model="m",
        skills=skills,
        **kwargs,
    )


def _system(llm: FakeLLMClient, index: int = 0) -> str:
    content = llm.calls[index].messages[0].content
    assert isinstance(content, str)
    return content


# --- LS1: a load in every mode and role ---------------------------------------------------------

_MODE_ROWS: list[tuple[str, ToolMode | None, dict[str, Any], str, bool]] = [
    ("json_assistant", ToolMode.JSON, {}, "assistant", False),
    ("json_user", ToolMode.JSON, {"tool_message_role": "user"}, "user", False),
    ("prose", ToolMode.PROSE, {}, "assistant", False),
    ("native", ToolMode.NATIVE, {}, "tool", True),
    ("legacy_tool_role", None, {}, "tool", False),
    ("legacy_native_flag", None, {"safety": SafetyConfig(native_tools_enabled=True)}, "tool", True),
]


def _load_reply(key: str, native: bool, args: dict[str, Any]) -> Any:
    if native:
        return make_multi_tool_response([("load_skill", args)])
    if key == "prose":
        return make_response(_prose_tool("load_skill", args))
    return make_response(_json_tool("load_skill", args))


def _final_reply(key: str, mode: ToolMode | None) -> Any:
    if mode is ToolMode.NATIVE:
        return make_response("done")
    if key == "prose":
        return make_response("Thought: t\nFinal Answer: done")
    return make_response(_json_final())


@pytest.mark.parametrize(
    ("key", "mode", "extra", "role", "native"), _MODE_ROWS, ids=[row[0] for row in _MODE_ROWS]
)
async def test_load_skill_returns_the_body_in_every_mode_and_role(
    key: str, mode: ToolMode | None, extra: dict[str, Any], role: str, native: bool
) -> None:
    """LS1: the observation is the body; the next request carries it in the mode's tool message."""
    llm = FakeLLMClient([_load_reply(key, native, {"name": "pdf-forms"}), _final_reply(key, mode)])
    events = await collect(_loop(llm, mode=mode, skills=_skills(), **extra).run(list(_USER)))
    observations = [event for event in events if isinstance(event, ObservationEvent)]
    assert [(event.tool_name, event.result) for event in observations] == [
        ("load_skill", ToolResult(output=_BODY))
    ]
    reply = llm.calls[1].messages[-1]
    if role == "tool":
        assert (reply.role, reply.name, reply.content) == ("tool", "load_skill", _BODY)
        assert reply.tool_call_id == observations[0].call_id
    else:
        assert (reply.role, reply.content) == (role, f"Tool load_skill returned: {_BODY}")
        assert reply.tool_call_id is None and reply.name is None
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"


# --- LS2, LS3: how the tool and the catalog are declared --------------------------------------------

_DECLARATION = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": _LOAD_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": SkillSet([]).load_tool().schema.properties,
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}


async def test_native_declares_load_skill_last_and_keeps_tools_out_of_the_prompt() -> None:
    """LS2: NATIVE ``tools`` ends with the exact declaration; the prompt has ``# Skills``, no ``# Tools``."""
    llm = FakeLLMClient([make_response("done")])
    skills = _skills()
    await collect(_loop(llm, mode=ToolMode.NATIVE, skills=skills).run(list(_USER)))
    request = llm.calls[0]
    assert request.tools is not None
    assert [entry["function"]["name"] for entry in request.tools] == ["search", "load_skill"]
    assert request.tools[-1] == _DECLARATION
    system = _system(llm)
    assert f"# Skills\n{skills.render_catalog()}" in system
    assert "# Tools" not in system


@pytest.mark.parametrize(
    "mode", [ToolMode.JSON, ToolMode.PROSE, None], ids=["json", "prose", "legacy"]
)
async def test_text_modes_list_load_skill_last_in_the_tool_block(mode: ToolMode | None) -> None:
    """LS2: the ``# Tools`` block ends with the load tool; ``# Skills`` follows it; no ``tools`` param."""
    llm = FakeLLMClient([_final_reply("prose" if mode is ToolMode.PROSE else "json", mode)])
    skills = _skills()
    await collect(_loop(llm, mode=mode, skills=skills).run(list(_USER)))
    args = json.dumps(
        SkillSet([]).load_tool().schema.properties, sort_keys=True, ensure_ascii=False
    )
    load_line = f"- load_skill: {_LOAD_DESCRIPTION}\n  args: {args}"
    system = _system(llm)
    assert f"{load_line}\n\n# Skills\n{skills.render_catalog()}\n\n# Output Format\n" in system
    assert system.index("- search:") < system.index("- load_skill:")
    assert llm.calls[0].tools is None


async def test_native_with_no_host_tools_declares_only_load_skill() -> None:
    """LS3: an empty host registry plus skills declares one tool; without skills, no ``tools`` key."""
    with_skills = FakeLLMClient([make_response("done")])
    await collect(
        _loop(with_skills, mode=ToolMode.NATIVE, skills=_skills(), registry=Registry()).run(
            list(_USER)
        )
    )
    assert with_skills.calls[0].tools == [_DECLARATION]
    assert with_skills.calls[0].tool_choice == "auto"
    without = FakeLLMClient([make_response("done")])
    await collect(
        _loop(without, mode=ToolMode.NATIVE, skills=None, registry=Registry()).run(list(_USER))
    )
    body = (await wire_bodies(without, stream=False))[0]
    assert "tools" not in body and "tool_choice" not in body


# --- LS4, LS5: errors and reference files through the loop ---------------------------------------


async def test_unknown_skill_is_a_tool_failure_and_the_run_continues() -> None:
    """LS4: ``ToolFailedEvent`` with E0, the model sees it, and the run reaches its final."""
    llm = FakeLLMClient(
        [make_response(_json_tool("load_skill", {"name": "nope"})), make_response(_json_final())]
    )
    events = await collect(_loop(llm, mode=ToolMode.JSON, skills=_skills()).run(list(_USER)))
    error = "Unknown skill: 'nope'. Use a name listed in the Skills section."
    failures = [event for event in events if isinstance(event, ToolFailedEvent)]
    assert [(event.tool_name, event.error) for event in failures] == [("load_skill", error)]
    assert llm.calls[1].messages[-1].content == f"Tool load_skill failed: {error}"
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"


async def test_reference_file_read_and_refusal_through_the_loop(tmp_path: Path) -> None:
    """LS5: a contained file is returned; a traversal is a tool error with no host path."""
    directory = make_skill_dir(
        tmp_path, "pdf", body="See reference/forms.md.", files={"reference/forms.md": "Field ids."}
    )
    skills = SkillSet([Skill.from_directory(directory)])
    llm = FakeLLMClient(
        [
            make_multi_tool_response(
                [
                    ("load_skill", {"name": "pdf", "file": "reference/forms.md"}),
                    ("load_skill", {"name": "pdf", "file": "../../etc/passwd"}),
                ]
            ),
            make_response("done"),
        ]
    )
    events = await collect(_loop(llm, mode=ToolMode.NATIVE, skills=skills).run(list(_USER)))
    replies = [message for message in llm.calls[1].messages if message.role == "tool"]
    assert [reply.content for reply in replies] == [
        "Field ids.",
        "Tool error: File refused: '../../etc/passwd' is not a path inside skill 'pdf'.",
    ]
    assert all(str(tmp_path) not in str(reply.content) for reply in replies)
    assert isinstance(events[-1], FinalEvent)


# --- LS6: construction errors ------------------------------------------------------------------


def test_a_skills_value_that_is_not_a_skill_set_raises_type_error() -> None:
    """LS6: ``TypeError`` before anything else is built."""
    with pytest.raises(TypeError, match=r"^skills must be a SkillSet or None; got list$"):
        _loop(FakeLLMClient([]), mode=ToolMode.JSON, skills=[_BODY])  # type: ignore[arg-type]


def test_a_host_tool_named_like_the_load_tool_raises() -> None:
    """LS6 (D4b): the message names the tool and the ``tool_name`` fix."""
    registry = Registry()
    registry.register(FakeTool("load_skill"))
    with pytest.raises(ValueError) as caught:
        _loop(FakeLLMClient([]), mode=ToolMode.JSON, skills=_skills(), registry=registry)
    assert str(caught.value) == (
        "skills=SkillSet(...) adds a tool named 'load_skill', but the registry already has a "
        "tool with that name. Pass SkillSet(..., tool_name=...) with another name."
    )


def test_a_custom_tool_name_avoids_the_clash() -> None:
    """LS6 (D4b): with ``tool_name=`` the same registry is accepted."""
    registry = Registry()
    registry.register(FakeTool("load_skill"))
    loop = _loop(
        FakeLLMClient([]),
        mode=ToolMode.JSON,
        skills=SkillSet(_skills(), tool_name="read_skill"),
        registry=registry,
    )
    assert [tool.name for tool in loop._tool_snapshot()] == ["load_skill", "read_skill"]


def test_a_skill_named_like_a_host_tool_raises() -> None:
    """LS6 (D4c): a skill and a tool may not share a name."""
    with pytest.raises(ValueError) as caught:
        _loop(
            FakeLLMClient([]),
            mode=ToolMode.JSON,
            skills=_skills(Skill(name="search", description="d", body="b")),
        )
    assert str(caught.value) == (
        "skill names ['search'] are also names of registered tools; rename the skills or the "
        "tools, so the model cannot confuse a skill with a tool."
    )


def test_skills_together_with_prompts_skills_raises() -> None:
    """LS6 (D4d): the loop owns the slot when ``skills=`` is set."""
    with pytest.raises(ValueError) as caught:
        _loop(
            FakeLLMClient([]),
            mode=ToolMode.JSON,
            skills=_skills(),
            prompts=PromptSections(persona=_PERSONA, skills="hand-written catalog"),
        )
    assert str(caught.value).startswith("skills=SkillSet(...) conflicts with prompts.skills")


def test_an_empty_skill_set_skips_every_check() -> None:
    """LS6 (D9): an empty set is ``None``, so even a clashing registry and slot are accepted."""
    registry = Registry()
    registry.register(FakeTool("load_skill"))
    loop = _loop(
        FakeLLMClient([]),
        mode=ToolMode.JSON,
        skills=SkillSet([]),
        registry=registry,
        prompts=PromptSections(persona=_PERSONA, skills="hand-written catalog"),
    )
    assert "# Skills\nhand-written catalog" in loop._system_prompt


# --- LS7: an empty SkillSet equals None -----------------------------------------------------------


def _differential_script(mode: ToolMode | None) -> list[Any]:
    if mode is ToolMode.NATIVE:
        return [
            make_multi_tool_response([("search", {"q": "x"}), ("ghost", {})]),
            make_response("done"),
        ]
    if mode is ToolMode.PROSE:
        return [
            make_response(_prose_tool("search", {"q": "x"})),
            make_response("Thought: t\nFinal Answer: done"),
        ]
    return [make_response(_json_tool("search", {"q": "x"})), make_response(_json_final())]


@pytest.mark.parametrize(
    "mode",
    [None, ToolMode.JSON, ToolMode.PROSE, ToolMode.NATIVE],
    ids=["legacy", "json", "prose", "native"],
)
async def test_an_empty_skill_set_sends_what_no_skills_sends(mode: ToolMode | None) -> None:
    """LS7 (D9): bodies, events and the system prompt match ``skills=None`` exactly.

    A relative pin (both runs move together under a change to the shared
    path); the four golden fixtures are the absolute pin.
    """
    prompts = PromptSections(persona=_PERSONA, additional_context="Context.")
    runs: list[tuple[str, dict[str, list[dict[str, Any]]]]] = []
    for skills in (None, SkillSet([])):
        llm = FakeLLMClient(_differential_script(mode))
        loop = _loop(llm, mode=mode, skills=skills, prompts=prompts)
        events: list[AgentEvent] = await collect(loop.run(list(_USER)))
        bodies = await wire_bodies(llm, stream=False)
        runs.append((loop._system_prompt, _normalise_scenario(bodies, serialise_events(events))))
    assert runs[0] == runs[1]
    assert json.dumps(runs[0][1], sort_keys=True) == json.dumps(runs[1][1], sort_keys=True)


# --- LS8, LS9: the caller's registry ---------------------------------------------------------------


def test_the_callers_registry_is_untouched() -> None:
    """LS8 (D2): ``_registry`` is the caller's object, holding only the caller's tools."""
    registry = _host_registry()
    loop = _loop(FakeLLMClient([]), mode=ToolMode.NATIVE, skills=_skills(), registry=registry)
    assert loop._registry is registry
    assert [tool.name for tool in registry.list()] == ["search"]
    without = _loop(FakeLLMClient([]), mode=ToolMode.NATIVE, skills=None, registry=_host_registry())
    plain = AgentLoop(
        llm=FakeLLMClient([]),
        registry=_host_registry(),
        prompts=PromptSections(persona=_PERSONA),
        safety=SafetyConfig(),
        model="m",
        tool_mode=ToolMode.NATIVE,
    )
    assert without._system_prompt == plain._system_prompt
    assert (
        without._skill_registry is None and without._registry_for("load_skill") is without._registry
    )


async def test_a_late_registration_under_the_load_tools_name_is_shadowed() -> None:
    """LS9 (D4e): one declaration (the loop's), and calls go to the loop's tool."""
    registry = _host_registry()
    llm = FakeLLMClient(
        [make_multi_tool_response([("load_skill", {"name": "docx"})]), make_response("done")]
    )
    loop = _loop(llm, mode=ToolMode.NATIVE, skills=_skills(), registry=registry)
    impostor = FakeTool("load_skill", result=ToolResult(output="IMPOSTOR"))
    registry.register(impostor)
    events = await collect(loop.run(list(_USER)))
    tools = llm.calls[0].tools or []
    assert [entry["function"]["name"] for entry in tools] == ["search", "load_skill"]
    assert tools[-1] == _DECLARATION
    assert impostor.call_count == 0
    observation = next(event for event in events if isinstance(event, ObservationEvent))
    assert observation.result == ToolResult(output="Docx body.")


# --- LS10, LS11: batches and interventions ------------------------------------------------------------


async def test_a_native_batch_of_two_loads_gives_two_observations_in_order() -> None:
    """LS10: both calls run, and their replies follow call order."""
    llm = FakeLLMClient(
        [
            make_multi_tool_response(
                [("load_skill", {"name": "docx"}), ("load_skill", {"name": "pdf-forms"})]
            ),
            make_response("done"),
        ]
    )
    loop = _loop(
        llm,
        mode=ToolMode.NATIVE,
        skills=_skills(),
        safety=SafetyConfig(max_concurrent_tool_calls=2),
    )
    events = await collect(loop.run(list(_USER)))
    assert [event.result.output for event in events if isinstance(event, ObservationEvent)] == [
        "Docx body.",
        _BODY,
    ]
    assert [message.content for message in llm.calls[1].messages if message.role == "tool"] == [
        "Docx body.",
        _BODY,
    ]


async def test_interventions_see_load_skill_like_any_tool() -> None:
    """LS11: ``before_tool`` can deny a load; ``after_tool`` notes are appended to a load's body."""
    seen: list[tuple[str, dict[str, Any]]] = []

    def before_tool(
        _sid: str | None, _cid: str, name: str, args: dict[str, Any]
    ) -> DenyToolCall | None:
        seen.append((name, args))
        return DenyToolCall(reason="not this one") if args.get("name") == "docx" else None

    def after_tool(
        _sid: str | None, _cid: str, name: str, _args: dict[str, Any], _result: ToolResult
    ) -> str:
        return f"note for {name}"

    llm = FakeLLMClient(
        [
            make_multi_tool_response(
                [("load_skill", {"name": "docx"}), ("load_skill", {"name": "pdf-forms"})]
            ),
            make_response("done"),
        ]
    )
    loop = _loop(
        llm,
        mode=ToolMode.NATIVE,
        skills=_skills(),
        interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
    )
    await collect(loop.run(list(_USER)))
    assert seen == [("load_skill", {"name": "docx"}), ("load_skill", {"name": "pdf-forms"})]
    assert [message.content for message in llm.calls[1].messages if message.role == "tool"] == [
        "Tool call denied: not this one",
        f"{_BODY}\n\nnote for load_skill",
    ]


# --- LS12, LS13, LS14 ------------------------------------------------------------------------------


async def test_a_non_ascii_description_appears_literally_in_the_prompt_and_the_body() -> None:
    """LS12: Arabic text reaches the system prompt and the wire body as written."""
    description = "املأ النماذج الضريبية"
    llm = FakeLLMClient([make_response(_json_final())])
    skills = SkillSet([Skill(name="tax", description=description, body="b")])
    await collect(_loop(llm, mode=ToolMode.JSON, skills=skills).run(list(_USER)))
    assert f"- tax: {description}" in _system(llm)
    assert (
        f"- tax: {description}"
        in (await wire_bodies(llm, stream=False))[0]["messages"][0]["content"]
    )


async def test_a_streamed_json_run_loads_a_skill() -> None:
    """LS13: ``stream=True`` under JSON: the load happens, then the final answer streams."""
    final = _json_final("streamed")
    llm = FakeLLMClient(
        [
            make_stream_chunks([_json_tool("load_skill", {"name": "docx"})]),
            make_stream_chunks([final[:20], final[20:]]),
        ]
    )
    events = await collect(
        _loop(llm, mode=ToolMode.JSON, skills=_skills(), stream=True).run(list(_USER))
    )
    assert any(isinstance(event, TokenEvent) for event in events)
    assert llm.calls[1].messages[-1].content == "Tool load_skill returned: Docx body."
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "streamed"


async def test_a_skill_load_satisfies_require_tool_before_final() -> None:
    """LS14 (D12): a ``load_skill`` call counts as a tool call, so the final is accepted at once."""
    llm = FakeLLMClient(
        [make_response(_json_tool("load_skill", {"name": "docx"})), make_response(_json_final())]
    )
    loop = _loop(
        llm,
        mode=ToolMode.JSON,
        skills=_skills(),
        safety=SafetyConfig(require_tool_before_final=True),
    )
    events = await collect(loop.run(list(_USER)))
    assert len(llm.calls) == 2
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"
    assert sum(isinstance(event, ActionEvent) for event in events) == 1


# --- LS15: the real client --------------------------------------------------------------------------

_ENDPOINT = "https://example.com/v1/chat/completions"


def _completion(content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> bytes:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


async def test_the_real_client_declares_load_skill_and_sends_the_body_back(
    httpx_mock: HTTPXMock,
) -> None:
    """LS15: over HTTP, ``tools`` includes ``load_skill`` and the next body has the body as the tool reply."""
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "load_skill", "arguments": '{"name": "pdf-forms"}'},
    }
    for body in (_completion(None, [call]), _completion("done")):
        httpx_mock.add_response(
            method="POST", url=_ENDPOINT, content=body, headers={"content-type": "application/json"}
        )
    client = OpenAICompatibleClient(
        api_key="k", base_url="https://example.com/v1", timeout=5.0, max_retries=0
    )
    try:
        loop = AgentLoop(
            llm=client,
            registry=_host_registry(),
            prompts=PromptSections(persona=_PERSONA),
            safety=SafetyConfig(),
            model="m",
            tool_mode=ToolMode.NATIVE,
            skills=_skills(),
        )
        events = await collect(loop.run(list(_USER)))
    finally:
        await client.aclose()
    first, second = (json.loads(request.content) for request in httpx_mock.get_requests())
    assert [tool["function"]["name"] for tool in first["tools"]] == ["search", "load_skill"]
    assert "# Skills\n" in first["messages"][0]["content"]
    replayed_id = second["messages"][-2]["tool_calls"][0]["id"]
    assert second["messages"][-1] == {
        "role": "tool",
        "tool_call_id": replayed_id,
        "name": "load_skill",
        "content": _BODY,
    }
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "done"


# --- LS16: the Runner's audit -----------------------------------------------------------------------


async def test_the_runner_audit_of_a_skill_load_carries_argument_metadata_only(
    tmp_path: Path,
) -> None:
    """LS16: ``tool_invocation.args`` has types and lengths, never the skill name or the file path."""
    directory = make_skill_dir(tmp_path, "pdf", files={"secret-dir/forms.md": "Field ids."})
    llm = FakeLLMClient(
        [
            make_response(_json_tool("load_skill", {"name": "pdf", "file": "secret-dir/forms.md"})),
            make_response(_json_final()),
        ]
    )
    loop = _loop(llm, mode=ToolMode.JSON, skills=SkillSet([Skill.from_directory(directory)]))
    spy = SpyAuditSink()
    runner = AgentRunner(loop=loop, state=MemoryStateStore(), audit=spy)
    await collect(runner.run("s1", "fill the form"))
    invocation = next(event for event in spy.events if event.event_type == "tool_invocation")
    assert invocation.payload["tool_name"] == "load_skill"
    assert invocation.payload["args"] == {
        "file": {"type": "str", "len": len("secret-dir/forms.md")},
        "name": {"type": "str", "len": 3},
    }
    assert "secret-dir" not in json.dumps(invocation.payload["args"])


# --- The README's manual wiring ------------------------------------------------------------------


@pytest.mark.parametrize("mode", [ToolMode.JSON, ToolMode.NATIVE], ids=["json", "native"])
async def test_manual_wiring_sends_what_skills_kwarg_sends(mode: ToolMode) -> None:
    """README: ``registry.register(load_tool())`` plus ``PromptSections(skills=...)`` equals ``skills=``."""
    script = (
        [make_multi_tool_response([("load_skill", {"name": "docx"})]), make_response("done")]
        if mode is ToolMode.NATIVE
        else [
            make_response(_json_tool("load_skill", {"name": "docx"})),
            make_response(_json_final()),
        ]
    )
    skills = _skills()
    by_kwarg = FakeLLMClient(list(script))
    kwarg_events = await collect(_loop(by_kwarg, mode=mode, skills=skills).run(list(_USER)))
    registry = _host_registry()
    registry.register(skills.load_tool())
    by_hand = FakeLLMClient(list(script))
    hand_loop = _loop(
        by_hand,
        mode=mode,
        skills=None,
        registry=registry,
        prompts=PromptSections(persona=_PERSONA, skills=skills.render_catalog()),
    )
    hand_events = await collect(hand_loop.run(list(_USER)))
    assert _normalise_scenario(
        await wire_bodies(by_kwarg, stream=False), serialise_events(kwarg_events)
    ) == _normalise_scenario(
        await wire_bodies(by_hand, stream=False), serialise_events(hand_events)
    )


def test_the_skills_type_error_escapes_an_odd_class_name() -> None:
    """Aegis L2 sweep: ``TypeError`` text built from a host class name is escaped."""
    odd = type("Odd\x1b[2J", (), {})
    with pytest.raises(TypeError) as caught:
        _loop(FakeLLMClient([]), mode=ToolMode.JSON, skills=odd())  # type: ignore[arg-type]
    assert str(caught.value) == "skills must be a SkillSet or None; got Odd\\u001b[2J"
