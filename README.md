<p align="center">
  <img src=".github/banner.png" alt="fifty-agent-sdk — a reusable agent loop for python." width="100%">
</p>

# fifty-agent-sdk

[![PyPI](https://img.shields.io/pypi/v/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![CI](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

fifty-agent-sdk is a reusable agent loop for python. it implements a custom reACT loop with json-mode tool calls, an mcp client, and pluggable llm, state, and tool backends. it exists because the loop, the parser, the safety checks, and the runner kept getting rewritten per project. this is that loop, factored out once: write the tools, hand them to the runner, let it iterate.

## At a glance

- talks to any openai-compatible chat-completions endpoint by swapping one `base_url`: openai, google distributed cloud, a local oss server.
- llm clients, state stores, and tools are pluggable behind protocols: bring your own, the loop stays the same.
- the run emits a typed event stream the caller consumes, so you watch the react loop step by step.
- an iteration cap and per-tool timeouts bound every run, with a fallback answer on error or cap: a loop that can't end is a loop that doesn't ship.
- zero-infra by default: no db, no redis, until you opt into an extra.

## Installation

```
pip install fifty-agent-sdk
```

Optional extras:

- `pip install 'fifty-agent-sdk[sql]'` — enables SqlStateStore, SqlAuditSink, SQLAlchemy
- `pip install 'fifty-agent-sdk[redis]'` — enables RedisStateStore

Importing `fifty_agent_sdk` pulls neither extra; the extra symbols are re-exported lazily, and first access without the relevant extra installed raises a clear `ImportError`. The `sql` extra installs SQLAlchemy but not a database driver — bring your own async driver (e.g. `aiosqlite` for SQLite, `asyncpg` for PostgreSQL).

Requires Python >=3.11.

## Quickstart

the example builds a tool, hands it to the `AgentRunner`, and consumes the typed event stream the run emits.

```python
import asyncio
from typing import Any

from fifty_agent_sdk import (
    AgentLoop,
    AgentRunner,
    MemoryStateStore,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    ToolMode,
    tool,
)


@tool()
async def get_weather(city: str) -> dict[str, Any]:
    """Return the current weather for a city."""
    return {"city": city, "temp_c": 21}


async def main() -> None:
    # 1. An LLM client — points at any OpenAI-compatible endpoint.
    #    Pass base_url=... to target GDC or a local OSS server instead of OpenAI.
    llm = OpenAICompatibleClient(api_key="sk-...")

    # 2. A tool registry — register the decorated tool.
    registry = Registry()
    registry.register(get_weather)

    # 3. The ReACT loop — LLM + registry + prompts + safety + tool mode.
    #    `tool_mode` picks how the model calls tools. ToolMode.JSON supplies
    #    the JSON parser and teaches the model its envelope; switch to
    #    ToolMode.NATIVE for provider function calling (see "tool modes").
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        prompts=PromptSections(persona="You are helpful."),
        safety=SafetyConfig(),
        model="gpt-4o",
        tool_mode=ToolMode.JSON,
    )

    # 4. The runner — wraps the loop with conversation-state persistence.
    runner = AgentRunner(
        loop=loop,
        state=MemoryStateStore(),
        system_prompt="You are a helpful weather assistant.",
    )

    # 5. Drive a turn and consume the event stream.
    async for event in runner.run("session-1", "What's the weather in Paris?"):
        print(event)


asyncio.run(main())
```

## Core concepts

### tools

the registry of functions the agent can call. each tool is a side-effecting action exposed to the loop, so the model can do something in the world and not just talk about it.

### tool modes

how the model calls tools, set with one value: `AgentLoop(tool_mode=ToolMode.JSON | ToolMode.PROSE | ToolMode.NATIVE)`. the mode sets the parser, the output format, the role tool results go back in, how tools are declared, and the retry reminder, all together. switching protocol means changing one value.

| | `JSON` | `PROSE` | `NATIVE` |
|---|---|---|---|
| tools declared via `tools` param | no | no | yes, `tool_choice="auto"` |
| prompt tool block | rendered | rendered | suppressed |
| text parser | `JsonModeParser` | `ProseModeParser` | final-only: text without `tool_calls` is the answer |
| output format | `JSON_MODE_OUTPUT_FORMAT` | `PROSE_MODE_OUTPUT_FORMAT` | none (plain-text final) |
| tool-result role | `"assistant"` (`"user"` allowed) | same as JSON | always `"tool"`, paired by `tool_call_id` |
| `stream=True` | allowed | allowed | rejected at construction |

`NATIVE` follows the openai-compatible function-calling protocol. a response with `tool_calls` is a tool turn, and any other text is the final answer, word for word. a text-shaped tool call (a json `{"action": "tool", ...}` envelope, a prose `Action:` block) is never run, so a `role="tool"` message always follows an assistant turn that carries its id. an empty response gets one retry with a native reminder; if that fails too, the run ends with a `ParserError` event and the fallback answer. with an empty registry, `tools` is left out of the request.

a mode owns its knobs. you can still pass one when it fits the mode, such as a custom parser that wraps `JsonModeParser` under `JSON`, `tool_message_role="user"` for chat templates that need strict user/assistant alternation, or an output format like "answer in markdown" under `NATIVE`. a value that belongs to another mode raises `ValueError` when the loop is built. the sdk never silently picks one:

```python
AgentLoop(..., tool_mode=ToolMode.NATIVE, parser=JsonModeParser())
# ValueError: tool_mode=ToolMode.NATIVE conflicts with parser=JsonModeParser(): under NATIVE
# a tool call is only ever a structured tool_calls entry, and a text parser could dispatch a
# text tool call. Drop parser=; NATIVE supplies its own final-only text parser.
```

omit `tool_mode` and the loop sends the same request bodies and system prompt as 1.7.0 (same keys, values and JSON types). the one event-level change is `ThoughtEvent.text` on native tool turns, which can now be non-empty (see CHANGELOG). `parser=` is required, `tool_message_role` defaults to `"tool"`, and `SafetyConfig(native_tools_enabled=True)` declares tools natively without changing the parser. that flag still works and is not deprecated, but on its own it leaves a text tool call dispatchable. to migrate, drop `parser=`, `output_format=` and `native_tools_enabled`, and pass `tool_mode=ToolMode.NATIVE`. the final answer is then plain text in `FinalEvent.text`, not a json envelope.

### llm

the llm client. a protocol plus an openai-compatible adapter, so the loop talks to any chat-completions endpoint by changing one base_url. a `max_tokens` cap is sent as `max_completion_tokens` for gpt-5.x and o-series models, which reject `max_tokens`; `max_tokens_param=` overrides that choice per client.

for reasoning models, `AgentLoop(reasoning_effort="medium")` sets `reasoning_effort` on every request the loop builds, including retries, re-asks, native and streamed turns. it is sent for any model name, with no filtering, so a provider that rejects it fails the run loudly on the first call. the string `"none"` is a level and is sent; `None`, the default, leaves it out. `AgentLoop(temperature=...)` works the same way: omit it and the loop keeps sending `0.0`, pass a number to send that, or pass `None` to leave `temperature` out of every request. `None` means leave it out, not use the default. the loop never couples the two, so if your model rejects `temperature` alongside `reasoning_effort` (openai documents this for gpt-5.1; the sdk has not verified it), set `temperature=None` yourself:

```python
loop = AgentLoop(
    ...,
    model="gpt-5.1",
    reasoning_effort="medium",
    temperature=None,
)
```

to vary either value per request, build a `ChatRequest(..., reasoning_effort=..., temperature=...)` and call the client directly. a custom `LLMClient` has to forward `reasoning_effort` itself.

### state

the state stores. where conversation state persists between turns, with branching built in: fork a session, switch between branches, truncate back to an earlier point. `MemoryStateStore` needs no infrastructure, but it is process-local and non-durable: by default it lazily expires whole sessions when monotonic inactivity reaches 3,600 seconds and retains at most 1,000 sessions using LRU eviction. successful reads refresh inactivity. `SqlStateStore` and `RedisStateStore` are durable backends behind the extras.

a runner hands back the store it was built with as `runner.state`, so the branching calls above are reachable from a runner you already have:

```python
store = MemoryStateStore()
runner = AgentRunner(loop=..., state=store)

runner.state is store  # True — the exact instance, never a copy or a wrapper
branch = await runner.state.fork(session_id, from_sequence=4)
```

configure either in-memory bound independently when an ephemeral workload needs different limits:

```python
store = MemoryStateStore(ttl_seconds=900, max_sessions=250)
```

the former unbounded behavior remains available as an explicit opt-in with `MemoryStateStore(ttl_seconds=None, max_sessions=None)`. prefer a durable backend instead when conversation state must survive process restarts.

identity is the point rather than convenience: a second store constructed over the same engine carries its own lock registry, so two writers could interleave on one session. sharing `runner.state` shares the serialization too.

it is read-only, for correctness and not for style. `run()` appends the user message, drives the loop, then appends the assistant message — a store swapped in between those appends would split one turn across two backends. assignment raises `AttributeError`, and mypy rejects it statically. to use a different store, construct another runner; `__init__` does no i/o. the declared type is the `StateStore` protocol, so keep your own concretely-typed reference if you need backend-specific api like `SqlStateStore.aclose()`.

`runner.state` is the supported way in. `_state` is private, carries no semver protection, and may be renamed or removed in a patch release.

### streaming

a typed event stream the caller consumes while the loop runs. each step in the run surfaces as an event instead of waiting for a final blob.

### safety

the caps that bound a run: a max-iteration ceiling on react cycles and a per-tool timeout, plus the fallback answer returned when a run errors or hits the cap. a loop that can't end is a loop that doesn't ship.

### audit

the audit sinks and observability hooks. they record what the agent did, so a run can be read back after it finishes. hooks that change a run instead of watching it are `Interventions`, below.

### interventions

hooks whose return values the loop honours. `Hooks` watch a run; `Interventions` change it, at two points around each tool call. `before_tool` runs before the call and can deny it or replace its arguments. `after_tool` runs after the tool returns and can add a note to what the model reads next. they are wired on the loop only, and the runner takes none: a runner's `session_id` reaches both hooks, and it is `None` when you drive the loop directly.

```python
from fifty_agent_sdk import (
    AgentLoop,
    DenyToolCall,
    Interventions,
    ObservationEvent,
    ReplaceToolArgs,
)

card_calls: set[str] = set()  # call ids your ui already rendered as cards


def before_tool(session_id, call_id, tool_name, args):
    if tool_name == "delete_record":
        return DenyToolCall(reason="deleting records is disabled in this workspace")
    if args.get("limit", 0) > 50:
        return ReplaceToolArgs(args={**args, "limit": 50})
    return None  # run the call as the model asked


def after_tool(session_id, call_id, tool_name, args, result):
    if call_id in card_calls:
        return "The user can already see these records; answer without listing them."
    return None  # no note


loop = AgentLoop(
    ...,
    interventions=Interventions(before_tool=before_tool, after_tool=after_tool),
)

async for event in runner.run(session_id, message):
    if isinstance(event, ObservationEvent) and shows_as_card(event.result):
        render_card(event.result.output)
        card_calls.add(event.call_id)
```

`after_tool` runs once for every call that returned a `ToolResult`, a success or `is_error=True`, and never for an unknown tool, a timeout or a denied call. it runs after your code handled that call's event (inline in your `async for` body; under a runner, after `on_tool_end` too), so in the example the card set is already up to date. a non-blank string it returns is added after a blank line to the observation the model reads next, the same way in every tool mode; `None` or a blank string adds nothing. the sdk never writes to the `ToolResult`, and the events are the ones you would get without the hook, so your own renderer is unaffected. treat `result` as read-only. `args` is the hook's own deep copy of the dispatched arguments, taken after the tool returns, so nested values may carry in-place edits the tool made to its own arguments; changing the copy affects nothing else. the note inherits the observation's role, so keep it neutral and factual: in the `"user"` role it reads as the user's words, and in the `"assistant"` role as the model's own.

`before_tool` sees every call, including tool names the model made up. its `args` is a deep copy of the model's arguments, taken before dispatch, so editing it, even a nested value, changes nothing: return `None` to run the call as asked, `ReplaceToolArgs(args=...)` to run it with other arguments, or `DenyToolCall(reason=...)` to skip it. a denied call still emits `ActionEvent` and `ToolStartedEvent`, then a `ToolFailedEvent` with `"Tool call denied: <reason>"`, and the model reads that reason. like a note, the reason takes the observation's role, so in the `"user"` role it carries user authority. a replacement is what `ActionEvent.args`, `on_tool_start` and the audit payload show; it is never written into the model's own turn. with the shipped parsers and client the copy is complete however deeply the model nests its arguments: they decode them with `json.loads`, whose dicts and lists are copied without recursion, and strings, numbers, booleans and `None` are kept as they are. anything else is copied with `copy.deepcopy`, together with everything inside it. apart from running out of memory, only a value that is not plain json and cannot be copied makes that hook's copy one level deep instead, with a warning. only your own code can put one there: a custom parser or `LLMClient`, or, for `after_tool`'s copy, a `before_tool` replacement or any code that stores one into the arguments' nested values, such as a tool or an event consumer writing into `ActionEvent.args`. change arguments with `ReplaceToolArgs`, never by editing them in place, and alert on `intervention.args_not_copyable`.

when a hook fails:

- `after_tool` fails soft: if it raises or returns anything other than a string or `None`, the note is skipped and the run continues.
- `before_tool` follows `before_tool_fallback`. `BeforeToolFallback.DENY`, the default, fails closed: the call is denied. `BeforeToolFallback.ALLOW` fails open: the call runs with the model's original arguments. use `DENY` for guards and for hooks that scope arguments, because a scoping hook (say, one that adds the current tenant) that failed open would run the model's unscoped call. use `ALLOW` only for advisory hooks, where availability matters more than the check.
- a returned `DenyToolCall` is always honoured, whichever fallback you pick.
- every fallback logs a warning under `fifty_agent_sdk.interventions` with the call id and the exception or returned type, never the exception's text. under `ALLOW` that line, with `fallback="call_allowed"`, is the only sign a check was skipped, so alert on it.
- `asyncio.CancelledError` propagates from both hooks.

```python
Interventions(before_tool=normalise_dates, before_tool_fallback=BeforeToolFallback.ALLOW)
```

`after_tool` cannot replace or remove observation text, change `is_error` or `output`, or change an event. `before_tool` cannot rename a tool; deny instead. to screen mcp `isError` text, use `on_tool_error` (see mcp). a note or denial is part of that call's observation for the rest of the run and is never persisted: the next turn sees only the final answer. hooks may be sync or async and are awaited inline, one at a time even within a batch, so keep them fast: a call's own event reaches you before its `after_tool` runs, but later events wait for it. there is no hook that rewrites the llm request; wrap the `LLMClient` instead. without `interventions`, the loop sends the same request bodies (same keys, values and JSON types) and emits the same event stream as 1.9.0 (timestamps and minted ids aside).

### mcp

an mcp client over streamable http, adapted into the same registry the in-proc tools live in. a `tools/call` that comes back `isError=True` is a recoverable observation the model can reason about, not a dead run — and `on_tool_error` is the seam for screening that server-controlled text before the model reads it. an `after_tool` intervention can only add to that text, so screening stays here.

```python
def screen(message: str, content: list[dict]) -> str:
    # `message` is the sdk's bounded default; `content` is the server's raw
    # error blocks (read-only). return the string the model should see.
    if any("PII" in str(block) for block in content):  # your own predicate
        return "the upstream tool failed"
    return message


client = MCPClient(MCPClientConfig(base_url=...), auth=..., on_tool_error=screen)
provider = MCPProvider(client)
await provider.attach(registry)
```

the hook may be sync or async, and it only ever fires on a per-call `isError` result — never on success, never on a transport failure (that still raises `MCPError`). if it raises, returns a non-string, or returns a blank string, the sdk falls back to its own bounded message and logs a warning; it can never change `is_error` or `output`.

## Architecture

```
fifty_agent_sdk  —  module graph (from src/fifty_agent_sdk/, ground-truth imports)

src/fifty_agent_sdk/
├─ ▢ audit
├─ errors
├─ interventions
├─ ▢ llm
├─ loop
├─ ▢ mcp
├─ ▢ observability
├─ ▢ parser
├─ prompts
├─ ▶ runner
├─ safety
├─ ▢ state
├─ streaming
├─ tool_mode
└─ ▢ tools

depends (→):
   audit → errors
   interventions → observability
   interventions → tools
   llm → errors
   loop → errors
   loop → interventions
   loop → llm
   loop → observability
   loop → parser
   loop → prompts
   loop → safety
   loop → streaming
   loop → tool_mode
   loop → tools
   mcp → errors
   observability → llm
   parser → errors
   parser → llm
   runner → audit
   runner → errors
   runner → llm
   runner → loop
   runner → observability
   runner → state
   runner → streaming
   state → errors
   state → llm
   streaming → tools
   tool_mode → parser
   tool_mode → prompts
   tool_mode → safety
   tools → errors
   tools → llm
   tools → mcp

legend: ▶ entry   ▢ package   name module   → depends
```

## Highlights

- **branching** — first-class conversation branching on `StateStore`: `fork`, `list_branches`, `switch_branch`, branch-scoped `get_messages(..., branch_id=...)`, plus `BranchInfo` and `TRUNK_BRANCH_ID`. a session is now a tree of branches with an active head, and `append` writes to the active branch (the edit-a-message / regenerate model). implemented across memory, SQL, and Redis backends, data-additive and zero-migration: existing sessions read as the trunk branch. breaking for custom `StateStore` implementations: they must add the new methods.
- **`StateStore.truncate_after(session_id, sequence, *, branch_id=None)`** — a destructive hard-delete of a branch's tail (messages with sequence > N), for redaction, retention, and rollback. only the target branch's own messages are removed (a `fork`'s inherited prefix is never touched), and it is idempotent: a no-op on an unknown session or branch.

editing a turn is a consumer-side fork-then-append, and the original line stays reachable:

```python
# Edit a turn = fork the history before it, switch onto the new branch, then
# append the edited message. `store` is any StateStore; import `ChatMessage`
# from fifty_agent_sdk.
branch = await store.fork(session_id, from_sequence=4)  # keep messages 1..4
await store.switch_branch(session_id, branch)
await store.append(session_id, ChatMessage(role="user", content="...edited..."))
await store.get_messages(session_id, branch_id="trunk")  # original line intact
```

## Links

- [Homepage](https://github.com/fiftynotai/fifty-agent-sdk)
- [Repository](https://github.com/fiftynotai/fifty-agent-sdk)
- [Issues](https://github.com/fiftynotai/fifty-agent-sdk/issues)
- [Changelog](https://github.com/fiftynotai/fifty-agent-sdk/blob/main/CHANGELOG.md)

## License

MIT.
