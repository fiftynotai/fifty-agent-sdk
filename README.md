<p align="center">
  <img src=".github/banner.png" alt="fifty-agent-sdk — an embeddable ReACT loop. any endpoint. no infra." width="100%">
</p>

# fifty-agent-sdk

[![PyPI](https://img.shields.io/pypi/v/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/fifty-agent-sdk)](https://pypi.org/project/fifty-agent-sdk/)
[![CI](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fiftynotai/fifty-agent-sdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

an embeddable ReACT agent loop you drop into your own service. it talks to
any OpenAI-compatible Chat Completions endpoint — OpenAI, Google Distributed
Cloud, a local OSS server — by changing one `base_url`. transport-free,
infra-free by default, and pluggable behind protocols the whole way down.

## At a glance

- endpoint-agnostic — one `base_url` points the loop at OpenAI, GDC, or a local OSS server; the loop doesn't care which.
- pluggable tools — `@tool` derives a JSON Schema from type hints; MCP discovery adds remote tools to the same registry.
- pluggable state — conversation state lives behind a `StateStore` protocol: in-memory, SQL, or Redis.
- typed event stream — every ReACT step emits exactly one `AgentEvent`; every run ends with exactly one `FinalEvent`.
- safety caps — iteration ceiling, per-tool timeouts, and a fallback answer on error or cap.
- zero-infra default — memory backends ship in core; run an agent with no extra dependency.

## Installation

```
pip install fifty-agent-sdk
```

The SDK requires Python 3.11 or newer. The core install ships with the
in-memory backends, so an agent can run with no infrastructure dependency.

Two optional extras add durable backends:

- `pip install 'fifty-agent-sdk[sql]'` — pulls SQLAlchemy and enables
  `SqlStateStore` (durable conversation state) and `SqlAuditSink` (durable
  audit log).
- `pip install 'fifty-agent-sdk[redis]'` — pulls redis-py and enables
  `RedisStateStore` (Redis-backed conversation state).

Importing `fifty_agent_sdk` itself pulls neither SQLAlchemy nor redis-py. The
extra symbols are re-exported lazily; first access to one without the
relevant extra installed raises a clear `ImportError`.

## Quickstart

A complete agent fits in a handful of lines. The example below defines a
tool, wires the loop and runner, and drains the event stream:

```python
import asyncio
from typing import Any

from fifty_agent_sdk import (
    JSON_MODE_OUTPUT_FORMAT,
    AgentLoop,
    AgentRunner,
    JsonModeParser,
    MemoryStateStore,
    OpenAICompatibleClient,
    PromptSections,
    Registry,
    SafetyConfig,
    tool,
)


@tool()
async def get_weather(city: str) -> dict[str, Any]:
    """Return the current weather for a city."""
    return {"city": city, "temp_c": 21}


async def main() -> None:
    # 1. An LLM client — points at any OpenAI-compatible endpoint.
    llm = OpenAICompatibleClient(api_key="sk-...")

    # 2. A tool registry — register the decorated tool.
    registry = Registry()
    registry.register(get_weather)

    # 3. The ReACT loop — LLM + registry + parser + prompts + safety.
    #    `output_format` shows the model the JSON envelope the parser
    #    expects; without it JsonModeParser raises ParserError on every turn.
    loop = AgentLoop(
        llm=llm,
        registry=registry,
        parser=JsonModeParser(),
        prompts=PromptSections(persona="You are helpful."),
        safety=SafetyConfig(),
        model="gpt-4o",
        output_format=JSON_MODE_OUTPUT_FORMAT,
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

### Tools

The `@tool` decorator turns an async function into a `Tool`: it derives a
JSON Schema for the arguments from the function's type annotations and
docstring. A `Registry` is the dispatch table the loop talks to —
`Registry().register(my_tool)` adds a tool by name. `InProcProvider` is a
convenience helper for bulk-registering a batch of decorated callables.

### LLM clients

`LLMClient` is the protocol the loop depends on — anything implementing it
can drive the agent. `OpenAICompatibleClient` is the shipped implementation;
it works against any OpenAI-compatible Chat Completions endpoint. Point it at
GDC, a local OSS server, or OpenAI itself by passing `base_url` — the
provider differences are absorbed entirely by that one argument.

### State stores

`StateStore` is the protocol for conversation-state persistence across turns.
`MemoryStateStore` is the default in-memory implementation and needs no
infrastructure. `SqlStateStore` and `RedisStateStore` are durable backends
behind the `sql` and `redis` extras respectively.

### The event stream

Every step of the ReACT cycle emits exactly one event from the `AgentEvent`
union: `ThoughtEvent`, `ActionEvent`, `ToolStartedEvent`,
`ToolProgressEvent` (reserved — never emitted by the v1 loop),
`ObservationEvent`, `ToolFailedEvent`, `TokenEvent`, `FinalEvent`, and
`ErrorEvent`. Events carry a monotonic `sequence` counter and a `timestamp`
so consumers can detect drops or reorders. Every run ends with exactly one
`FinalEvent` — consumers can rely on it as the "iteration done" signal.

### Safety

`SafetyConfig` bounds a run: it caps the iteration count, sets per-tool
timeouts, and supplies the fallback answer used when a run terminates on an
error or safety cap.

### Audit & observability

Two optional collaborators plug into the runner. An `AuditSink` records an
`AuditEvent` at session start, each tool invocation, the final answer, and
any error. A `Hooks` instance fires lifecycle callbacks for logging, metrics,
and tracing. Both are best-effort and isolated — a raising sink or hook
never aborts a live run.

## Design principles

- **Pluggable everything.** LLM clients, tool sources, state stores, audit
  sinks, observability hooks all sit behind protocols.
- **Transport-free.** No HTTP, no WebSocket, no auth. Consumers wrap the
  `Runner` in whatever transport makes sense.
- **Production-first.** Iter caps, tool timeouts, structured errors,
  graceful fallbacks, full-fidelity event stream.
- **Standalone usable.** Memory backends ship by default so you can run an
  agent without any infrastructure dependency.

## Links

- package — https://pypi.org/project/fifty-agent-sdk/
- source & issues — https://github.com/fiftynotai/fifty-agent-sdk
- changelog — [CHANGELOG.md](CHANGELOG.md)
- contributing — issues and PRs welcome.

## License

[MIT](LICENSE) © fifty.dev
