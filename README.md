# agent-sdk

Production-grade reusable agent loop SDK. Provides a custom ReACT loop that
runs against any OpenAI-compatible Chat Completions endpoint (OpenAI itself,
Google Distributed Cloud, local OSS servers, etc.), with pluggable tool
sources (in-process registration + MCP discovery), pluggable conversation
state storage, and a full-fidelity event stream.

## Status

**Skeleton.** Implementation lands brief-by-brief.

## Planned shape

```
agent_sdk/
├── loop.py             # ReACT iterator
├── runner.py           # Drives the loop with state load/save + event emit
├── parser/             # json_mode (default), prose_mode, native_tools (reserved)
├── prompts.py          # Templated system prompts
├── safety.py           # Iter cap, tool timeout, fallback wrapping
├── streaming.py        # Typed event stream protocol
├── llm/                # LLMClient protocol + OpenAICompatibleClient
├── tools/              # Tool protocol, Registry, MCPProvider, InProcProvider
├── mcp/                # Generic MCP client (talks to any FastMCP server)
├── state/              # StateStore protocol + Memory/SQL/Redis backends
├── audit/              # AuditSink protocol + console/SQL backends
├── observability/      # Hooks for logging/metrics/tracing
└── errors.py           # Typed exceptions
```

## Design principles

- **Pluggable everything.** LLM clients, tool sources, state stores, audit
  sinks, observability hooks all sit behind protocols.
- **Transport-free.** No HTTP, no WebSocket, no auth. Consumers wrap the
  `Runner` in whatever transport makes sense.
- **Production-first.** Iter caps, tool timeouts, structured errors,
  graceful fallbacks, full-fidelity event stream.
- **Standalone usable.** Memory backends ship by default so you can run an
  agent without any infrastructure dependency.

## License

MIT
