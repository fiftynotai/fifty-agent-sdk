# Changelog

All notable changes to `fifty-agent-sdk` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-22

First public open-source release as a standalone package (`fifty-agent-sdk`),
extracted with its full commit history from the monorepo it was first built in.

### Added
- Standard MCP client over Streamable HTTP via the official `mcp` SDK, exposed
  through `MCPProvider` (full `initialize → tools/list → tools/call` handshake).
  The MCP path is now standard-only.

### Changed
- Import root is now `fifty_agent_sdk` (was `agent_sdk`).
- Distributed and published as `fifty-agent-sdk` on PyPI.

## [1.0.0]

Initial production release: custom ReACT loop, JSON-mode tool calling, a
pluggable LLM client (any OpenAI-compatible endpoint), in-process + MCP tool
sources, pluggable conversation-state storage (memory / SQL / Redis), audit
sinks, observability hooks, and a full-fidelity event stream.
