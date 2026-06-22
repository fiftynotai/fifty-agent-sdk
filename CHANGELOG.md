# Changelog

All notable changes to `fifty-agent-sdk` are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `Registry.invoke` now enforces timeouts via `asyncio.timeout` instead of
  `asyncio.wait_for`, running the tool coroutine inline in the caller's task.
  This makes `KeyboardInterrupt`/`SystemExit` propagation deterministic across
  Python 3.11–3.13 and fixes a pytest-session abort on the 3.11 CI leg. (BR-002)

### Changed
- `fifty_agent_sdk.__version__` is now derived from installed distribution
  metadata (`importlib.metadata`) rather than a hardcoded string, so it can no
  longer drift from `pyproject.toml`. (TD-001)

## [1.1.1] - 2026-06-22

### Fixed
- `fifty_agent_sdk.__version__` now reports the correct release version. It was
  pinned at `1.0.0` and missed the 1.1.0 bump; all version sources
  (`pyproject.toml` and `__init__.py`) are now in agreement.

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
