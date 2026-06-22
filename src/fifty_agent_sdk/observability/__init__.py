"""Observability subpackage: vendor-neutral run-instrumentation hooks.

Re-exports the :class:`Hooks` dataclass — a container of optional callables
the SDK invokes at well-defined points of a run. ``Hooks`` is dependency-free
(it pulls no APM vendor and no optional extra), so it is an EAGER export with
no lazy ``__getattr__`` hook.

The :func:`fifty_agent_sdk.observability.hooks.invoke_hook` dispatch primitive is
kept internal — it is shared between :class:`fifty_agent_sdk.runner.AgentRunner`
and :class:`fifty_agent_sdk.loop.AgentLoop` but is not part of the consumer-facing
surface. See :mod:`fifty_agent_sdk.observability.hooks` for the full hook contract.
"""

from __future__ import annotations

from fifty_agent_sdk.observability.hooks import Hooks

__all__ = ["Hooks"]
