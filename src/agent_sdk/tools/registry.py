"""The :class:`Registry` — the loop's only view into the tool universe.

The Registry knows nothing about how a :class:`Tool` came to exist —
in-proc, MCP, RPC, dynamic stubs are all the same to it. Its responsibilities
are exactly three:

1. Name lookup (:meth:`Registry.get`).
2. Timeout enforcement (:meth:`Registry.invoke` wraps the call in
   :func:`asyncio.wait_for`).
3. Exception classification: not-found and timeout escape as exceptions;
   ordinary tool exceptions become :class:`ToolResult` with ``is_error=True``;
   :class:`BaseException` and :class:`AgentSdkError` propagate untouched.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from agent_sdk.errors import AgentSdkError, ToolNotFound, ToolTimeout
from agent_sdk.tools.protocol import Tool, ToolResult

_log = structlog.get_logger(__name__)


class Registry:
    """In-memory dispatch table from tool name to :class:`Tool`.

    The registry is intentionally narrow. It does not own tool lifecycles,
    does not enforce schemas at invocation time (the tool implementation
    does), and does not introspect arguments. Its job is to be the boundary
    the ReACT loop talks to so that no other layer needs to think about how
    a tool was constructed.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a :class:`Tool` instance.

        Subsequent registrations with the same ``tool.name`` overwrite the
        previous entry (last write wins). This is deliberate: tests, fixtures,
        and providers commonly re-register tools when reconfiguring. A
        ``structlog`` warning is emitted on overwrite so unintended
        collisions in production are still visible.

        Args:
            tool: An object satisfying the :class:`Tool` protocol.

        Raises:
            TypeError: If ``tool`` does not satisfy the :class:`Tool`
                protocol at runtime. The ``runtime_checkable`` Protocol's
                ``isinstance`` check verifies that ``name``, ``description``,
                ``schema``, and an ``invoke`` attribute are all present.
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"register() requires a Tool; got {type(tool).__name__}"
            )
        if tool.name in self._tools:
            _log.warning("tool overwritten", name=tool.name)
        self._tools[tool.name] = tool

    def list(self) -> list[Tool]:
        """Return a snapshot of currently registered tools.

        The returned list is freshly constructed on each call; mutating it
        does not affect the registry. Order is insertion order (Python dicts
        preserve insertion order since 3.7).
        """
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        """Look up a tool by name.

        Args:
            name: The tool name to resolve.

        Returns:
            The registered :class:`Tool`.

        Raises:
            ToolNotFound: When ``name`` is not registered. ``context``
                contains ``name`` and ``available`` (the list of currently
                registered names) so callers can produce useful error
                messages.
        """
        try:
            return self._tools[name]
        except KeyError as e:
            raise ToolNotFound(
                f"Tool '{name}' is not registered",
                context={"name": name, "available": list(self._tools)},
            ) from e

    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        timeout: float | None,
    ) -> ToolResult:
        """Dispatch to a tool with timeout enforcement and exception classification.

        Failure handling is split deliberately into three classes:

        - **Unknown name** -> :class:`ToolNotFound` raises. This is a caller
          bug (the LLM hallucinated a tool, the registry was wired wrong); it
          should never be swallowed.
        - **Timeout** -> :class:`ToolTimeout` raises. The loop in BR-006 will
          catch this and decide whether to retry or abort.
        - **Tool raises Exception** -> a :class:`ToolResult` with
          ``is_error=True`` is returned. The LLM sees the failure as data and
          can choose to retry with different args or pick a different tool.
        - **SDK errors, ``CancelledError``, ``KeyboardInterrupt``,
          ``SystemExit``** -> propagate untouched. SDK errors are typed
          contracts and shouldn't be silently downgraded; the others are
          process-fatal / cancellation signals that must not be swallowed.

        Args:
            name: Name of the tool to invoke.
            args: Argument dict forwarded to ``tool.invoke``.
            timeout: Maximum number of seconds to wait. Pass ``None`` to
                disable timeout enforcement.

        Returns:
            The :class:`ToolResult` produced by the tool, or a synthesized
            :class:`ToolResult` with ``is_error=True`` if the tool body
            raised a plain :class:`Exception`.

        Raises:
            ToolNotFound: If ``name`` is not registered.
            ToolTimeout: If the tool's invocation exceeds ``timeout`` seconds.
            agent_sdk.errors.AgentSdkError: If the tool raises any SDK error.
            asyncio.CancelledError: If the surrounding task is cancelled.
            KeyboardInterrupt, SystemExit: Never swallowed.
        """
        tool = self.get(name)  # raises ToolNotFound
        coro = tool.invoke(args)
        try:
            if timeout is None:
                return await coro
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError as e:
            # asyncio.TimeoutError is aliased to builtins.TimeoutError in
            # Python 3.11+; catch the canonical name.
            raise ToolTimeout(
                f"Tool '{name}' exceeded timeout of {timeout}s",
                context={"name": name, "timeout": timeout},
            ) from e
        except (AgentSdkError, asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # SDK errors are typed contracts and shouldn't be downgraded.
            # CancelledError / KeyboardInterrupt / SystemExit must not be
            # swallowed (cancellation and process-fatal signals).
            raise
        except Exception as e:
            return ToolResult(
                output=None,
                is_error=True,
                error=f"{type(e).__name__}: {e}",
            )


__all__ = ["Registry"]
