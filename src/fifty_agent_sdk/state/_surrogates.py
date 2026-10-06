"""Surrogate code points in the message text the durable state stores write (BR-025).

UTF-8 cannot encode a surrogate code point (U+D800-U+DFFF), but a Python
``str`` can hold one, for example a final answer decoded from a provider
body whose JSON holds a ``\\ud800`` escape. Before BR-025,
``SqlStateStore.append`` (SQLite through aiosqlite) raised
``UnicodeEncodeError`` and ``RedisStateStore.append`` (fakeredis) raised
pydantic's ``PydanticSerializationError`` for such a message, outside their
``StateStoreError`` wrapping; see the "Text UTF-8 cannot encode (BR-025)"
section of :mod:`fifty_agent_sdk.state.sql` and of
:mod:`fifty_agent_sdk.state.redis` for what was measured.

:func:`escape_message_surrogates` is called once in each of those two
methods, before the message is written. It runs
:func:`fifty_agent_sdk._model_json.escape_surrogates` on each of
``content``, ``name`` and ``tool_call_id`` that is a ``str``. Each surrogate
code point in them becomes its six-character ``\\udXXX`` escape, the
spelling ``json.dumps`` uses. Any other value (``None``, or what a message
built with ``model_construct``, copied with ``model_copy(update=...)`` or
changed by assignment after validation holds) is passed through as it is:
``escape_surrogates`` would raise ``AttributeError`` on it, where before
BR-025 the stores wrote such a value or raised their own error for it.

- When no field changes, the function returns ``message`` itself, so a
  message without a surrogate code point in those fields is written by the
  same code, from the same object, as before (measured: BR-025 evidence,
  probe P4 and round 2).
- Otherwise it returns ``message.model_copy(update=...)`` with the three
  values, without re-validation. It never changes ``message``: the Runner
  keeps using the message it appended (``runner.py:752`` adds its user
  message to the history the loop sends), so a change in place would alter
  the request.
- ``role`` and ``tool_calls`` are left as they are, and a copy shares
  ``tool_calls`` with ``message``. A surrogate code point in a tool call is
  not escaped, so each store handles it as before BR-025 (decision D3; see
  the stores' module docstrings): ``SqlStateStore`` on SQLite stores it, and
  ``RedisStateStore`` raises ``PydanticSerializationError`` for one in a
  tool call's name, id or argument value.

The escape cannot be reversed. Reading the message back returns the six
characters, nothing decodes them, and they cannot be told apart from the
same six characters typed.

Why the two durable stores, and not the Runner or ``MemoryStateStore``: an
escape in the Runner would leave a direct ``append`` caller raising as
before, and would change what ``MemoryStateStore`` keeps. That store keeps
the message object itself and never encodes it, so it keeps the code point
and returns it.

This module is private and carries no semver protection.
"""

from __future__ import annotations

from fifty_agent_sdk._model_json import escape_surrogates
from fifty_agent_sdk.llm.types import ChatMessage


def _escape_text(value: object) -> object:
    """Run ``escape_surrogates`` on a ``str``; return any other value as it is.

    A validated :class:`ChatMessage` holds a ``str`` (or ``None`` for
    ``name`` and ``tool_call_id``) in these fields. One built with
    ``model_construct``, copied with ``model_copy(update=...)``, or changed
    by assignment after validation, can hold any value, and the stores wrote
    such a value before BR-025 (or raised their own error for it), so it is
    passed through unchanged.
    """
    return escape_surrogates(value) if isinstance(value, str) else value


def escape_message_surrogates(message: ChatMessage) -> ChatMessage:
    """Return ``message`` with each surrogate code point in its text fields written as ``\\udXXX`` (BR-025).

    The escape cannot be reversed: the six characters cannot be told apart
    from the same six characters typed (see the module docstring).

    Args:
        message: The message a durable store is about to write. It is never
            changed.

    Returns:
        ``message`` itself when ``content``, ``name`` and ``tool_call_id``
        hold no surrogate code point, else a copy whose ``str`` fields carry
        the escape. A value that is not a ``str``, ``role`` and
        ``tool_calls`` are not changed.
    """
    content = _escape_text(message.content)
    name = _escape_text(message.name)
    tool_call_id = _escape_text(message.tool_call_id)
    if content is message.content and name is message.name and tool_call_id is message.tool_call_id:
        return message
    return message.model_copy(
        update={"content": content, "name": name, "tool_call_id": tool_call_id}
    )


__all__ = ["escape_message_surrogates"]
