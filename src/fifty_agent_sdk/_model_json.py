"""Text the SDK writes for a model to read, to a log line and to two state stores: JSON (BR-020), surrogate escapes (BR-022, BR-024, BR-025) and control-character escapes for log values (BR-028).

Routing rule: every ``json.dumps`` whose output reaches a model prompt goes
through :func:`dumps_for_model`. Since 1.11.0 those are the three call sites
that render tool results, the text-mode tool list and the ``arguments`` string
of a replayed native tool call. JSON that is stored, hashed or used as a key
(for example the Redis branch metadata in ``state/redis.py``) never goes
through it and keeps the stdlib defaults, so stored bytes do not change.

:func:`escape_surrogates` writes each surrogate code point as its
``\\udXXX`` escape. The loop runs it once on the text of every tool-result
message it builds (``AgentLoop._build_tool_message``, BR-022), and on the
model's tool name in its ``tool_invoked`` debug log line (BR-024; since
BR-028 through :func:`escape_for_log`). The shipped client runs it when it
serialises a request, on the three fields
that carry text the model wrote: an assistant message's content, each tool
call's name and a ``"tool"`` message's name
(``OpenAICompatibleClient._serialize_message``, BR-024). The BR-024 escapes
exist only in the request and the log line, so the loop's messages and
events, and the messages the Runner passes to its state store, keep the
model's text. The same rule holds for it as for :func:`dumps_for_model`:
text that is stored, hashed or used as a key never goes through it, with
one exception. ``SqlStateStore.append`` and ``RedisStateStore.append`` run
it on a message's ``content``, ``name`` and ``tool_call_id``
(``state/_surrogates.py``, BR-025), because before BR-025 those stores
could not store those fields holding a surrogate code point at all
(measured with SQLite through aiosqlite and with fakeredis; Postgres not
measured): SQLite's driver and pydantic's ``model_dump_json()`` raised.
Text without one is returned as the same object, so it is stored as before.
There too the escape cannot be reversed: reading the message back returns
the six characters, which cannot be told apart from the same six characters
typed.

:func:`escape_for_log` is for a log value that a model or a remote MCP
server can control (BR-028). It runs :func:`escape_surrogates`, then writes
each control character, U+0000-U+001F and U+007F-U+009F, as six characters:
a backslash, ``u`` and four lowercase hex digits. Eight logging calls use
it: the loop's two ``tool_invoked`` lines, ``Registry.register``'s
``tool overwritten`` line (which escapes a host's own ``str`` tool name the
same way), ``MCPProvider``'s ``mcp.tool_overwrite`` and
``mcp.refresh_failed`` (its ``error_message``), and ``MCPClient``'s three
``mcp.tool_error_hook_*`` calls. That escape cannot be told apart from the
same six characters typed either. It never runs on text sent to a model, on
events or on stored text: a control character there (a newline in an
assistant message, say) is content.

This module is private and carries no semver protection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final


def dumps_for_model(
    obj: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
) -> str:
    """Serialise ``obj`` to JSON text for a model prompt, keeping non-ASCII literal.

    The text is ``json.dumps(obj, ensure_ascii=False, default=default,
    sort_keys=sort_keys)``. By default ``json.dumps`` writes each non-ASCII
    character as a ``\\uXXXX`` escape: 6 characters, or 12 for a character
    outside the Basic Multilingual Plane, which is written as a surrogate
    pair. In the case BR-020 measured, one Arabic-heavy tool result was
    141,325 characters escaped and 35,840 unescaped, and the provider
    rejected the request that added the escaped form for exceeding its
    max_prompt_length of 131,072.

    For data whose strings contain only code points U+0000-U+007E the result
    equals ``json.dumps(obj, default=default, sort_keys=sort_keys)`` byte for
    byte. U+007F (DEL) is the one ASCII exception: the default form writes it
    as ``\\u007f``, this function writes the character itself (still valid
    JSON and valid UTF-8).

    Fallback: when the text cannot be encoded as UTF-8, which happens only
    when a string holds a surrogate code point (U+D800-U+DFFF; a Python
    string can hold one, for example from ``json.loads('"\\ud800"')``), the
    function returns the escaped default form ``json.dumps(obj,
    default=default, sort_keys=sort_keys)`` instead. That form is ASCII and
    always encodes. The fallback is all-or-nothing: a payload with any
    surrogate code point is escaped in full, exactly as before BR-020. The
    reason: the ``openai`` release in the dev environment (2.43.0) encodes the
    request body as strict UTF-8, and a probe showed that a surrogate code
    point written literally into ``arguments`` makes ``complete()`` raise
    ``UnicodeEncodeError`` before any request is sent.

    Only ``UnicodeEncodeError`` from that encode check is caught. Every
    exception from ``json.dumps`` itself (``TypeError``, ``ValueError``,
    ``RecursionError``) propagates unchanged, so each caller keeps its own
    fallback.

    Never use this for JSON that is stored, hashed or used as a key: changing
    its encoding there would change stored bytes.

    Args:
        obj: The value to serialise.
        default: Passed to ``json.dumps`` unchanged (called for values that
            are not JSON-serialisable).
        sort_keys: Passed to ``json.dumps`` unchanged.

    Returns:
        JSON text with non-ASCII characters written literally, or the escaped
        default form when the literal text holds a surrogate code point.
    """
    text = json.dumps(obj, ensure_ascii=False, default=default, sort_keys=sort_keys)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # BR-020: a surrogate code point cannot be sent as UTF-8; return the
        # pre-1.11.0 escaped form, which is ASCII, so this text encodes.
        return json.dumps(obj, default=default, sort_keys=sort_keys)
    return text


def escape_surrogates(text: str) -> str:
    """Write each surrogate code point in model-facing, logged or stored message text as its ``\\udXXX`` escape (BR-022, BR-024, BR-025).

    UTF-8 can encode every code point except the surrogates, U+D800-U+DFFF.
    A Python ``str`` can still hold them (``json.loads('"\\ud800"')`` or a
    ``surrogateescape`` decode makes one), and the ``openai`` releases BR-022
    measured (2.43.0 and 2.54.0) encode the request body as strict UTF-8, so
    one surrogate code point in a tool-result message made the next request
    raise ``UnicodeEncodeError`` before it was sent. BR-024 measured the same
    for text the model wrote and the SDK sent back (an echoed completion, a
    tool name), and the shipped client's request serialiser calls this
    function too, as does the loop's ``tool_invoked`` log line, through
    :func:`escape_for_log` (see the module docstring). Two state stores
    raised for such text too, and since BR-025 they call it as well; whoever
    reads that text back cannot tell the escape from the same six characters
    typed either (below). It leaves control characters as they are;
    :func:`escape_for_log` escapes those, for log values only.

    The function tries ``text.encode("utf-8")``, which fails exactly when the
    text holds a surrogate code point. If it succeeds, ``text`` itself is
    returned (the same object): every other code point, non-ASCII text and
    U+007F included, is left as it is. If it fails, each surrogate code point
    becomes the six characters ``\\udXXX`` (lowercase hex), the spelling
    ``json.dumps`` uses for it, and nothing else changes. Two surrogate code
    points in a row (``chr(0xD83D) + chr(0xDE00)``) are escaped one by one,
    not combined into the character they would pair to.

    The escape is not reversible: the model cannot tell it from those six
    characters typed literally, and the function leaves such typed text as
    it is. Unlike :func:`dumps_for_model`'s all-or-nothing fallback, only the
    surrogate code points change. That fallback can escape every non-ASCII
    character because a JSON reader decodes the escapes back. Plain text has
    no decoder, and escaping every non-ASCII character would bring back the
    length BR-020 removed (six characters per Arabic letter).

    Only ``UnicodeEncodeError`` from the encode check is caught. Never use
    this on text that is stored, hashed or used as a key, except where a
    store cannot hold a surrogate code point in that field at all: BR-025
    runs it on the ``content``, ``name`` and ``tool_call_id`` that
    ``SqlStateStore`` and ``RedisStateStore`` write (see the module
    docstring). There it changes only text those stores could not store in
    those fields before, and whoever reads it back cannot tell the escape
    from the same six characters typed.

    Args:
        text: The text of a message, or of a message field such as a tool
            name, that the SDK is about to send to a model, write to a log
            line or, in ``SqlStateStore`` and ``RedisStateStore``, store.

    Returns:
        ``text`` itself when it encodes as UTF-8, else a copy in which every
        surrogate code point is written as ``\\udXXX``.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        # BR-022: "backslashreplace" writes each code point the strict codec
        # rejects, which for UTF-8 is exactly the surrogates, as \udXXX.
        return text.encode("utf-8", "backslashreplace").decode("utf-8")
    return text


_CONTROL_ESCAPES: Final[dict[int, str]] = {
    code_point: f"\\u{code_point:04x}" for code_point in (*range(0x00, 0x20), *range(0x7F, 0xA0))
}
"""The 65 control code points (Unicode general category Cc) and the six characters each becomes in a log value (BR-028)."""


def escape_for_log(text: str) -> str:
    """Write a log value that a model or a remote MCP server can control without any control character (BR-028).

    The value first goes through :func:`escape_surrogates`. Then each of the
    65 control characters U+0000-U+001F and U+007F-U+009F (the Unicode
    general category Cc) becomes six characters: a backslash, ``u`` and the
    code point as four lowercase hex digits, for example ``\\u001b`` for
    ESC. ``json.dumps`` writes the same text for 60 of them by default; it
    writes backspace, tab, line feed, form feed and carriage return as
    two-character escapes. Every other code point is left as it is, so a
    value with no control character and no surrogate code point comes back
    equal (as a plain ``str``). The bidirectional formatting characters,
    zero-width characters and U+2028/U+2029 are left as they are too (BR-028
    plan, decision D1).

    Why: under structlog's default configuration, which the SDK does not
    change, structlog 26.1.0's ``ConsoleRenderer`` writes a string value as
    it is unless it holds a space, tab, ``=``, a quote, CR or LF, and
    24.1.0 writes every string value as it is. A control character in a
    value written as it is reached the log output as it was, and on 24.1.0 a
    line feed split the line (measured on CPython 3.11.15, 3.13.2 and
    3.14.3; BR-028 evidence, P1). The MCP and registry log lines passed a
    surrogate code point as it was too, so with a strict UTF-8 stdout they
    raised as ``tool_invoked`` did before BR-024; the first step now escapes
    it there as well (BR-028 evidence, P3).

    For log values only. Never use it on text sent to a model, on events or
    on stored text. Like the surrogate escape it cannot be reversed: the
    escape cannot be told apart from the same six characters typed, which
    are left as they are. It catches nothing.

    Args:
        text: A tool name or an error message that a model or an MCP server
            can control, about to be logged.

    Returns:
        ``text`` with each surrogate code point and each control character
        escaped.
    """
    return escape_surrogates(text).translate(_CONTROL_ESCAPES)


__all__ = ["dumps_for_model", "escape_for_log", "escape_surrogates"]
