"""Text that the SDK writes for a model to read: JSON (BR-020) and tool-result message text (BR-022).

Routing rule: every ``json.dumps`` whose output reaches a model prompt goes
through :func:`dumps_for_model`. Since 1.10.2 those are the three call sites
that render tool results, the text-mode tool list and the ``arguments`` string
of a replayed native tool call. JSON that is stored, hashed or used as a key
(for example the Redis branch metadata in ``state/redis.py``) never goes
through it and keeps the stdlib defaults, so stored bytes do not change.

:func:`escape_surrogates` (BR-022) runs once on the text of every
tool-result message the loop builds (``AgentLoop._build_tool_message``), so
a surrogate code point in it reaches the model as its ``\\udXXX`` escape.
The same rule holds for it: never use it on text that is stored, hashed or
used as a key.

This module is private and carries no semver protection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


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
        # pre-1.10.2 escaped form, which is ASCII, so this text encodes.
        return json.dumps(obj, default=default, sort_keys=sort_keys)
    return text


def escape_surrogates(text: str) -> str:
    """Write each surrogate code point in model-facing text as its ``\\udXXX`` escape (BR-022).

    UTF-8 can encode every code point except the surrogates, U+D800-U+DFFF.
    A Python ``str`` can still hold them (``json.loads('"\\ud800"')`` or a
    ``surrogateescape`` decode makes one), and the ``openai`` releases BR-022
    measured (2.43.0 and 2.54.0) encode the request body as strict UTF-8, so
    one surrogate code point in a tool-result message made the next request
    raise ``UnicodeEncodeError`` before it was sent.

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
    this on text that is stored, hashed or used as a key.

    Args:
        text: The text of a message the SDK is about to send to a model.

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


__all__ = ["dumps_for_model", "escape_surrogates"]
