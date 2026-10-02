"""JSON text that the SDK writes for a model to read (BR-020).

Routing rule: every ``json.dumps`` whose output reaches a model prompt goes
through :func:`dumps_for_model`. Since 1.10.2 those are the three call sites
that render tool results, the text-mode tool list and the ``arguments`` string
of a replayed native tool call. JSON that is stored, hashed or used as a key
(for example the Redis branch metadata in ``state/redis.py``) never goes
through it and keeps the stdlib defaults, so stored bytes do not change.

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


__all__ = ["dumps_for_model"]
