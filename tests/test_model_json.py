"""Tests for the private ``fifty_agent_sdk._model_json.dumps_for_model`` helper (BR-020).

The helper renders the JSON the SDK writes for a model to read: non-ASCII
text literal (``ensure_ascii=False``), with the escaped ``json.dumps`` default
form as an all-or-nothing fallback when the text cannot be encoded as UTF-8.

What these pin: literal output and its length for non-ASCII payloads; equality
with ``json.dumps`` defaults, as text and as UTF-8 bytes, for payloads whose
strings hold only U+0000-U+007E, including every code point 0x00-0x7E as a key
and as a value; U+007F as the one ASCII difference; the surrogate fallback,
including that it passes ``sort_keys`` and ``default`` on; and that
``json.dumps``'s own exceptions propagate. The three call sites are pinned in
``tests/loop/test_loop_non_ascii.py`` and ``tests/llm/test_openai_compat.py``.

BR-022 adds ``escape_surrogates``, which writes each surrogate code point
(U+D800-U+DFFF) in tool-result message text as its ``\\udXXX`` escape. Pinned
here: every other code point is left alone and the same object comes back;
each of the 2048 surrogate code points gets exactly ``json.dumps``' escape;
only the surrogates change, two in a row are escaped one by one; and typed
escape text is left as it is, so the escape cannot be undone. Its call site
is pinned in ``tests/loop/test_loop_tool_result_containment.py``.

BR-028 adds ``escape_for_log``, for a log value a model or a remote MCP
server controls: ``escape_surrogates`` first, then each of the 65 control
code points U+0000-U+001F and U+007F-U+009F (Unicode category Cc) as six
characters, a backslash, ``u`` and four lowercase hex digits. Pinned here:
each of the 65 gets exactly that escape, and the two ranges equal the
category Cc on the running interpreter (U1); every other code point that is
not a surrogate comes back unchanged, the bidirectional formatting
characters, zero-width characters, U+2028, U+2029 and NBSP included (U2);
each surrogate code point gets ``escape_surrogates``' escape (U3); typed
escape text is left as it is, so this escape cannot be undone either (U4);
and ``escape_surrogates`` itself still leaves control characters alone, so
the text the SDK sends or stores keeps them (U5). The tests reach the
helper as ``_model_json.escape_for_log`` so that each fails on its own on a
tree without it. Its call sites are pinned in
``tests/loop/test_loop_provider_text_replay.py`` (B16-B19) and
``tests/tools/test_tools_console_log.py`` (T1-T6).
"""

from __future__ import annotations

import json
import unicodedata
from datetime import UTC, datetime
from typing import Any

import pytest

from fifty_agent_sdk import _model_json
from fifty_agent_sdk._model_json import dumps_for_model, escape_surrogates

_MIXED_NON_ASCII: dict[str, Any] = {
    "name": "فاطمة الزهراء",
    "city": "東京",
    "accented": "José Núñez, Zoë",
    "emoji": "ok 😀",
    "مفتاح": ["قيمة", {"nested": "naïve café"}],
    "count": 3,
}
"""Arabic, CJK, accented Latin, a character outside the BMP, a non-ASCII key, nesting."""

_ASCII_PAYLOADS = [
    pytest.param({"b": [1, 2, {"c": None}], "a": {"z": True, "y": False}}, id="nested"),
    pytest.param(
        {"quote": 'say "hi"', "path": "C:\\dir\\file", "slash": "a/b"},
        id="quotes_and_backslashes",
    ),
    pytest.param(
        {"tabs": "a\nb\tc\rd", "controls": "".join(chr(cp) for cp in range(0x20))},
        id="control_characters",
    ),
    pytest.param("".join(chr(cp) for cp in range(0x20, 0x7F)), id="printable_ascii"),
    pytest.param([0, -1, 1.5, 1e100, 2**64, True, False, None, ""], id="scalars"),
    pytest.param([float("nan"), float("inf"), float("-inf")], id="non_finite_floats"),
    pytest.param({2: "two", 1.5: "one and a half", False: "false key"}, id="non_str_keys"),
    pytest.param({None: "null key"}, id="none_key"),
]
"""Payloads whose strings hold only U+0000-U+007E; every dict with two or more keys lists them out of sorted order."""


def test_dumps_for_model_keeps_non_ascii_literal() -> None:
    """Arabic text comes out as the characters themselves, with no ``\\u`` escape (BR-020)."""
    result = dumps_for_model({"name": "فاطمة"})

    assert result == '{"name": "فاطمة"}'
    assert "\\u" not in result


def test_dumps_for_model_length_equals_unescaped_dumps() -> None:
    """A non-ASCII payload's output, and so its length, equals ``json.dumps(..., ensure_ascii=False)``; it decodes to the same value (BR-020)."""
    escaped = json.dumps(_MIXED_NON_ASCII)
    unescaped = json.dumps(_MIXED_NON_ASCII, ensure_ascii=False)
    # Precondition: the payload really exercises the bug.
    assert len(unescaped) < len(escaped)

    result = dumps_for_model(_MIXED_NON_ASCII)

    assert result == unescaped
    assert len(result) == len(unescaped)
    assert json.loads(result) == json.loads(escaped) == _MIXED_NON_ASCII


@pytest.mark.parametrize("sort_keys", [False, True], ids=["unsorted", "sorted"])
@pytest.mark.parametrize("payload", _ASCII_PAYLOADS)
def test_dumps_for_model_ascii_matches_default_dumps(payload: Any, sort_keys: bool) -> None:
    """A payload whose strings hold only U+0000-U+007E serialises exactly as ``json.dumps`` defaults do, as text and as UTF-8 bytes (BR-020)."""
    expected = json.dumps(payload, sort_keys=sort_keys)

    result = dumps_for_model(payload, sort_keys=sort_keys)

    assert result == expected
    assert result.encode("utf-8") == expected.encode("utf-8")


def test_dumps_for_model_ascii_code_points_match_default_except_del() -> None:
    """Every code point 0x00-0x7E, in a key and in a value, serialises as ``json.dumps`` defaults do (BR-020).

    U+007F (DEL) is the single ASCII exception, which the CHANGELOG states: the
    default form writes ``\\u007f`` and the helper writes the character itself.
    """
    for cp in range(0x7F):
        payload = {"k" + chr(cp): "v" + chr(cp)}
        assert dumps_for_model(payload) == json.dumps(payload), f"code point {cp:#04x}"

    del_payload = {"k\x7f": "v\x7f"}
    assert json.dumps(del_payload) == '{"k\\u007f": "v\\u007f"}'
    assert dumps_for_model(del_payload) == '{"k\x7f": "v\x7f"}'


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"s": chr(0xD800)}, id="lone_high_surrogate"),
        pytest.param({"a": "فاطمة", "s": chr(0xDC00)}, id="surrogate_beside_arabic"),
        pytest.param({"s": chr(0xD83D) + chr(0xDE00)}, id="two_surrogate_code_points"),
    ],
)
def test_dumps_for_model_surrogate_code_point_falls_back_to_escaped_form(
    payload: dict[str, str],
) -> None:
    """A payload holding a surrogate code point gets the escaped ``json.dumps`` default form, which encodes as UTF-8 (BR-020).

    The fallback is all-or-nothing: Arabic text beside the surrogate is escaped too.
    """
    # Precondition: the literal form really cannot be encoded.
    with pytest.raises(UnicodeEncodeError):
        json.dumps(payload, ensure_ascii=False).encode("utf-8")

    result = dumps_for_model(payload)

    assert result == json.dumps(payload)
    result.encode("utf-8")  # must not raise


def test_dumps_for_model_surrogate_fallback_keeps_sort_keys() -> None:
    """The surrogate fallback passes ``sort_keys`` on: its text equals ``json.dumps(payload, sort_keys=True)`` (BR-020)."""
    payload = {"z": chr(0xD800), "a": "فاطمة"}
    expected = json.dumps(payload, sort_keys=True)
    # Precondition: sorting changes the escaped text, so a fallback that drops it is visible.
    assert expected != json.dumps(payload)

    result = dumps_for_model(payload, sort_keys=True)

    assert result == expected
    result.encode("utf-8")  # must not raise


def test_dumps_for_model_surrogate_fallback_keeps_default() -> None:
    """The surrogate fallback passes ``default`` on: a datetime beside a surrogate gives ``json.dumps(payload, default=str)`` (BR-020)."""
    payload = {"when": datetime(2026, 10, 2, 9, 1, tzinfo=UTC), "s": chr(0xD800)}
    expected = json.dumps(payload, default=str)
    # Precondition: without ``default`` the escaped form cannot be built at all.
    with pytest.raises(TypeError):
        json.dumps(payload)

    result = dumps_for_model(payload, default=str)

    assert result == expected
    result.encode("utf-8")  # must not raise


def test_dumps_for_model_propagates_serialisation_errors() -> None:
    """An exception from ``json.dumps`` itself leaves the helper unchanged, so each caller keeps its own fallback (BR-020)."""

    class _Opaque:
        pass

    def _raising_default(value: object) -> object:
        raise TypeError("cannot serialise this value")

    with pytest.raises(TypeError, match="cannot serialise this value"):
        dumps_for_model({"x": _Opaque()}, default=_raising_default)

    circular: list[Any] = []
    circular.append(circular)
    with pytest.raises(ValueError, match="Circular reference"):
        dumps_for_model(circular)


# --- escape_surrogates (BR-022) --------------------------------------------------------


def test_escape_surrogates_leaves_text_without_surrogates_unchanged() -> None:
    """Text holding every code point except U+D800-U+DFFF comes back unchanged, as the same object (BR-022).

    That covers ASCII, U+007F, non-ASCII text and characters outside the
    Basic Multilingual Plane.
    """
    text = "".join(chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)
    # Precondition: the text is complete and encodes as UTF-8.
    assert len(text) == 0x110000 - 0x800
    text.encode("utf-8")

    result = escape_surrogates(text)

    assert result is text
    assert result == text


def test_escape_surrogates_writes_each_surrogate_as_its_json_escape() -> None:
    """Each of the 2048 surrogate code points becomes the six characters ``json.dumps`` writes for it, lowercase, and the result encodes as UTF-8 (BR-022)."""
    for cp in range(0xD800, 0xE000):
        expected = f"\\u{cp:04x}"
        assert json.dumps(chr(cp))[1:-1] == expected, f"code point {cp:#06x}"

        result = escape_surrogates(chr(cp))

        assert result == expected, f"code point {cp:#06x}"
        result.encode("utf-8")  # must not raise


def test_escape_surrogates_escapes_only_the_surrogates() -> None:
    """Only the surrogate code points change: Arabic beside one stays literal, two in a row are escaped one by one, and a ``surrogateescape`` decode is covered (BR-022)."""
    assert escape_surrogates("فاطمة" + chr(0xD800) + "x") == "فاطمة\\ud800x"
    assert escape_surrogates(chr(0xD83D) + chr(0xDE00)) == "\\ud83d\\ude00"
    decoded = b"caf\xe9".decode("utf-8", "surrogateescape")
    # Precondition: the decode produced a surrogate code point.
    assert decoded == "caf" + chr(0xDCE9)
    assert escape_surrogates(decoded) == "caf\\udce9"


def test_escape_surrogates_leaves_escape_text_alone() -> None:
    """The six ASCII characters ``\\ud800`` come back as they are, so a typed escape and an escaped code point look the same (BR-022)."""
    typed = "\\ud800"
    assert len(typed) == 6

    assert escape_surrogates(typed) is typed
    assert escape_surrogates(typed + chr(0xD800)) == typed + typed


# --- escape_for_log (BR-028) --------------------------------------------------------------

_CONTROL_CODE_POINTS = [*range(0x00, 0x20), *range(0x7F, 0xA0)]
"""C0 (U+0000-U+001F, TAB, LF and CR included), DEL and C1 (U+0080-U+009F): the 65 BR-028 escapes."""

_BIDI_CONTROLS = [0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)]


def test_escape_for_log_escapes_every_control_code_point() -> None:
    """U1: each of the 65 control code points becomes a backslash, ``u`` and four lowercase hex digits, and the two ranges are exactly the category Cc (BR-028).

    The set is checked against ``unicodedata.category`` over every code point
    on the running interpreter, an independent derivation. The expected text
    is built from its parts, never from the helper.
    """
    category_cc = {cp for cp in range(0x110000) if unicodedata.category(chr(cp)) == "Cc"}
    assert set(_CONTROL_CODE_POINTS) == category_cc
    assert len(_CONTROL_CODE_POINTS) == 65

    for cp in _CONTROL_CODE_POINTS:
        expected = "a" + "\\u" + format(cp, "04x") + "b"
        assert len(expected) == 8

        assert _model_json.escape_for_log("a" + chr(cp) + "b") == expected, f"{cp:#06x}"

    every = "".join(map(chr, _CONTROL_CODE_POINTS))
    escaped = _model_json.escape_for_log(every)
    assert escaped == "".join("\\u" + format(cp, "04x") for cp in _CONTROL_CODE_POINTS)
    assert escaped.isascii()
    assert escaped.isprintable()


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(" ", id="space"),
        pytest.param("~", id="tilde_u007e"),
        pytest.param("\u00a0", id="nbsp"),
        pytest.param("\u00a1", id="u00a1"),
        *[pytest.param("find" + chr(cp) + "x", id=f"bidi_{cp:04x}") for cp in _BIDI_CONTROLS],
        pytest.param("\u200b\u200c\u200d", id="zero_width"),
        pytest.param("a\u2028b\u2029c", id="line_and_paragraph_separators"),
        pytest.param("\ufeff", id="bom"),
        pytest.param("بحث", id="arabic"),
        pytest.param("ok \U0001f600", id="astral_emoji"),
    ],
)
def test_escape_for_log_leaves_named_code_points_alone(text: str) -> None:
    """U2, named rows: NBSP, the bidirectional formatting characters, zero-width characters, U+2028/U+2029, Arabic and an astral emoji come back unchanged (BR-028, decision D1)."""
    assert _model_json.escape_for_log(text) == text


def test_escape_for_log_leaves_every_other_code_point_alone() -> None:
    """U2: a text holding every code point that is neither Cc nor a surrogate comes back unchanged (BR-028).

    That is printable ASCII, U+00A0 onwards, and characters outside the
    Basic Multilingual Plane: a value without a control character or a
    surrogate code point is logged as before.
    """
    controls = set(_CONTROL_CODE_POINTS)
    text = "".join(
        chr(cp) for cp in range(0x110000) if cp not in controls and not 0xD800 <= cp <= 0xDFFF
    )
    # Precondition: the text is complete.
    assert len(text) == 0x110000 - 0x800 - 65

    assert _model_json.escape_for_log(text) == text


def test_escape_for_log_escapes_surrogates_as_escape_surrogates_does() -> None:
    """U3: each surrogate code point gets ``escape_surrogates``' six characters, and a mixed text gets both escapes and nothing else changes (BR-028)."""
    for cp in range(0xD800, 0xE000):
        assert _model_json.escape_for_log(chr(cp)) == escape_surrogates(chr(cp)), f"{cp:#06x}"

    mixed = "x" + chr(0xD800) + "\x1b" + "\x85" + "بحث"
    assert _model_json.escape_for_log(mixed) == "x\\ud800\\u001b\\u0085بحث"


def test_escape_for_log_leaves_escape_text_alone() -> None:
    """U4: the six characters of the ESC escape, typed, and those of BR-024's U+D800 escape come back as they are, so a typed escape and an escaped control character look the same (BR-028)."""
    typed = "\\u001b"
    typed_surrogate = "\\ud800"
    assert len(typed) == len(typed_surrogate) == 6

    assert _model_json.escape_for_log(typed) == typed
    assert _model_json.escape_for_log(typed_surrogate) == typed_surrogate
    assert _model_json.escape_for_log(typed + "\x1b") == typed + typed


def test_escape_surrogates_still_leaves_control_characters_alone() -> None:
    """U5: ``escape_surrogates``, which runs on the wire and on stored text, returns a text with ESC, LF and TAB as the same object (BR-028, decision D3)."""
    text = "a\x1b\n\tb"

    assert escape_surrogates(text) is text
