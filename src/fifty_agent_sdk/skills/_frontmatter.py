"""Private strict reader for the YAML frontmatter of a ``SKILL.md`` file (FR-004 D1).

The SDK reads two keys, ``name`` and ``description``, from a skill file's
frontmatter. It does not depend on a YAML library to do so (a runtime
dependency is a consumer contract; coding_guidelines, Decisions, TD-004 and
TD-007). Instead it accepts a small subset of YAML and REJECTS everything
outside it, with a reason code and a 1-based line number.

The subset is chosen so that, for every input this reader accepts, the
``name`` and ``description`` it returns equal what ``yaml.safe_load`` (PyYAML,
YAML 1.1) reads for those keys, as ``str``. ``tests/skills/
test_frontmatter_differential.py`` checks this against PyYAML, within two
stated limits:

* Keys other than ``name`` and ``description`` are not interpreted (rule 5),
  so an input whose YAML error sits only in such a key's value (``metadata:
  a: b``) is accepted here while ``yaml.safe_load`` raises. For those inputs
  the test (D-3) compares against the document reduced to its ``name`` and
  ``description`` entries.
* No YAML 1.2 loader is run. That no YAML 1.2 loader reads an accepted value
  as anything but that string rests on reading this module's pattern
  constants, which also refuse the YAML 1.2 core schema's implicit types.

The grammar (plan FR-004, "Accepted frontmatter grammar"):

1. One leading BOM is stripped, and ``\\r\\n`` and lone ``\\r`` become
   ``\\n``.
2. Line 1 is ``---`` (trailing spaces or tabs allowed). The frontmatter runs
   to the next line that is ``---`` (trailing spaces or tabs allowed).
3. The body is everything after the closing line, without the newlines that
   follow it and without trailing whitespace. It must not be empty.
4. In the frontmatter, blank lines and comment lines are skipped. Every
   column-0 line is a key line, ``KEY:`` then the end of the line or spaces
   and a value, with KEY matching ``[A-Za-z0-9_][A-Za-z0-9_-]*``; a line
   indented with spaces belongs to the key above it. A key may appear once.
5. Keys other than ``name`` and ``description`` are tolerated and not
   interpreted (D7), with one check: a quoted value or a ``[``/``{``
   collection in them must close on its own line. YAML lets those span lines
   regardless of indentation, so an unclosed one could swallow a following
   ``name:`` line that this reader would take as a key.
6. ``name`` and ``description`` are a plain scalar (optionally continued on
   indented lines), a one-line single- or double-quoted scalar, or a ``|`` /
   ``>`` block scalar. A plain value YAML would read as something other than
   a string (``no``, ``on``, ``~``, ``123``, ``0x1f``, ``1.5``, ``.inf``,
   ``2024-01-01``, ``<<``, ``=``) is rejected with "quote this value".

Characters refused anywhere in the frontmatter: TAB (PyYAML does not skip
tabs between tokens, so their meaning differs between YAML loaders), the
characters PyYAML's reader refuses (``yaml/reader.py`` ``NON_PRINTABLE``:
C0 and C1 controls other than line feed, U+007F, U+FFFE, U+FFFF, surrogate
code points), the characters YAML treats as line breaks (U+0085, U+2028,
U+2029) and U+FEFF. Refusing them costs little and keeps the line model of
this reader equal to YAML's.

Every function here runs in time linear in its input, with no recursion.
Nothing here logs, and no message quotes a value from the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

_KNOWN_FIELDS: Final = ("name", "description")
"""The two keys the SDK interprets. Every other key is tolerated and skipped (D7)."""

_KEY_LINE: Final = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_-]*):(?: +(.*))?")
"""A top-level key line: ``KEY:`` then the end of the line, or spaces and a value."""

_DELIMITER: Final = re.compile(r"---[ \t]*")
"""An opening or closing frontmatter delimiter line (trailing spaces or tabs allowed)."""

_REFUSED_CHARACTER: Final = re.compile(
    "[^\n\x20-\x7e\xa0-\u2027\u202a-\ud7ff\ue000-\ufefe\uff00-\ufffd\U00010000-\U0010ffff]"
)
"""A character the frontmatter may not hold (see the module docstring).

Derived from PyYAML's ``Reader.NON_PRINTABLE`` (``yaml/reader.py``), with TAB,
CR, U+0085, U+2028, U+2029 and U+FEFF removed from the accepted set.
"""

_BLOCK_INDICATOR: Final = re.compile(r"([|>])([-+]?)(?: *| +#.*)")
"""A block scalar header: ``|`` or ``>``, optional chomping, then nothing but a comment."""

# --- YAML implicit types --------------------------------------------------
# A plain scalar matching any of these is read by some YAML loader as a value
# that is not a string, so this reader rejects it ("quote this value").
#
# The first seven are copied verbatim from PyYAML 6.0.3's
# ``yaml/resolver.py`` (``Resolver.add_implicit_resolver``), the YAML 1.1
# loader the differential test checks against.

_YAML11_BOOL: Final = re.compile(
    r"""^(?:yes|Yes|YES|no|No|NO
    |true|True|TRUE|false|False|FALSE
    |on|On|ON|off|Off|OFF)$""",
    re.X,
)
_YAML11_FLOAT: Final = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
    |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_YAML11_INT: Final = re.compile(
    r"""^(?:[-+]?0b[0-1_]+
    |[-+]?0[0-7_]+
    |[-+]?(?:0|[1-9][0-9_]*)
    |[-+]?0x[0-9a-fA-F_]+
    |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$""",
    re.X,
)
_YAML11_MERGE: Final = re.compile(r"^(?:<<)$")
_YAML11_NULL: Final = re.compile(
    r"""^(?: ~
    |null|Null|NULL
    | )$""",
    re.X,
)
_YAML11_TIMESTAMP: Final = re.compile(
    r"""^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
    |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
     (?:[Tt]|[ \t]+)[0-9][0-9]?
     :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
     (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$""",
    re.X,
)
_YAML11_VALUE: Final = re.compile(r"^(?:=)$")

# The YAML 1.1 specification's own boolean and null sets are wider than
# PyYAML's (``y``, ``n`` and any letter case), and YAML 1.2 loaders (the core
# schema, and js-yaml's default schema with underscores and ``0b``/``0o``)
# read numbers PyYAML keeps as strings, such as ``1e5`` and ``0o17``.
# These are rejected too, so the accepted subset is a string in both.
_ANY_CASE_BOOL_OR_NULL: Final = re.compile(r"(?:y|yes|n|no|true|false|on|off|~|null)", re.I)
_YAML12_INT: Final = re.compile(r"[-+]?(?:0b[01_]+|0o[0-7_]+|0x[0-9a-fA-F_]+|[0-9][0-9_]*)")
_YAML12_FLOAT: Final = re.compile(
    r"[-+]?(?:\.[0-9][0-9_]*|[0-9][0-9_]*(?:\.[0-9_]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)"
)

_IMPLICIT_TYPES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (_YAML11_BOOL, "a boolean"),
    (_ANY_CASE_BOOL_OR_NULL, "a boolean or a null"),
    (_YAML11_NULL, "a null"),
    (_YAML11_INT, "a number"),
    (_YAML12_INT, "a number"),
    (_YAML11_FLOAT, "a number"),
    (_YAML12_FLOAT, "a number"),
    (_YAML11_TIMESTAMP, "a date"),
    (_YAML11_MERGE, "a merge key"),
    (_YAML11_VALUE, "a special value"),
)
"""Each implicit-type pattern with the words used in its rejection message."""

_PLAIN_FIRST_REFUSED: Final = frozenset("[]{},&*!|>'\"%@`#")
"""Characters that cannot start a plain scalar in YAML (``#`` starts a comment)."""

_DOUBLE_QUOTED_ESCAPES: Final[dict[str, str]] = {
    "\\": "\\",
    '"': '"',
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\x00",
}
"""The double-quoted escapes this reader accepts (a subset of PyYAML's ``ESCAPE_REPLACEMENTS``)."""

_HEX_ESCAPE_LENGTHS: Final[dict[str, int]] = {"x": 2, "u": 4, "U": 8}
"""``\\xXX``, ``\\uXXXX`` and ``\\UXXXXXXXX``: hex digits after each escape letter."""

_HEX_DIGITS: Final = frozenset("0123456789abcdefABCDEF")

# Scanner results for the values of keys this reader does not interpret.
_SCAN_OK: Final = 0
_SCAN_OPEN: Final = 1
_SCAN_BLOCK: Final = 2

# Scanner states.
_AT_TOKEN: Final = 0
_IN_PLAIN: Final = 1
_AFTER_TOKEN: Final = 2


class _FrontmatterError(Exception):
    """A ``SKILL.md`` text this reader refuses.

    Attributes:
        reason: A fixed code, one of the :class:`~fifty_agent_sdk.errors.SkillError`
            reasons (``no_frontmatter``, ``unclosed_frontmatter``,
            ``malformed_frontmatter``, ``duplicate_key``, ``missing_field``,
            ``invalid_field``, ``empty_body``).
        detail: A message that names the problem without quoting the file.
        line: The 1-based line of ``SKILL.md`` it is on, or ``None``.
        field: ``"name"``, ``"description"`` or ``"body"`` when the problem is
            in one of them, else ``None``.
    """

    def __init__(
        self, reason: str, detail: str, *, line: int | None = None, field: str | None = None
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.line = line
        self.field = field


@dataclass(frozen=True, slots=True)
class _Parsed:
    """What :func:`parse_skill_md` read.

    Attributes:
        name: The ``name`` value, as YAML reads it (not stripped).
        description: The ``description`` value, as YAML reads it (not stripped).
        frontmatter: The text between the two delimiter lines, each line ending
            in ``\\n`` (newlines already normalised).
        body: The text after the closing delimiter, trimmed as the grammar says.
        name_line: The 1-based line of the ``name:`` key.
        description_line: The 1-based line of the ``description:`` key.
    """

    name: str
    description: str
    frontmatter: str
    body: str
    name_line: int
    description_line: int


def normalise_text(text: str) -> str:
    """Strip one leading BOM and turn ``\\r\\n`` and lone ``\\r`` into ``\\n``."""
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_skill_md(text: str) -> _Parsed:
    """Read the frontmatter and body of a ``SKILL.md`` text (FR-004 D1).

    Args:
        text: The file's text. A leading BOM and ``\\r\\n`` / ``\\r`` line
            endings are normalised here too, so a caller may pass raw text.

    Returns:
        The ``name``, ``description``, raw frontmatter and body.

    Raises:
        _FrontmatterError: When the text is outside the accepted grammar
            (module docstring), with a reason code and, where there is one,
            the 1-based line.
    """
    lines = normalise_text(text).split("\n")
    if _DELIMITER.fullmatch(lines[0]) is None:
        raise _FrontmatterError(
            "no_frontmatter", "the file must start with a line holding only ---", line=1
        )
    closing = next(
        (index for index in range(1, len(lines)) if _DELIMITER.fullmatch(lines[index])), None
    )
    if closing is None:
        raise _FrontmatterError(
            "unclosed_frontmatter",
            "the frontmatter that starts on line 1 has no closing --- line",
            line=1,
        )
    frontmatter_lines = lines[1:closing]
    frontmatter = "".join(f"{line}\n" for line in frontmatter_lines)
    _check_characters(frontmatter)
    values = _read_fields(frontmatter_lines)
    for field in _KNOWN_FIELDS:
        if field not in values:
            raise _FrontmatterError(
                "missing_field", f"the frontmatter has no {field} key", field=field
            )
    body = "\n".join(lines[closing + 1 :]).lstrip("\n").rstrip()
    if not body:
        raise _FrontmatterError(
            "empty_body", "the file has no instructions after the frontmatter", field="body"
        )
    name, name_line = values["name"]
    description, description_line = values["description"]
    return _Parsed(
        name=name,
        description=description,
        frontmatter=frontmatter,
        body=body,
        name_line=name_line,
        description_line=description_line,
    )


def _check_characters(frontmatter: str) -> None:
    """Refuse a character the frontmatter may not hold, naming its line and code point."""
    match = _REFUSED_CHARACTER.search(frontmatter)
    if match is None:
        return
    code_point = ord(match.group())
    line = frontmatter.count("\n", 0, match.start()) + 2
    if code_point == 0x09:
        detail = "a TAB character is not allowed in the frontmatter; use spaces"
    else:
        detail = f"the character U+{code_point:04X} is not allowed in the frontmatter"
    raise _FrontmatterError("malformed_frontmatter", detail, line=line)


def _indent(line: str) -> int:
    """The number of leading spaces."""
    return len(line) - len(line.lstrip(" "))


def _is_blank(line: str) -> bool:
    """A line holding only spaces (tabs were refused before this runs)."""
    return not line.strip(" ")


def _is_comment(line: str) -> bool:
    """A line whose first non-space character is ``#``."""
    return line.lstrip(" ").startswith("#")


def _read_fields(frontmatter_lines: list[str]) -> dict[str, tuple[str, int]]:
    """Split the frontmatter into top-level keys and read ``name`` and ``description``.

    Returns:
        ``{field: (value, key_line)}`` for each known field present.
    """
    # Group the lines: each top-level key owns the lines up to the next key line.
    entries: list[tuple[str, int, str, list[tuple[int, str]]]] = []
    for offset, line in enumerate(frontmatter_lines):
        line_number = offset + 2
        if line.startswith(" ") or not line:
            if entries:
                entries[-1][3].append((line_number, line))
            elif not _is_blank(line) and not _is_comment(line):
                raise _FrontmatterError(
                    "malformed_frontmatter",
                    "an indented line comes before the first key",
                    line=line_number,
                )
            continue
        if line.startswith("#"):
            if entries:
                entries[-1][3].append((line_number, line))
            continue
        match = _KEY_LINE.fullmatch(line)
        if match is None:
            raise _FrontmatterError(
                "malformed_frontmatter",
                "a line that starts in column 1 must be a key such as name: followed "
                "by a space and its value",
                line=line_number,
            )
        entries.append((match.group(1), line_number, match.group(2) or "", []))

    seen: set[str] = set()
    values: dict[str, tuple[str, int]] = {}
    for key, line_number, inline, following in entries:
        if key in seen:
            raise _FrontmatterError(
                "duplicate_key",
                f"the key {key} appears more than once",
                line=line_number,
                field=key if key in _KNOWN_FIELDS else None,
            )
        seen.add(key)
        if key in _KNOWN_FIELDS:
            values[key] = (_read_value(key, line_number, inline, following), line_number)
        else:
            _check_skipped(key, line_number, inline, following)
    return values


# ---------------------------------------------------------------------------
# name and description
# ---------------------------------------------------------------------------


def _invalid(field: str, line: int, detail: str) -> _FrontmatterError:
    return _FrontmatterError("invalid_field", f"{field}: {detail}", line=line, field=field)


def _read_value(field: str, line: int, inline: str, following: list[tuple[int, str]]) -> str:
    """Read the value of ``name`` or ``description`` in one of the accepted styles."""
    if not inline or inline.startswith("#"):
        raise _invalid(
            field,
            line,
            "the value is empty; put it on the same line as the key, or use | or >",
        )
    first = inline[0]
    if first == "'" or first == '"':
        return _read_quoted(field, line, inline, following)
    if first == "|" or first == ">":
        return _read_block(field, line, inline, following)
    return _read_plain(field, line, inline, following)


def _reject_continuation(field: str, following: list[tuple[int, str]], why: str) -> None:
    """Refuse any indented line under the key other than blank and comment lines."""
    for line_number, text in following:
        if not _is_blank(text) and not _is_comment(text):
            raise _invalid(field, line_number, why)


def _check_after_quote(field: str, line: int, rest: str) -> None:
    """After a closing quote only spaces and an optional ``#`` comment may follow."""
    stripped = rest.lstrip(" ")
    if not stripped:
        return
    if stripped.startswith("#") and len(stripped) < len(rest):
        return
    raise _invalid(field, line, "only a comment (after a space) may follow the closing quote")


def _read_quoted(field: str, line: int, inline: str, following: list[tuple[int, str]]) -> str:
    """A single- or double-quoted value that opens and closes on the key line."""
    quote = inline[0]
    chunks: list[str] = []
    index = 1
    length = len(inline)
    closed_at = -1
    while index < length:
        char = inline[index]
        if quote == "'":
            if char == "'":
                if index + 1 < length and inline[index + 1] == "'":
                    chunks.append("'")
                    index += 2
                    continue
                closed_at = index
                break
            chunks.append(char)
            index += 1
            continue
        if char == '"':
            closed_at = index
            break
        if char != "\\":
            chunks.append(char)
            index += 1
            continue
        if index + 1 >= length:
            break
        letter = inline[index + 1]
        if letter in _DOUBLE_QUOTED_ESCAPES:
            chunks.append(_DOUBLE_QUOTED_ESCAPES[letter])
            index += 2
            continue
        digits = _HEX_ESCAPE_LENGTHS.get(letter)
        if digits is None:
            raise _invalid(
                field,
                line,
                'unsupported escape in a double-quoted value; use \\\\, \\", \\/, \\n, '
                "\\t, \\r, \\0, \\xXX, \\uXXXX or \\UXXXXXXXX",
            )
        hex_text = inline[index + 2 : index + 2 + digits]
        if len(hex_text) != digits or any(c not in _HEX_DIGITS for c in hex_text):
            raise _invalid(
                field, line, f"\\{letter} must be followed by {digits} hexadecimal digits"
            )
        code_point = int(hex_text, 16)
        if 0xD800 <= code_point <= 0xDFFF or code_point > 0x10FFFF:
            raise _invalid(
                field, line, "an escape may not stand for a surrogate or out-of-range code point"
            )
        chunks.append(chr(code_point))
        index += 2 + digits
    if closed_at < 0:
        raise _invalid(field, line, "a quoted value must end on the same line")
    _check_after_quote(field, line, inline[closed_at + 1 :])
    _reject_continuation(field, following, "a quoted value must be on one line")
    return "".join(chunks)


def _read_block(field: str, line: int, inline: str, following: list[tuple[int, str]]) -> str:
    """A ``|`` (literal) or ``>`` (folded) block scalar, read as PyYAML reads it."""
    header = _BLOCK_INDICATOR.fullmatch(inline)
    if header is None:
        raise _invalid(
            field,
            line,
            "a block value starts with |, |-, |+, >, >- or >+ and nothing after it but a "
            "comment (an indentation number such as |2 is not supported)",
        )
    folded = header.group(1) == ">"
    chomping = header.group(2)

    # The content indentation is the first non-blank line's; leading blank
    # lines may not be longer than it (YAML 1.2 makes that an error).
    first_content = next(
        (position for position, (_n, text) in enumerate(following) if not _is_blank(text)),
        None,
    )
    if first_content is None or _indent(following[first_content][1]) == 0:
        raise _invalid(field, line, "the block value has no indented content")
    content_indent = _indent(following[first_content][1])
    for line_number, text in following[:first_content]:
        if len(text) > content_indent:
            raise _invalid(
                field,
                line_number,
                "a blank line before the block content has more spaces than the content",
            )

    contents: list[str] = []
    blanks_before: list[int] = []
    pending_blanks = 0
    ended_at: int | None = None
    for position in range(first_content, len(following)):
        line_number, text = following[position]
        if _is_blank(text) and len(text) <= content_indent:
            pending_blanks += 1
            continue
        indent = _indent(text)
        if indent < content_indent:
            ended_at = position
            break
        if folded and indent > content_indent:
            raise _invalid(
                field,
                line_number,
                "a folded value (>) cannot have more-indented lines; use | instead",
            )
        blanks_before.append(pending_blanks)
        pending_blanks = 0
        contents.append(text[content_indent:])
    if ended_at is not None:
        _reject_continuation(
            field,
            following[ended_at:],
            "this line is less indented than the block value above it",
        )

    # Blank lines before the first content line are kept, one "\n" each, in
    # both styles (PyYAML's scan_block_scalar_indentation returns them).
    pieces: list[str] = ["\n" * first_content, contents[0]]
    for text, blanks in zip(contents[1:], blanks_before[1:], strict=True):
        if folded:
            pieces.append(" " if blanks == 0 else "\n" * blanks)
        else:
            pieces.append("\n" * (blanks + 1))
        pieces.append(text)
    if chomping == "":
        pieces.append("\n")
    elif chomping == "+":
        pieces.append("\n" * (pending_blanks + 1))
    return "".join(pieces)


def _strip_plain_comment(text: str) -> tuple[str, bool]:
    """Cut a `` #`` comment off a plain-scalar line; say whether there was one."""
    cut = text.find(" #")
    if cut < 0:
        return text.rstrip(" "), False
    return text[:cut].rstrip(" "), True


def _check_plain_part(field: str, line: int, part: str) -> None:
    """A plain value may not hold ``: `` or end in ``:`` (YAML reads a mapping there)."""
    if ": " in part or part.endswith(":"):
        raise _invalid(field, line, "a value holding ': ' or ending in ':' must be quoted")


def _read_plain(field: str, line: int, inline: str, following: list[tuple[int, str]]) -> str:
    """A plain scalar on the key line, optionally continued on indented lines."""
    first = inline[0]
    if first in _PLAIN_FIRST_REFUSED or (first in "-?:" and (len(inline) == 1 or inline[1] == " ")):
        raise _invalid(field, line, "a value starting with this character must be quoted")
    part, ended = _strip_plain_comment(inline)
    _check_plain_part(field, line, part)
    parts = [part]
    saw_blank = False
    for line_number, text in following:
        if _is_blank(text):
            saw_blank = True
            continue
        if _is_comment(text):
            ended = True
            continue
        if ended or saw_blank:
            raise _invalid(
                field,
                line_number,
                "a value continued over several lines may not hold a blank or comment "
                "line; use | or >",
            )
        part, ended = _strip_plain_comment(text.strip(" "))
        _check_plain_part(field, line_number, part)
        parts.append(part)
    value = " ".join(parts)
    for pattern, kind in _IMPLICIT_TYPES:
        if pattern.fullmatch(value):
            raise _invalid(field, line, f"quote this value; YAML would read it as {kind}, not text")
    return value


# ---------------------------------------------------------------------------
# Keys that are not interpreted
# ---------------------------------------------------------------------------


def _check_skipped(key: str, line: int, inline: str, following: list[tuple[int, str]]) -> None:
    """Skip an uninterpreted key's value, refusing a quote or collection left open (D7)."""
    open_detail = f"a quoted or [ ] / {{ }} value under {key} must close on the line where it opens"
    block_indent: int | None = None
    result = _scan_value(inline)
    if result == _SCAN_OPEN:
        raise _FrontmatterError("malformed_frontmatter", open_detail, line=line)
    if result == _SCAN_BLOCK:
        block_indent = 0
    for line_number, text in following:
        if _is_blank(text):
            continue
        indent = _indent(text)
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        if _is_comment(text):
            continue
        result = _scan_value(text[indent:])
        if result == _SCAN_OPEN:
            raise _FrontmatterError("malformed_frontmatter", open_detail, line=line_number)
        if result == _SCAN_BLOCK:
            block_indent = indent


def _closing_quote(text: str, start: int) -> int:
    """The index of the quote closing the one at ``start``, or ``-1`` if it does not close."""
    quote = text[start]
    index = start + 1
    length = len(text)
    while index < length:
        char = text[index]
        if quote == "'":
            if char == "'":
                if index + 1 < length and text[index + 1] == "'":
                    index += 2
                    continue
                return index
        elif char == "\\":
            index += 2
            continue
        elif char == '"':
            return index
        index += 1
    return -1


def _scan_value(text: str) -> int:
    """Say whether a line of an uninterpreted value leaves a quote or a collection open.

    A light YAML tokenizer for one line: it finds where tokens start, skips
    quoted scalars (refusing one that does not close), counts ``[``/``{``
    depth, and stops at a comment. A ``|`` or ``>`` where a value starts
    (outside a collection) is a block scalar header; the caller then skips the
    more-indented lines after it unread.

    Returns:
        ``_SCAN_OPEN`` for a quote or collection still open at the end of the
        line, ``_SCAN_BLOCK`` for a block scalar header, else ``_SCAN_OK``.
    """
    length = len(text)
    index = 0
    depth = 0
    state = _AT_TOKEN
    while index < length:
        char = text[index]
        if char == " ":
            index += 1
            continue
        if char == "#" and (index == 0 or text[index - 1] == " "):
            break
        next_char = text[index + 1] if index + 1 < length else " "
        if char == ":" and (next_char == " " or (depth and next_char in ",[]{}")):
            state = _AT_TOKEN
            index += 1
            continue
        if state == _IN_PLAIN and not (depth and char in ",[]{}"):
            index += 1
            continue
        if state == _AT_TOKEN and char in "\"'":
            closing = _closing_quote(text, index)
            if closing < 0:
                return _SCAN_OPEN
            index = closing + 1
            state = _AFTER_TOKEN
            continue
        if char in "[{":
            depth += 1
            state = _AT_TOKEN
        elif char in "]}":
            depth = max(depth - 1, 0)
            state = _AFTER_TOKEN
        elif char == ",":
            state = _AT_TOKEN if depth else _AFTER_TOKEN
        elif state == _AT_TOKEN and char in "-?" and next_char == " ":
            pass
        elif state == _AT_TOKEN and char in "&!":
            while index + 1 < length and text[index + 1] != " ":
                if depth and text[index + 1] in ",[]{}":
                    break
                index += 1
        elif state == _AT_TOKEN and depth == 0 and char in "|>":
            return _SCAN_BLOCK
        else:
            state = _IN_PLAIN
        index += 1
    return _SCAN_OPEN if depth else _SCAN_OK


__all__ = ["normalise_text", "parse_skill_md"]
