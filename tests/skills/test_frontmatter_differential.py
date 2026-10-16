"""Differential tests: the frontmatter reader against PyYAML (FR-004 D1, plan D-1 and D-2).

The property (plan, "Agreement property"):

* **D-1.** For every input the reader accepts, its ``name`` and
  ``description`` equal ``yaml.safe_load(frontmatter)["name"]`` and
  ``["description"]``, and both are ``str``.
* **D-2.** For every input ``yaml.safe_load`` rejects, or reads to a value
  that is not a ``str`` for either field, the reader rejects.

Scope, stated rather than implied. Keys other than ``name`` and
``description`` are tolerated and NOT interpreted (D7), so a YAML error
confined to such a key's value (``metadata: a: b``) is accepted by the reader
while ``yaml.safe_load`` raises. D-2's corpus therefore holds only inputs
whose YAML problem is in ``name``, ``description`` or the top-level
structure, and D-3 checks such accepted inputs against the document reduced
to its ``name`` and ``description`` entries. What the reader does guarantee
for uninterpreted keys is that they cannot change which lines are
``name:`` and ``description:`` (a quote or bracket left open is refused;
``test_frontmatter.py`` F6).

PyYAML is YAML 1.1. The YAML 1.2 side (js-yaml and other loaders) is not run
here: it rests on the reader also refusing every YAML 1.2 implicit type
(``1e5``, ``0o17``, ``true``) and the YAML 1.1 specification's ``y``/``n``
booleans, by reading ``_frontmatter.py``'s pattern constants.

PyYAML is a dev-only dependency (``pyproject.toml``, the ``dev`` extra); the
SDK never imports it.
"""

from __future__ import annotations

import random
import re
from typing import Any

import pytest
import yaml

from fifty_agent_sdk.skills._frontmatter import _FrontmatterError, parse_skill_md


def _ours(frontmatter: str) -> tuple[str, str] | None:
    """The reader's ``(name, description)``, or ``None`` when it refuses."""
    try:
        parsed = parse_skill_md(f"---\n{frontmatter}---\nBody\n")
    except _FrontmatterError:
        return None
    assert parsed.frontmatter == frontmatter
    return parsed.name, parsed.description


def _theirs(frontmatter: str) -> tuple[Any, Any] | None:
    """PyYAML's ``(name, description)``, or ``None`` when it raises or the document is not a mapping."""
    try:
        loaded = yaml.safe_load(frontmatter)
    except Exception:
        return None
    if not isinstance(loaded, dict) or "name" not in loaded or "description" not in loaded:
        return None
    return loaded["name"], loaded["description"]


# ---------------------------------------------------------------------------
# D-1: accepted inputs agree with PyYAML
# ---------------------------------------------------------------------------

_TEXT_VALUES: list[str] = [
    "Fill in PDF forms",
    "مرحبا بالعالم، املأ النماذج",
    "Ship it 🚀 with emoji 😀",
    "façade and naïve café",
    "C# and C++ and a#b",
    "see https://example.com/a:b for details",
    "100% done, (mostly) [really]",
    "quotes ' and \" inside",
    "dash - in the middle",
    "trailing question?",
    "x\u00a0nbsp and \u200b zero-width",
    "under_score and back\\slash",
]
"""Values every style below can carry (no ``: ``, no `` #``, no leading indicator)."""


def _single(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _double(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _styles(value: str) -> list[str]:
    """``description`` written in each accepted style, all reading as ``value`` (or ``value + "\\n"``)."""
    return [
        f"description: {value}\n",
        f"description: {_single(value)}\n",
        f"description: {_double(value)}\n",
        f"description: |-\n  {value}\n",
        f"description: >\n  {value}\n",
        f"description: {value}\n  {value}\n",
    ]


_CONTEXTS: list[str] = [
    "name: pdf-forms\n{description}",
    "# leading comment\nname: pdf-forms # trailing\n\n{description}",
    "name: pdf-forms\nlicense: MIT\nmetadata:\n  author: me\n  tags: [a, b]\n{description}"
    "allowed-tools:\n  - Bash\n  - mcp__x__y\n",
    "{description}name: 'pdf-forms'\nallowed-tools: Bash(git:*) Read\n",
]

_D1_CORPUS: list[str] = [
    # Each value in every style; the surrounding context rotates.
    _CONTEXTS[(index + position) % len(_CONTEXTS)].format(description=style)
    for index, value in enumerate(_TEXT_VALUES)
    for position, style in enumerate(_styles(value))
] + [
    # Escapes, blocks and folding with more than one line.
    'name: pdf\ndescription: "tab\\there, nl\\n, \\x41\\u00e9\\U0001F600, del \\x7f"\n',
    "name: pdf\ndescription: |\n  one\n    two\n\n  three\n",
    "name: pdf\ndescription: |+\n  one\n\n\nlicense: x\n",
    "name: pdf\ndescription: >-\n  one\n  two\n\n  three\n",
    "name: pdf\ndescription: >\n\n  after a blank\n",
    "name: pdf\ndescription: one\n  two\n  three # c\n",
    "name: -pdf\ndescription: ?maybe\n",
    "name: '123'\ndescription: \"no\"\n",
    "name: pdf\ndescription: '#not a comment'\n",
    "name: pdf\ndescription: |\n  # literal hash\n  ---\n",
]


def test_d1_corpus_is_large_and_every_input_is_accepted() -> None:
    """D-1: the corpus holds at least 60 inputs, and the reader accepts each (a vacuity guard)."""
    assert len(_D1_CORPUS) >= 60
    refused = [frontmatter for frontmatter in _D1_CORPUS if _ours(frontmatter) is None]
    assert refused == []


@pytest.mark.parametrize("frontmatter", _D1_CORPUS)
def test_d1_accepted_values_equal_pyyaml(frontmatter: str) -> None:
    """D-1: an accepted input's two values equal PyYAML's, as ``str``."""
    ours = _ours(frontmatter)
    theirs = _theirs(frontmatter)
    assert ours is not None
    assert theirs is not None
    assert isinstance(theirs[0], str) and isinstance(theirs[1], str)
    assert ours == theirs


# ---------------------------------------------------------------------------
# D-2: inputs PyYAML rejects or types are rejected
# ---------------------------------------------------------------------------

_D2_CORPUS: list[str] = [
    *(
        f"name: {value}\ndescription: d\n"
        for value in (
            "no",
            "yes",
            "on",
            "Off",
            "~",
            "null",
            "123",
            "-1",
            "0x1f",
            "0b101",
            "017",
            "1_000",
            "1:30",
            "1.5",
            ".inf",
            ".NaN",
            "2024-01-01",
            "[a, b]",
            "{a: b}",
            "- a",
            "*undefined",
            "!!int 3",
            "=",
        )
    ),
    "name: pdf\ndescription: a: b\n",
    "name: pdf\ndescription: 'unclosed\n",
    'name: pdf\ndescription: "unclosed\n',
    'name: pdf\ndescription: "a"x\n',
    'name: pdf\ndescription: "a\\q"\n',
    'name: pdf\ndescription: "\\U00110000"\n',
    "name: pdf\ndescription:\n",
    "name: pdf\n\tdescription: d\n",
    "description: d\n",
    "name: pdf\ndescription: |\n  a\n b\n",
    "name: pdf\ndescription: d\n  e: f\n",
    "name: pdf\ndescription: a\x7fb\n",
    "name: pdf\ndescription: a\x01b\n",
    "name: pdf\ndescription: d\n- item\n",
    "- name: pdf\n- description: d\n",
]


@pytest.mark.parametrize("frontmatter", _D2_CORPUS)
def test_d2_inputs_pyyaml_rejects_or_types_are_rejected(frontmatter: str) -> None:
    """D-2: PyYAML raises or reads a non-``str`` for a field (checked), and the reader refuses."""
    theirs = _theirs(frontmatter)
    assert theirs is None or not (isinstance(theirs[0], str) and isinstance(theirs[1], str))
    assert _ours(frontmatter) is None


def test_d2_corpus_is_large() -> None:
    """D-2: at least 30 inputs."""
    assert len(_D2_CORPUS) >= 30


# ---------------------------------------------------------------------------
# D-3: seeded random documents
# ---------------------------------------------------------------------------

_ATOMS: list[str] = [
    *("a", "pdf", "x y", "a  b", "fa\u00e7ade", "\u0645\u0631\u062d\u0628\u0627", "\U0001f600"),
    *("yes", "no", "on", "~", "null", "123", "0x1f", "017", "1.5", ".inf", "2024-01-01"),
    *("<<", "=", "1e5", "0o17", "-x", "-", "?x", "? x", ":x", ": x", "a: b", "a:", "a#b"),
    *("a #b", "#c", "[x]", "{x}", "&a", "*a", "!t", "|", ">", "'q'", '"q"', "'it''s'"),
    *('"a\\nb"', '"\\x41"', '"\\u00e9"', '"\\ud800"', '"\\e"', "'unclosed", '"unclosed'),
    *("http://x.y/z", "- b", "%x", "@x", "`x", "x\u00a0y", "---", "...", "y", "TRUE"),
    *('"a" # c', "'a' x", '"a"x', "1:30", "+12", "._", "0b101", "1_000", "12-34", ".5"),
]
_KEYS: list[str] = ["name", "description", "license", "metadata", "allowed-tools", "x-y", "Name"]


def _value_lines(rng: random.Random) -> tuple[str, list[str]]:
    roll = rng.random()
    if roll < 0.5:
        return " " + rng.choice(_ATOMS), []
    if roll < 0.7:
        header = rng.choice(["|", "|-", "|+", ">", ">-", ">+", "|2", "| #c", ">#c"])
        width = rng.choice([1, 2, 4])
        lines = [
            rng.choice(
                [
                    "",
                    " " * width + rng.choice(_ATOMS),
                    " " * (width + 2) + rng.choice(_ATOMS),
                    " " * width,
                    " " * (width + 3),
                    "#c",
                    " " * width + "# c",
                ]
            )
            for _ in range(rng.randint(0, 3))
        ]
        return " " + header, lines
    if roll < 0.85:
        lines = [" " * rng.choice([1, 2]) + rng.choice(_ATOMS) for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.3:
            lines.insert(rng.randint(0, len(lines)), rng.choice(["", "  # c", "#c"]))
        return " " + rng.choice(_ATOMS), lines
    nested = [
        "  "
        + rng.choice([f"k: {rng.choice(_ATOMS)}", f"- {rng.choice(_ATOMS)}", rng.choice(_ATOMS)])
        for _ in range(rng.randint(1, 3))
    ]
    return "", nested


def _random_frontmatter(rng: random.Random) -> str:
    keys = rng.sample(_KEYS, rng.randint(1, 5))
    if rng.random() < 0.85:
        keys += [key for key in ("name", "description") if key not in keys]
    rng.shuffle(keys)
    if rng.random() < 0.1:
        keys.append(rng.choice(keys))
    lines: list[str] = []
    for key in keys:
        inline, rest = _value_lines(rng)
        lines.append(f"{key}:{inline}")
        lines.extend(rest)
        if rng.random() < 0.1:
            lines.append(rng.choice(["", "# top", "  # in"]))
    return "".join(f"{line}\n" for line in lines)


_TOP_LEVEL_KEY = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_-]*):")


def _known_entries_only(frontmatter: str) -> str:
    """The document with every top-level entry other than ``name`` and ``description`` removed."""
    kept: list[str] = []
    keep = False
    for line in frontmatter.split("\n")[:-1]:
        match = _TOP_LEVEL_KEY.match(line)
        if match and not line.startswith(" "):
            keep = match.group(1) in ("name", "description")
        if keep:
            kept.append(line)
    return "".join(f"{line}\n" for line in kept)


@pytest.mark.parametrize("seed", range(5))
def test_d3_random_accepted_documents_agree_with_pyyaml(seed: int) -> None:
    """D-3: on 2,000 random documents per seed, every accepted one agrees with PyYAML.

    When PyYAML loads the whole document the values must be equal. When it
    raises, the error must be confined to uninterpreted keys: PyYAML must load
    the document reduced to its ``name`` and ``description`` entries, to the
    reader's values. A floor on accepted documents keeps the test from passing
    vacuously.
    """
    rng = random.Random(seed)
    accepted = 0
    for _ in range(2000):
        frontmatter = _random_frontmatter(rng)
        ours = _ours(frontmatter)
        if ours is None:
            continue
        accepted += 1
        theirs = _theirs(frontmatter)
        if theirs is None:
            theirs = _theirs(_known_entries_only(frontmatter))
        assert theirs == ours, frontmatter
    assert accepted >= 50
