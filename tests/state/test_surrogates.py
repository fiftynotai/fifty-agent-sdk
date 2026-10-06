"""Tests for the private :func:`fifty_agent_sdk.state._surrogates.escape_message_surrogates` (BR-025).

``SqlStateStore.append`` and ``RedisStateStore.append`` pass every message
through this helper before they write it. It writes each surrogate code
point (U+D800-U+DFFF) in ``content``, ``name`` and ``tool_call_id`` as its
six-character ``\\udXXX`` escape (which cannot be told apart from the same
six characters typed), returns the message itself when nothing changes,
never changes its input, leaves ``role`` and ``tool_calls`` alone, and
passes a value that is not a ``str`` through as it is. The stores' own tests
(``test_sql.py``, ``test_redis.py``, BR-025 sections) pin what is stored;
``tests/runner/test_runner_state_surrogates.py`` pins a run through
``AgentRunner``.

Expected escapes are written as literals, never computed with
``escape_surrogates``.
"""

from __future__ import annotations

import pytest

from fifty_agent_sdk import ChatMessage, ToolCall
from fifty_agent_sdk.state._surrogates import escape_message_surrogates

_SURROGATE = chr(0xD800)
_ESCAPE = "\\ud800"  # the six characters written for chr(0xD800)
_TYPED = "a\\ud800b"  # the same six characters typed: no surrogate code point
_ARABIC = "مرحبا بالعالم"


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            ChatMessage(role="tool", content=_ARABIC, name="بحث", tool_call_id="call_1"),
            id="arabic",
        ),
        pytest.param(
            ChatMessage(role="assistant", content="smile " + chr(0x1F600), name="بوت"),
            id="astral_emoji",
        ),
        pytest.param(ChatMessage(role="user", content="del" + chr(0x7F) + "end"), id="u007f"),
        pytest.param(
            ChatMessage(role="tool", content=_TYPED, name="find\\ud800", tool_call_id="c\\udfff"),
            id="typed_escape",
        ),
        pytest.param(ChatMessage(role="system", content="plain"), id="name_and_id_none"),
        pytest.param(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(name="search", args={"q": _ARABIC}, id="call_a")],
            ),
            id="with_tool_calls",
        ),
    ],
)
def test_message_without_surrogates_is_returned_unchanged(message: ChatMessage) -> None:
    """A message with no surrogate code point in its three text fields comes back equal, each field the same object (BR-025, AC-3).

    The identity checks are per field, not on the message, so a helper that
    always returns a copy (the battery's control C2) still passes: what the
    stores write depends on the field values, not on which message object
    holds them.
    """
    result = escape_message_surrogates(message)

    assert result == message
    assert result.content is message.content
    assert result.name is message.name
    assert result.tool_call_id is message.tool_call_id
    assert result.tool_calls is message.tool_calls


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param(
            ChatMessage(role="assistant", content="a" + _SURROGATE + "b"),
            ChatMessage(role="assistant", content="a" + _ESCAPE + "b"),
            id="content",
        ),
        pytest.param(
            ChatMessage(role="tool", content="r", name="find" + _SURROGATE, tool_call_id="call_1"),
            ChatMessage(role="tool", content="r", name="find" + _ESCAPE, tool_call_id="call_1"),
            id="name",
        ),
        pytest.param(
            ChatMessage(role="tool", content="r", name="find", tool_call_id="c" + _SURROGATE),
            ChatMessage(role="tool", content="r", name="find", tool_call_id="c" + _ESCAPE),
            id="tool_call_id",
        ),
        pytest.param(
            ChatMessage(role="user", content="a" + chr(0xD83D) + chr(0xDE00) + "b"),
            ChatMessage(role="user", content="a\\ud83d\\ude00b"),
            id="two_in_a_row",
        ),
    ],
)
def test_each_text_field_holding_a_surrogate_is_escaped(
    message: ChatMessage, expected: ChatMessage
) -> None:
    """Each surrogate code point in ``content``, ``name`` or ``tool_call_id`` becomes its six-character escape; nothing else changes (BR-025).

    Two in a row (the halves of U+1F600) are escaped one by one, not
    combined into the character they pair to.
    """
    result = escape_message_surrogates(message)

    assert result == expected
    assert result.role == message.role


def test_absent_name_and_tool_call_id_stay_none() -> None:
    """``name`` and ``tool_call_id`` of ``None`` stay ``None`` when ``content`` is escaped (BR-025).

    ``escape_surrogates(None)`` raises ``AttributeError``, so the helper
    passes only a ``str`` to it.
    """
    result = escape_message_surrogates(ChatMessage(role="assistant", content="a" + _SURROGATE))

    assert result.content == "a" + _ESCAPE
    assert result.name is None
    assert result.tool_call_id is None


def test_tool_calls_are_left_as_they_are() -> None:
    """A surrogate code point in a tool call is not escaped, even when ``content`` is (BR-025 decision D3)."""
    calls = [ToolCall(name="find" + _SURROGATE, args={"q": "v" + _SURROGATE}, id="call_1")]
    message = ChatMessage(role="assistant", content="a" + _SURROGATE, tool_calls=calls)

    result = escape_message_surrogates(message)

    assert result.content == "a" + _ESCAPE
    assert result.tool_calls == calls
    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "find" + _SURROGATE
    assert result.tool_calls[0].args == {"q": "v" + _SURROGATE}


def test_input_message_is_not_changed() -> None:
    """The caller's message keeps its code points; the escape is in a copy (BR-025).

    The Runner keeps using the message it appended (its user message goes
    into the history the loop sends), so a change in place would alter the
    request.
    """
    message = ChatMessage(
        role="tool",
        content="a" + _SURROGATE,
        name="n" + _SURROGATE,
        tool_call_id="c" + _SURROGATE,
    )

    result = escape_message_surrogates(message)

    assert result.content == "a" + _ESCAPE
    assert message.content == "a" + _SURROGATE
    assert message.name == "n" + _SURROGATE
    assert message.tool_call_id == "c" + _SURROGATE


def _assigned(**fields: object) -> ChatMessage:
    """A validated message whose fields are then replaced by assignment, which skips validation."""
    message = ChatMessage(role="assistant", content="x", name="n", tool_call_id="t")
    for key, value in fields.items():
        setattr(message, key, value)
    return message


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(_assigned(content=123), id="content_int"),
        pytest.param(_assigned(content=None), id="content_none"),
        pytest.param(
            ChatMessage.model_construct(role="assistant", content=b"by"), id="content_bytes"
        ),
        pytest.param(_assigned(name=5), id="name_int"),
        pytest.param(_assigned(tool_call_id=7), id="tool_call_id_int"),
    ],
)
def test_fields_that_are_not_str_are_passed_through(message: ChatMessage) -> None:
    """A value that is not a ``str`` is returned as it is, so the stores handle it as before BR-025.

    A message holds one (other than ``None`` in ``name`` and ``tool_call_id``)
    only when validation was bypassed: ``model_construct``,
    ``model_copy(update=...)`` or assignment after validation.
    ``escape_surrogates`` would raise ``AttributeError`` on it, out of both
    stores' ``append`` (BR-025 round 2, M2); the stores' own BR-025 tests pin
    what they do with it.
    """
    result = escape_message_surrogates(message)

    assert result.content is message.content
    assert result.name is message.name
    assert result.tool_call_id is message.tool_call_id
