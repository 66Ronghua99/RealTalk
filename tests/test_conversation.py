"""Tests for conversation history management."""
import pytest

from realtalk.cognition.conversation import ConversationManager, ConversationTurn
from realtalk.cognition.llm import Message


class TestConversationManager:
    """Test conversation history management."""

    def test_add_user_message(self):
        """Test adding user messages."""
        conv = ConversationManager()
        conv.add_user_message("Hello")

        assert conv.get_turn_count() == 1
        assert conv.get_last_user_message() == "Hello"

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        conv = ConversationManager()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi there!")

        assert conv.get_turn_count() == 2
        assert conv.get_last_assistant_message() == "Hi there!"

    def test_get_messages(self):
        """Test getting messages for LLM."""
        conv = ConversationManager()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        conv.add_user_message("How are you?")

        messages = conv.get_messages(include_system=False)

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi!"
        assert messages[2].role == "user"
        assert messages[2].content == "How are you?"

    def test_get_messages_with_system_prompt(self):
        """Test getting messages with system prompt."""
        conv = ConversationManager()
        conv.add_user_message("Hello")

        messages = conv.get_messages(
            include_system=True,
            system_prompt="You are a helpful assistant."
        )

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].role == "user"

    def test_max_history_truncation(self):
        """Test that old turns are truncated."""
        conv = ConversationManager(max_history_turns=2)

        # Add 6 turns (3 exchanges)
        for i in range(3):
            conv.add_user_message(f"User {i}")
            conv.add_assistant_message(f"Assistant {i}")

        # Should keep only 2 exchanges (4 turns)
        assert conv.get_turn_count() == 4
        messages = conv.get_messages(include_system=False)
        assert messages[0].content == "User 1"  # Oldest kept
        assert messages[-1].content == "Assistant 2"  # Newest

    def test_clear(self):
        """Test clearing conversation history."""
        conv = ConversationManager()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        conv.clear()

        assert conv.is_empty()
        assert conv.get_turn_count() == 0

    def test_get_exchange_count(self):
        """Test counting complete exchanges."""
        conv = ConversationManager()

        assert conv.get_exchange_count() == 0

        conv.add_user_message("Hello")
        assert conv.get_exchange_count() == 0  # Incomplete

        conv.add_assistant_message("Hi!")
        assert conv.get_exchange_count() == 1  # Complete

        conv.add_user_message("How are you?")
        assert conv.get_exchange_count() == 1  # Incomplete again

    def test_context_window_limit(self):
        """Test that long content respects context window."""
        conv = ConversationManager(max_context_tokens=10)  # Very small for testing

        # Add a long message that would exceed token limit
        long_message = "This is a very long message that should exceed the token limit."
        conv.add_user_message(long_message)
        conv.add_assistant_message("Short reply.")
        conv.add_user_message("Another message.")

        messages = conv.get_messages(include_system=False)

        # Should only include messages that fit in context
        # The long message alone might exceed limit
        total_chars = sum(len(m.content) for m in messages)
        assert total_chars <= 10 * 4  # ~4 chars per token

    def test_is_empty(self):
        """Test empty conversation check."""
        conv = ConversationManager()
        assert conv.is_empty()

        conv.add_user_message("Hello")
        assert not conv.is_empty()

    def test_metadata(self):
        """Test adding metadata to turns."""
        conv = ConversationManager()
        conv.add_user_message("Hello", metadata={"emotion": "happy"})

        turn = conv._turns[0]
        assert turn.metadata["emotion"] == "happy"

    def test_get_recent_context(self):
        """Test getting recent context as formatted string."""
        conv = ConversationManager()
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")
        conv.add_user_message("How are you?")

        context = conv.get_recent_context(n_turns=2)

        assert "User: Hi!" not in context  # Too old
        assert "User: How are you?" in context
        assert "AI: Hi!" in context
