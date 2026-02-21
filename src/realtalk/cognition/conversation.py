"""Conversation history management for multi-turn dialogue."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .llm import Message


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Manages multi-turn conversation history.

    Provides:
    - Message history with automatic truncation
    - Conversation context tracking
    - Metadata for each turn (emotion, entities, etc.)

    Future extensions:
    - Topic tracking and switching detection
    - Emotional trajectory analysis
    - Long-term memory summarization
    """

    def __init__(
        self,
        max_history_turns: int = 10,
        max_context_tokens: int = 4000,
        enable_summarization: bool = False
    ):
        self.max_history_turns = max_history_turns
        self.max_context_tokens = max_context_tokens
        self.enable_summarization = enable_summarization

        self._turns: List[ConversationTurn] = []
        self._summary: Optional[str] = None  # For future use

    def add_user_message(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a user message to the conversation."""
        turn = ConversationTurn(
            role="user",
            content=text.strip(),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._turns.append(turn)
        self._maybe_truncate()

    def add_assistant_message(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an assistant message to the conversation."""
        turn = ConversationTurn(
            role="assistant",
            content=text.strip(),
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._turns.append(turn)
        self._maybe_truncate()

    def get_messages(self, include_system: bool = True, system_prompt: Optional[str] = None) -> List[Message]:
        """Get conversation history as LLM messages.

        Returns recent turns as Message objects for LLM API.
        If history exceeds max_context_tokens, older turns are excluded.
        """
        messages = []

        # Add system prompt if provided
        if include_system and system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        # Get recent turns that fit in context window
        recent_turns = self._get_recent_turns()

        for turn in recent_turns:
            messages.append(Message(role=turn.role, content=turn.content))

        return messages

    def get_recent_context(self, n_turns: int = 3) -> str:
        """Get recent conversation context as a formatted string.

        Useful for debugging or display purposes.
        """
        recent = self._turns[-n_turns * 2:] if len(self._turns) > n_turns * 2 else self._turns
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "AI"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def _get_recent_turns(self) -> List[ConversationTurn]:
        """Get recent turns that fit within token budget."""
        # Simple estimation: ~4 chars per token for CJK, ~4 chars for English
        total_chars = 0
        max_chars = self.max_context_tokens * 4

        recent_turns = []
        for turn in reversed(self._turns):
            turn_chars = len(turn.content)
            if total_chars + turn_chars > max_chars and recent_turns:
                # Would exceed budget, stop (but keep at least one turn)
                break
            recent_turns.insert(0, turn)
            total_chars += turn_chars

        return recent_turns

    def _maybe_truncate(self) -> None:
        """Truncate old turns if exceeding max_history_turns."""
        if len(self._turns) > self.max_history_turns * 2:  # *2 because each exchange has 2 turns
            # Keep only recent turns
            excess = len(self._turns) - self.max_history_turns * 2
            if self.enable_summarization and excess > 0:
                # Future: summarize truncated turns into _summary
                pass
            self._turns = self._turns[excess:]

    def get_turn_count(self) -> int:
        """Get total number of turns."""
        return len(self._turns)

    def get_exchange_count(self) -> int:
        """Get number of complete user-assistant exchanges."""
        return len(self._turns) // 2

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
        self._summary = None

    def is_empty(self) -> bool:
        """Check if conversation is empty."""
        return len(self._turns) == 0

    def get_last_user_message(self) -> Optional[str]:
        """Get the most recent user message."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the most recent assistant message."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None
