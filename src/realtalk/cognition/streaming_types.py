"""Core data types for streaming architecture.

Defines the data structures used in the streaming pipeline:
- SentenceOutput: Represents a complete sentence from LLM stream
- TTSTask: Wrapper for TTS generation task with sequence tracking
"""
from dataclasses import dataclass, field
from typing import Any, Awaitable, Optional


@dataclass
class SentenceOutput:
    """Sentence-level output from LLM stream.

    Represents a complete sentence ready for TTS processing.
    Maintains sequence ordering for correct playback.

    Attributes:
        sequence_number: Monotonically increasing sequence ID (0-indexed)
        text: Original text from LLM
        display_text: Text optimized for display (may contain formatting)
        tts_text: Text optimized for TTS (may have emojis/urls removed)
        is_first: Whether this is the first sentence (for fast-response optimization)
        is_final: Whether this is the last sentence of the response
        metadata: Optional metadata (emotion tags, timing info, etc.)
    """
    sequence_number: int
    text: str
    display_text: Optional[str] = None
    tts_text: Optional[str] = None
    is_first: bool = False
    is_final: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields if not provided."""
        if self.display_text is None:
            self.display_text = self.text
        if self.tts_text is None:
            self.tts_text = self._prepare_tts_text(self.text)

    @staticmethod
    def _prepare_tts_text(text: str) -> str:
        """Prepare text for TTS by removing problematic characters.

        Removes emojis, URLs, and other characters that don't translate well to speech.
        """
        import re

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove markdown links, keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Remove emoji (basic range)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())

        return text.strip()

    @property
    def effective_text(self) -> str:
        """Get the text to use for TTS (defaults to tts_text)."""
        return self.tts_text or self.text


@dataclass
class TTSTask:
    """TTS task wrapper for parallel generation.

    Wraps an async TTS generation task with its associated metadata
    for ordered delivery management.

    Attributes:
        sequence_number: Sequence ID matching the SentenceOutput
        text: Text to synthesize
        task: The async task generating audio
        is_started: Whether the TTS request has been initiated
        is_completed: Whether the TTS generation is complete
        audio_data: Resulting audio data (None until complete)
        error: Any error that occurred during generation
    """
    sequence_number: int
    text: str
    task: Optional[Any] = None  # asyncio.Task or coroutine
    is_started: bool = False
    is_completed: bool = False
    audio_data: Optional[bytes] = None
    error: Optional[Exception] = None

    def __repr__(self) -> str:
        return (f"TTSTask(seq={self.sequence_number}, "
                f"started={self.is_started}, "
                f"completed={self.is_completed}, "
                f"text='{self.text[:30]}...' if len(self.text) > 30 else '{self.text}')")


@dataclass
class PipelineConfig:
    """Configuration for the streaming pipeline.

    Attributes:
        # Sentence Divider settings
        min_first_sentence_length: Minimum chars before fast-split on comma
        max_first_sentence_length: Maximum chars for first sentence (forced split)
        sentence_delimiters: Regex pattern for sentence boundaries
        fast_split_chars: Characters to split on for fast first response
        buffer_timeout_ms: Max time to wait for sentence completion

        # TTSTaskManager settings
        max_concurrent_tasks: Maximum parallel TTS requests
        ordered_delivery: Whether to enforce ordered delivery (vs. fastest-first)
        task_timeout_seconds: Timeout for individual TTS requests

        # Pipeline settings
        enable_fast_first_response: Enable comma-based first sentence split
        enable_parallel_tts: Enable parallel TTS generation
    """
    # Sentence Divider defaults
    min_first_sentence_length: int = 10
    max_first_sentence_length: int = 50
    sentence_delimiters: str = r"[。！？.!?]"
    fast_split_chars: str = "，、,;"  # Comma and similar for fast split
    buffer_timeout_ms: float = 5000.0

    # TTS Task Manager defaults
    max_concurrent_tasks: int = 4
    ordered_delivery: bool = True
    task_timeout_seconds: float = 30.0

    # Pipeline defaults
    enable_fast_first_response: bool = True
    enable_parallel_tts: bool = True

    @classmethod
    def fast_response(cls) -> "PipelineConfig":
        """Create config optimized for fast first response."""
        return cls(
            min_first_sentence_length=5,
            max_first_sentence_length=30,
            enable_fast_first_response=True,
            enable_parallel_tts=True
        )

    @classmethod
    def quality_focused(cls) -> "PipelineConfig":
        """Create config optimized for quality (complete sentences)."""
        return cls(
            min_first_sentence_length=20,
            max_first_sentence_length=100,
            enable_fast_first_response=False,
            enable_parallel_tts=True
        )

    @classmethod
    def sequential(cls) -> "PipelineConfig":
        """Create config for sequential processing (no parallelism)."""
        return cls(
            max_concurrent_tasks=1,
            ordered_delivery=True,
            enable_parallel_tts=False
        )
