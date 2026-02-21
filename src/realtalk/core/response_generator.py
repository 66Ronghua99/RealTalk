"""Unified response generation using streaming pipeline.

Replaces the old sequential ResponseGenerator with the new StreamingPipeline
for parallel TTS generation and lower latency.

This module maintains backward compatibility while using the new streaming
architecture internally.
"""
import asyncio
from typing import Optional, Callable, List

from ..cognition.conversation import ConversationManager
from ..cognition.llm import BaseLLM, Message
from ..cognition.streaming_pipeline import StreamingPipeline
from ..cognition.streaming_types import PipelineConfig
from ..cognition.tts import BaseTTS
from ..logging_config import setup_logger

logger = setup_logger("realtalk.response_generator")


class GenerationConfig:
    """Configuration for response generation (backward compatible)."""

    def __init__(
        self,
        system_prompt: str = (
            "你是一个自然、有亲和力的AI对话伙伴。请遵循以下原则：\n"
            "1. 回复简洁自然，像日常对话一样\n"
            "2. 适当使用口语化表达，避免过于书面\n"
            "3. 记住对话上下文，保持连贯性\n"
            "4. 回复控制在2-3句话以内，除非需要详细解释"
        ),
        enable_streaming_tts: bool = True,
        sentence_delimiters: str = r"[。！？.!?]",
        max_sentences_per_response: int = 5,
        # New streaming options
        enable_parallel_tts: bool = True,
        max_concurrent_tts: int = 4,
        fast_first_response: bool = True
    ):
        self.system_prompt = system_prompt
        self.enable_streaming_tts = enable_streaming_tts
        self.sentence_delimiters = sentence_delimiters
        self.max_sentences_per_response = max_sentences_per_response
        self.enable_parallel_tts = enable_parallel_tts
        self.max_concurrent_tts = max_concurrent_tts
        self.fast_first_response = fast_first_response

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to PipelineConfig for StreamingPipeline."""
        return PipelineConfig(
            sentence_delimiters=self.sentence_delimiters,
            max_concurrent_tasks=self.max_concurrent_tts,
            enable_parallel_tts=self.enable_parallel_tts,
            enable_fast_first_response=self.fast_first_response
        )


class ResponseGenerator:
    """Unified response generator with streaming pipeline.

    Uses StreamingPipeline internally for:
    - Fast first response (split on comma for quick TTS start)
    - Parallel TTS generation (multiple sentences concurrently)
    - Ordered audio delivery (sequence guaranteed)

    Maintains backward compatible interface for existing code.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tts: BaseTTS,
        conversation: Optional[ConversationManager] = None,
        config: Optional[GenerationConfig] = None
    ):
        self.config = config or GenerationConfig()

        # Create StreamingPipeline with proper config
        pipeline_config = self.config.to_pipeline_config()

        self._pipeline = StreamingPipeline(
            llm=llm,
            tts=tts,
            config=pipeline_config,
            conversation=conversation,
            system_prompt=self.config.system_prompt
        )

        # State tracking (delegated to pipeline)
        self._on_sentence_start: Optional[Callable[[str], None]] = None
        self._on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self._on_complete: Optional[Callable[[str], None]] = None

        # Set up pipeline callbacks
        self._pipeline.set_callbacks(
            on_sentence_start=self._handle_sentence_start,
            on_audio_chunk=self._handle_audio_chunk,
            on_complete=self._handle_complete
        )

        logger.debug("ResponseGenerator initialized with StreamingPipeline")

    def _handle_sentence_start(self, text: str) -> None:
        """Proxy sentence start callback."""
        if self._on_sentence_start:
            try:
                self._on_sentence_start(text)
            except Exception as e:
                logger.error(f"Error in sentence_start callback: {e}")

    def _handle_audio_chunk(self, audio: bytes) -> None:
        """Proxy audio chunk callback."""
        if self._on_audio_chunk:
            try:
                self._on_audio_chunk(audio)
            except Exception as e:
                logger.error(f"Error in audio_chunk callback: {e}")

    def _handle_complete(self, text: str) -> None:
        """Proxy complete callback."""
        if self._on_complete:
            try:
                self._on_complete(text)
            except Exception as e:
                logger.error(f"Error in complete callback: {e}")

    def set_callbacks(
        self,
        on_sentence_start: Optional[Callable[[str], None]] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None
    ) -> None:
        """Set callbacks for generation events."""
        self._on_sentence_start = on_sentence_start
        self._on_audio_chunk = on_audio_chunk
        self._on_complete = on_complete

    async def generate_response(
        self,
        user_text: str,
        skip_history: bool = False
    ) -> Optional[str]:
        """Generate and speak a response to user input.

        Args:
            user_text: The user's input text
            skip_history: If True, don't add to conversation history

        Returns:
            The complete response text, or None if generation was skipped/stopped
        """
        try:
            return await self._pipeline.generate_response(user_text, skip_history)
        except Exception as e:
            logger.error(f"[ResponseGenerator] Generation error: {e}")
            return None

    def stop(self) -> None:
        """Signal to stop the current generation."""
        self._pipeline.stop()
        logger.info("[ResponseGenerator] Stop requested")

    def is_generating(self) -> bool:
        """Check if currently generating a response."""
        return self._pipeline.is_generating()

    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history."""
        return self._pipeline.get_conversation_history()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._pipeline.clear_history()
        logger.info("[ResponseGenerator] Conversation history cleared")

    async def close(self) -> None:
        """Close and cleanup resources."""
        await self._pipeline.close()
