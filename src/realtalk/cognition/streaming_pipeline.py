"""Streaming pipeline for real-time voice response generation.

Integrates SentenceDivider + TTSTaskManager to provide a unified interface
for streaming LLM → TTS processing with low latency and ordered delivery.

Design inspired by Open-LLM-VTuber's streaming architecture:
```
LLM Token Stream
    ↓
@sentence_divider  → SentenceOutput Stream (fast first response)
    ↓
@tts_task_manager  → TTSResult Stream (parallel gen, ordered delivery)
    ↓
Audio Playback
```
"""
import asyncio
from typing import AsyncIterator, Callable, List, Optional

from ..logging_config import setup_logger
from .conversation import ConversationManager
from .llm import BaseLLM, LLMResponse, Message
from .sentence_divider import SentenceDivider
from .streaming_types import PipelineConfig, SentenceOutput
from .tts import BaseTTS, TTSResult
from .tts_task_manager import TTSTaskManager

logger = setup_logger("realtalk.streaming_pipeline")


class StreamingPipeline:
    """Unified streaming pipeline for LLM → TTS processing.

    Provides:
    - Fast first response (split first sentence at comma)
    - Parallel TTS generation (multiple sentences concurrently)
    - Ordered audio delivery (sequence guaranteed)
    - Conversation history management
    - Multiple output modes (callback-based or async iterator)

    Usage:
        pipeline = StreamingPipeline(llm, tts, config)

        # Callback mode
        pipeline.set_callbacks(
            on_sentence=lambda s: print(f"Sentence: {s}"),
            on_audio=lambda audio: play(audio)
        )
        await pipeline.generate_response("Hello!")

        # Iterator mode
        async for tts_result in pipeline.generate_stream("Hello!"):
            play(tts_result.audio)
    """

    def __init__(
        self,
        llm: BaseLLM,
        tts: BaseTTS,
        config: Optional[PipelineConfig] = None,
        conversation: Optional[ConversationManager] = None,
        system_prompt: Optional[str] = None
    ):
        self.llm = llm
        self.tts = tts
        self.config = config or PipelineConfig()
        self.conversation = conversation or ConversationManager()

        # Default system prompt
        self.system_prompt = system_prompt or (
            "你是一个自然、有亲和力的AI对话伙伴。请遵循以下原则：\n"
            "1. 回复简洁自然，像日常对话一样\n"
            "2. 适当使用口语化表达，避免过于书面\n"
            "3. 记住对话上下文，保持连贯性\n"
            "4. 回复控制在2-3句话以内，除非需要详细解释"
        )

        # Pipeline components
        self._sentence_divider = SentenceDivider(self.config)
        self._tts_manager = TTSTaskManager(tts, self.config)

        # State tracking
        self._is_generating = False
        self._should_stop = False
        self._current_tasks: List[asyncio.Task] = []

        # Callbacks
        self._on_sentence_start: Optional[Callable[[str], None]] = None
        self._on_audio_chunk: Optional[Callable[[bytes], None]] = None
        self._on_complete: Optional[Callable[[str], None]] = None

        logger.debug(f"StreamingPipeline initialized: {self.config}")

    def set_callbacks(
        self,
        on_sentence_start: Optional[Callable[[str], None]] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None
    ) -> None:
        """Set callbacks for pipeline events.

        Args:
            on_sentence_start: Called when a new sentence is detected
            on_audio_chunk: Called when audio data is ready
            on_complete: Called when generation is complete
        """
        self._on_sentence_start = on_sentence_start
        self._on_audio_chunk = on_audio_chunk
        self._on_complete = on_complete

    async def generate_response(
        self,
        user_text: str,
        skip_history: bool = False
    ) -> Optional[str]:
        """Generate response using callbacks.

        This is the callback-based interface suitable for simple integrations.

        Args:
            user_text: User input text
            skip_history: If True, don't add to conversation history

        Returns:
            Complete response text or None if failed/stopped
        """
        full_text = ""

        async for result in self.generate_stream(user_text, skip_history):
            # Collect full text from sentences
            if hasattr(result, 'text'):
                full_text = result.text

        return full_text if full_text else None

    async def generate_stream(
        self,
        user_text: str,
        skip_history: bool = False
    ) -> AsyncIterator[TTSResult]:
        """Generate response as async iterator of TTSResult.

        This is the streaming interface that yields audio as soon as available.

        Args:
            user_text: User input text
            skip_history: If True, don't add to conversation history

        Yields:
            TTSResult objects with audio data in sequence order
        """
        if self._is_generating:
            logger.warning("Already generating response, skipping")
            return

        self._is_generating = True
        self._should_stop = False

        try:
            async for result in self._do_generate_stream(user_text, skip_history):
                if self._should_stop:
                    break
                yield result
        finally:
            self._is_generating = False
            self._should_stop = False

    async def _do_generate_stream(
        self,
        user_text: str,
        skip_history: bool
    ) -> AsyncIterator[TTSResult]:
        """Internal generation logic."""
        logger.info(f"[StreamingPipeline] Generating response for: '{user_text[:50]}...'")

        # Add user message to history
        if not skip_history:
            self.conversation.add_user_message(user_text)

        # Get messages with history
        messages = self.conversation.get_messages(
            include_system=True,
            system_prompt=self.system_prompt
        )

        # Track full response for history
        full_response = ""

        try:
            # Create LLM stream
            llm_stream = self.llm.stream_chat(
                messages,
                system_prompt=None  # Already included in messages
            )

            # Chain: LLM tokens → Sentences → TTS audio
            sentence_stream = self._sentence_divider.stream_sentences(
                self._llm_to_text_stream(llm_stream),
                is_complete_stream=False  # stream_chat yields deltas
            )

            # Process through TTS manager with ordered delivery
            async for tts_result in self._tts_manager.process_sentences(sentence_stream):
                if self._should_stop:
                    break

                # Update full response text
                if tts_result.text:
                    full_response = tts_result.text

                # Notify callbacks
                if self._on_audio_chunk and tts_result.audio:
                    self._on_audio_chunk(tts_result.audio)

                yield tts_result

            # Save to history
            if full_response and not skip_history:
                self.conversation.add_assistant_message(full_response)

            # Notify completion
            if self._on_complete:
                self._on_complete(full_response)

            logger.info(f"[StreamingPipeline] Complete: '{full_response[:80]}...'")

        except asyncio.CancelledError:
            logger.debug("[StreamingPipeline] Generation cancelled")
            raise
        except Exception as e:
            logger.error(f"[StreamingPipeline] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _llm_to_text_stream(
        self,
        llm_stream: AsyncIterator[LLMResponse]
    ) -> AsyncIterator[str]:
        """Convert LLM response stream to text stream.

        Accumulates content to build full response text.
        """
        async for response in llm_stream:
            if self._should_stop:
                break

            content = response.content
            if not content:
                continue

            # Yield the delta content for sentence processing
            yield content

    def stop(self) -> None:
        """Signal to stop the current generation."""
        self._should_stop = True
        self._tts_manager.cancel()
        logger.info("[StreamingPipeline] Stop requested")

    def is_generating(self) -> bool:
        """Check if currently generating a response."""
        return self._is_generating

    def get_conversation_history(self) -> List[Message]:
        """Get current conversation history."""
        return self.conversation.get_messages(include_system=False)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation.clear()
        logger.info("[StreamingPipeline] Conversation history cleared")

    async def close(self) -> None:
        """Close the pipeline and cleanup resources."""
        self.stop()
        await self.llm.close()
        await self.tts.close()
        logger.info("[StreamingPipeline] Closed")


class StreamingPipelineLegacy:
    """Legacy sequential pipeline for comparison/testing.

    This implements the old behavior where TTS is processed sequentially
    (one sentence at a time) rather than in parallel.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tts: BaseTTS,
        config: Optional[PipelineConfig] = None,
        conversation: Optional[ConversationManager] = None
    ):
        self.llm = llm
        self.tts = tts
        self.config = (config or PipelineConfig()).sequential()
        self.conversation = conversation or ConversationManager()

        # Use sequential config
        self._pipeline = StreamingPipeline(
            llm=llm,
            tts=tts,
            config=self.config,
            conversation=self.conversation
        )

    async def generate_response(self, user_text: str) -> Optional[str]:
        """Generate response sequentially."""
        return await self._pipeline.generate_response(user_text)

    async def generate_stream(self, user_text: str) -> AsyncIterator[TTSResult]:
        """Generate response as stream (sequential)."""
        async for result in self._pipeline.generate_stream(user_text):
            yield result

    def stop(self) -> None:
        """Stop generation."""
        self._pipeline.stop()

    def is_generating(self) -> bool:
        """Check if generating."""
        return self._pipeline.is_generating()


# Factory function
async def create_streaming_pipeline(
    llm: Optional[BaseLLM] = None,
    tts: Optional[BaseTTS] = None,
    config: Optional[PipelineConfig] = None,
    fast_response: bool = True
) -> StreamingPipeline:
    """Factory to create a configured StreamingPipeline.

    Args:
        llm: LLM instance (created from config if None)
        tts: TTS instance (created from config if None)
        config: Pipeline config (default if None)
        fast_response: If True, use fast response config

    Returns:
        Configured StreamingPipeline instance
    """
    from .llm import create_llm
    from .tts import create_tts

    llm = llm or await create_llm()
    tts = tts or await create_tts()

    if config is None and fast_response:
        config = PipelineConfig.fast_response()
    elif config is None:
        config = PipelineConfig()

    return StreamingPipeline(llm=llm, tts=tts, config=config)
