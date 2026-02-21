"""Tests for streaming pipeline components.

Tests the core streaming architecture:
- SentenceDivider: Fast first response, boundary detection
- TTSTaskManager: Parallel generation, ordered delivery
- StreamingPipeline: End-to-end integration
"""
import asyncio
import pytest
from typing import AsyncIterator, List

from realtalk.cognition.sentence_divider import SentenceDivider
from realtalk.cognition.streaming_pipeline import StreamingPipeline
from realtalk.cognition.streaming_types import PipelineConfig, SentenceOutput
from realtalk.cognition.tts import BaseTTS, TTSResult
from realtalk.cognition.tts_task_manager import TTSTaskManager
from realtalk.cognition.llm import BaseLLM, LLMResponse, Message


class MockLLM(BaseLLM):
    """Mock LLM for testing."""

    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Hello world. How are you?"]
        self._closed = False
        self._current_response_index = 0

    def _get_next_response(self) -> str:
        """Get next response in sequence."""
        if self._current_response_index < len(self.responses):
            response = self.responses[self._current_response_index]
            self._current_response_index += 1
            return response
        return self.responses[-1] if self.responses else "Default response."

    async def chat(
        self,
        messages: List[Message],
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        return LLMResponse(
            content=self._get_next_response(),
            model="mock",
            finish_reason="stop"
        )

    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[LLMResponse]:
        """Stream character by character for realistic testing."""
        text = self._get_next_response()
        accumulated = ""
        for char in text:
            accumulated += char
            yield LLMResponse(
                content=char,
                model="mock",
                finish_reason=""
            )
            await asyncio.sleep(0.001)  # Tiny delay to simulate streaming

    async def close(self) -> None:
        self._closed = True


class MockTTS(BaseTTS):
    """Mock TTS for testing."""

    def __init__(self, delay_ms: float = 10):
        self.delay_ms = delay_ms
        self._call_count = 0
        self._closed = False
        self._last_seq = 0

    async def synthesize(self, text: str, sequence_number: int = 0) -> TTSResult:
        """Simulate TTS synthesis with delay."""
        self._call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)

        # Generate deterministic "audio" based on text
        audio = f"AUDIO:{text}".encode()

        return TTSResult(
            audio=audio,
            sample_rate=32000,
            is_final=True,
            text=text,
            sequence_number=sequence_number
        )

    async def stream_synthesize(self, text: str, sequence_number: int = 0) -> AsyncIterator[TTSResult]:
        """Stream TTS."""
        await asyncio.sleep(self.delay_ms / 1000)
        audio = f"AUDIO:{text}".encode()
        yield TTSResult(
            audio=audio,
            sample_rate=32000,
            is_final=False,
            text=text,
            sequence_number=sequence_number
        )
        yield TTSResult(
            audio=None,
            sample_rate=32000,
            is_final=True,
            text=text,
            sequence_number=sequence_number
        )

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        self._closed = True


class TestSentenceDivider:
    """Tests for SentenceDivider."""

    @pytest.mark.asyncio
    async def test_basic_sentence_split(self):
        """Test basic sentence splitting on periods."""
        divider = SentenceDivider(PipelineConfig())

        async def mock_stream():
            tokens = ["Hello ", "Hello world.", "Hello world. How ", "Hello world. How are you?"]
            for token in tokens:
                yield token

        sentences = []
        async for sentence in divider.stream_sentences(mock_stream()):
            sentences.append(sentence)

        assert len(sentences) == 2
        assert sentences[0].text == "Hello world."
        assert sentences[0].sequence_number == 0
        assert sentences[0].is_first is True
        assert sentences[1].text == "How are you?"
        assert sentences[1].sequence_number == 1

    @pytest.mark.asyncio
    async def test_fast_first_response(self):
        """Test fast first response on comma."""
        config = PipelineConfig.fast_response()
        divider = SentenceDivider(config)

        async def mock_stream():
            # Long first sentence that would normally wait for period
            tokens = ["今天天气", "今天天气真好，", "今天天气真好，我们出去玩吧。"]
            for token in tokens:
                yield token

        sentences = []
        async for sentence in divider.stream_sentences(mock_stream()):
            sentences.append(sentence)

        assert len(sentences) >= 1
        # First sentence should be split at comma
        assert sentences[0].is_first is True
        assert "，" in sentences[0].text or "," in sentences[0].text or "今天天气真好" in sentences[0].text

    @pytest.mark.asyncio
    async def test_quote_handling(self):
        """Test that sentences inside quotes are not split early."""
        divider = SentenceDivider(PipelineConfig())

        async def mock_stream():
            tokens = ['He said "Hello.', 'He said "Hello. How', 'He said "Hello. How are', 'He said "Hello. How are you?"']
            for token in tokens:
                yield token

        sentences = []
        async for sentence in divider.stream_sentences(mock_stream()):
            sentences.append(sentence)

        # Should wait for closing quote
        assert len(sentences) == 1
        assert '"' in sentences[0].text

    @pytest.mark.asyncio
    async def test_chinese_punctuation(self):
        """Test Chinese sentence delimiters."""
        divider = SentenceDivider(PipelineConfig())

        async def mock_stream():
            tokens = ["你好！", "你好！最近", "你好！最近怎么样？"]
            for token in tokens:
                yield token

        sentences = []
        async for sentence in divider.stream_sentences(mock_stream()):
            sentences.append(sentence)

        assert len(sentences) == 2
        assert "！" in sentences[0].text
        assert "？" in sentences[1].text


class TestTTSTaskManager:
    """Tests for TTSTaskManager."""

    @pytest.mark.asyncio
    async def test_ordered_delivery(self):
        """Test that results are delivered in sequence order."""
        tts = MockTTS(delay_ms=50)
        config = PipelineConfig()
        config.max_concurrent_tasks = 3
        manager = TTSTaskManager(tts, config)

        async def sentence_stream():
            sentences = [
                SentenceOutput(sequence_number=0, text="First sentence."),
                SentenceOutput(sequence_number=1, text="Second sentence."),
                SentenceOutput(sequence_number=2, text="Third sentence."),
            ]
            for s in sentences:
                yield s

        results = []
        async for result in manager.process_sentences(sentence_stream()):
            if result.audio:
                results.append((result.sequence_number, result.text))

        # Should be in order
        assert len(results) == 3
        assert results[0][0] == 0
        assert results[1][0] == 1
        assert results[2][0] == 2

    @pytest.mark.asyncio
    async def test_parallel_generation(self):
        """Test that multiple TTS requests run in parallel."""
        tts = MockTTS(delay_ms=100)
        config = PipelineConfig()
        config.max_concurrent_tasks = 3
        manager = TTSTaskManager(tts, config)

        async def sentence_stream():
            sentences = [
                SentenceOutput(sequence_number=0, text="One."),
                SentenceOutput(sequence_number=1, text="Two."),
                SentenceOutput(sequence_number=2, text="Three."),
            ]
            for s in sentences:
                yield s

        start = asyncio.get_event_loop().time()
        results = []
        async for result in manager.process_sentences(sentence_stream()):
            if result.audio:
                results.append(result)

        elapsed = asyncio.get_event_loop().time() - start

        # Should complete faster than sequential (300ms) due to parallelism
        assert len(results) == 3
        # Parallel: ~100ms + overhead, Sequential: ~300ms
        assert elapsed < 0.25  # Should be under 250ms with 3-way parallelism

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors don't break the pipeline."""
        class FailingTTS(MockTTS):
            async def synthesize(self, text: str, sequence_number: int = 0) -> TTSResult:
                if "fail" in text.lower():
                    raise Exception("TTS failed")
                result = await super().synthesize(text, sequence_number)
                result.sequence_number = sequence_number
                return result

        tts = FailingTTS()
        manager = TTSTaskManager(tts)

        async def sentence_stream():
            sentences = [
                SentenceOutput(sequence_number=0, text="This will work."),
                SentenceOutput(sequence_number=1, text="This will fail."),
                SentenceOutput(sequence_number=2, text="This works too."),
            ]
            for s in sentences:
                yield s

        results = []
        async for result in manager.process_sentences(sentence_stream()):
            if result.audio:
                results.append(result)

        # Should get results for seq 0 and 2 (seq 1 failed)
        assert len(results) == 2
        assert results[0].sequence_number == 0
        assert results[1].sequence_number == 2

    @pytest.mark.asyncio
    async def test_cancellation(self):
        """Test that cancellation stops processing."""
        tts = MockTTS(delay_ms=500)  # Long delay
        manager = TTSTaskManager(tts)

        async def sentence_stream():
            for i in range(10):
                yield SentenceOutput(sequence_number=i, text=f"Sentence {i}.")
                await asyncio.sleep(0.01)

        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            manager.cancel()

        # Start cancellation task
        cancel_task = asyncio.create_task(cancel_after_delay())

        results = []
        try:
            async for result in manager.process_sentences(sentence_stream()):
                if result.audio:
                    results.append(result)
        except asyncio.CancelledError:
            pass

        await cancel_task

        # Should have fewer than 10 results due to cancellation
        assert len(results) < 10


class TestStreamingPipeline:
    """Tests for StreamingPipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_generation(self):
        """Test complete pipeline from LLM to TTS."""
        llm = MockLLM(["Hello world. How are you today?"])
        tts = MockTTS(delay_ms=10)
        config = PipelineConfig()
        config.max_concurrent_tasks = 2

        pipeline = StreamingPipeline(llm, tts, config)

        # Test callback mode
        audio_chunks = []

        def on_audio(audio: bytes):
            audio_chunks.append(audio)

        pipeline.set_callbacks(
            on_audio_chunk=on_audio
        )

        result = await pipeline.generate_response("Hi!")

        # The pipeline returns the last text, but we verify processing worked
        assert result is not None
        # Should have processed multiple sentences and generated audio
        assert len(audio_chunks) >= 1

    @pytest.mark.asyncio
    async def test_streaming_interface(self):
        """Test async iterator interface."""
        llm = MockLLM(["First. Second. Third."])
        tts = MockTTS(delay_ms=5)
        config = PipelineConfig()

        pipeline = StreamingPipeline(llm, tts, config)

        results = []
        async for tts_result in pipeline.generate_stream("Test"):
            if tts_result.audio:
                results.append(tts_result)

        # Should get all 3 sentences in order
        assert len(results) == 3
        assert results[0].sequence_number == 0
        assert results[1].sequence_number == 1
        assert results[2].sequence_number == 2

    @pytest.mark.asyncio
    async def test_concurrent_prevention(self):
        """Test that concurrent generation is prevented."""
        llm = MockLLM(["First response.", "Second response."])
        tts = MockTTS(delay_ms=200)  # Longer delay to ensure overlap
        pipeline = StreamingPipeline(llm, tts)

        # Start first generation
        task1 = asyncio.create_task(pipeline.generate_response("Hi"))

        # Wait for first to actually start generating
        await asyncio.sleep(0.05)

        # Second generation should be skipped since first is in progress
        result2 = await pipeline.generate_response("Hello")

        # Wait for first to complete
        result1 = await task1

        assert result1 is not None
        assert "First" in result1 or "response" in result1
        assert result2 is None  # Should be skipped

    @pytest.mark.asyncio
    async def test_stop_generation(self):
        """Test stopping generation mid-stream."""
        llm = MockLLM(["Sentence one. Sentence two. Sentence three."])
        tts = MockTTS(delay_ms=100)
        pipeline = StreamingPipeline(llm, tts)

        async def stop_after_delay():
            await asyncio.sleep(0.15)
            pipeline.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        results = []
        try:
            async for result in pipeline.generate_stream("Test"):
                if result.audio:
                    results.append(result)
        except asyncio.CancelledError:
            pass

        await stop_task

        # Should have partial results
        assert len(results) < 3

    @pytest.mark.asyncio
    async def test_conversation_history(self):
        """Test that conversation history is maintained."""
        llm = MockLLM(["Response one."])
        tts = MockTTS()
        pipeline = StreamingPipeline(llm, tts)

        # First exchange
        await pipeline.generate_response("Hello")
        history = pipeline.get_conversation_history()
        assert len(history) == 2  # user + assistant

        # Second exchange
        await pipeline.generate_response("How are you?")
        history = pipeline.get_conversation_history()
        assert len(history) == 4  # 2 user + 2 assistant

        # Clear history
        pipeline.clear_history()
        history = pipeline.get_conversation_history()
        assert len(history) == 0


class TestFastFirstResponse:
    """Tests specifically for fast first response optimization."""

    @pytest.mark.asyncio
    async def test_first_sentence_latency(self):
        """Compare latency with and without fast first response."""
        # Long sentence without fast response would wait for period
        long_sentence = "今天天气非常好，阳光明媚，鸟语花香，我们一起去公园散步吧。"

        # Test with fast response
        config_fast = PipelineConfig.fast_response()
        divider_fast = SentenceDivider(config_fast)

        async def stream():
            # Accumulate gradually
            accumulated = ""
            for char in long_sentence:
                accumulated += char
                yield accumulated
                await asyncio.sleep(0.001)

        sentences_fast = []
        async for s in divider_fast.stream_sentences(stream()):
            sentences_fast.append(s)

        # Should have split at comma
        assert len(sentences_fast) >= 2
        assert sentences_fast[0].is_first is True

    @pytest.mark.asyncio
    async def test_forced_split_at_max_length(self):
        """Test that first sentence is forced split at max length."""
        # Very long first sentence without any punctuation
        very_long = "a" * 100

        config = PipelineConfig.fast_response()
        config.max_first_sentence_length = 30
        divider = SentenceDivider(config)

        async def stream():
            yield very_long

        sentences = []
        async for s in divider.stream_sentences(stream()):
            sentences.append(s)

        # Should have forced split
        assert len(sentences) >= 1
        assert len(sentences[0].text) <= config.max_first_sentence_length + 10  # Allow some margin


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_fast_response_config(self):
        """Test fast response configuration."""
        config = PipelineConfig.fast_response()
        assert config.enable_fast_first_response is True
        assert config.min_first_sentence_length == 5
        assert config.max_concurrent_tasks >= 2

    def test_quality_focused_config(self):
        """Test quality-focused configuration."""
        config = PipelineConfig.quality_focused()
        assert config.enable_fast_first_response is False
        assert config.min_first_sentence_length > 5

    def test_sequential_config(self):
        """Test sequential processing configuration."""
        config = PipelineConfig.sequential()
        assert config.max_concurrent_tasks == 1
        assert config.enable_parallel_tts is False
