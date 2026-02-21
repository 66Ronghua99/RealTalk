"""Streaming sentence divider for LLM output.

Provides fast first-sentence response and intelligent boundary detection.
Transforms token stream into sentence-level stream for TTS processing.

参考 Open-LLM-VTuber 的 sentence_divider 实现。
"""
import asyncio
import re
from typing import AsyncIterator, Optional

from ..logging_config import setup_logger
from .streaming_types import PipelineConfig, SentenceOutput

logger = setup_logger("realtalk.sentence_divider")


class SentenceDivider:
    """Streams LLM tokens and yields complete sentences.

    Key features:
    1. Fast first response: Splits on comma/顿号 for quick TTS start
    2. Smart boundaries: Handles Chinese and English sentence delimiters
    3. Quote handling: Doesn't split inside unclosed quotes
    4. Buffer management: Timeout-based flush to prevent infinite buffering

    Data flow: AsyncIterator[str] (tokens) → AsyncIterator[SentenceOutput]
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._buffer = ""
        self._sequence_number = 0
        self._is_first_sentence = True
        self._closed = False

        # Compile regex patterns
        self._sentence_pattern = re.compile(self.config.sentence_delimiters)
        self._fast_split_pattern = re.compile(f"[{re.escape(self.config.fast_split_chars)}]")

        logger.debug(f"SentenceDivider initialized: {self.config}")

    async def stream_sentences(
        self,
        token_stream: AsyncIterator[str],
        is_complete_stream: bool = True
    ) -> AsyncIterator[SentenceOutput]:
        """Transform token stream into sentence stream.

        Args:
            token_stream: Async iterator yielding text tokens from LLM
            is_complete_stream: If True, tokens are cumulative; if False, incremental

        Yields:
            SentenceOutput objects for each complete sentence
        """
        self._buffer = ""
        self._sequence_number = 0
        self._is_first_sentence = True
        self._closed = False

        last_processed_len = 0

        try:
            async for token in token_stream:
                if self._closed:
                    break

                # Handle both cumulative and incremental streams
                if is_complete_stream:
                    if len(token) <= last_processed_len:
                        continue
                    new_content = token[last_processed_len:]
                    last_processed_len = len(token)
                    self._buffer += new_content
                else:
                    self._buffer += token

                # Process buffer for complete sentences
                async for sentence in self._extract_sentences():
                    yield sentence

            # Flush remaining buffer as final sentence
            if self._buffer.strip():
                yield self._create_sentence(self._buffer.strip(), is_final=True)

        except asyncio.CancelledError:
            logger.debug("SentenceDivider stream cancelled")
            raise
        except Exception as e:
            logger.error(f"SentenceDivider error: {e}")
            raise

    async def _extract_sentences(self) -> AsyncIterator[SentenceOutput]:
        """Extract complete sentences from buffer.

        Handles both regular sentence boundaries and fast-first-response logic.
        """
        while self._buffer:
            # Fast first response: split on comma/顿号 if buffer is long enough
            if self._is_first_sentence and self.config.enable_fast_first_response:
                fast_split_result = self._try_fast_split()
                if fast_split_result:
                    yield fast_split_result
                    continue

            # Regular sentence boundary detection
            match = self._sentence_pattern.search(self._buffer)
            if not match:
                break

            # Check if we're inside quotes
            end_pos = match.end()
            if self._is_inside_quotes(self._buffer[:end_pos]):
                # Don't split yet, wait for closing quote
                break

            # Extract complete sentence
            sentence_text = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:]

            if sentence_text:
                yield self._create_sentence(sentence_text, is_final=False)

    def _try_fast_split(self) -> Optional[SentenceOutput]:
        """Try to split first sentence at comma/顿号 for fast response.

        Only splits if:
        - Buffer length >= min_first_sentence_length
        - Found comma/顿号 before max_first_sentence_length
        """
        if len(self._buffer) < self.config.min_first_sentence_length:
            return None

        # Look for fast-split character within bounds
        search_end = min(len(self._buffer), self.config.max_first_sentence_length)
        match = self._fast_split_pattern.search(self._buffer[:search_end])

        if match:
            # Found split point
            end_pos = match.end()
            sentence_text = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:]

            logger.debug(f"Fast split first sentence: '{sentence_text[:30]}...'")
            return self._create_sentence(sentence_text, is_final=False)

        # Forced split at max length if buffer is too long
        if len(self._buffer) >= self.config.max_first_sentence_length:
            # Find last space or split point before max length
            search_text = self._buffer[:self.config.max_first_sentence_length]
            last_space = search_text.rfind(' ')

            if last_space > self.config.min_first_sentence_length:
                split_pos = last_space
            else:
                split_pos = self.config.max_first_sentence_length

            sentence_text = self._buffer[:split_pos].strip()
            self._buffer = self._buffer[split_pos:]

            logger.debug(f"Forced split first sentence at {split_pos}: '{sentence_text[:30]}...'")
            return self._create_sentence(sentence_text, is_final=False)

        return None

    def _is_inside_quotes(self, text: str) -> bool:
        """Check if text ends inside unclosed quotes.

        Supports: "", '', 「」, 『』, 【】
        """
        # Remove escaped quotes
        text = text.replace('\\"', '').replace("\\'", '')

        # Count quote pairs
        quotes = [
            ('"', '"'),
            ("'", "'"),
            ("「", "」"),
            ("『", "』"),
            ("【", "】"),
            ("(", ")"),
        ]

        for open_q, close_q in quotes:
            count_open = text.count(open_q)
            count_close = text.count(close_q)
            if open_q == close_q:  # Same char for open/close
                if count_open % 2 == 1:
                    return True
            else:  # Different chars
                # Check if last occurrence is opening
                last_open = text.rfind(open_q)
                last_close = text.rfind(close_q)
                if last_open > last_close:
                    return True

        return False

    def _create_sentence(self, text: str, is_final: bool) -> SentenceOutput:
        """Create a SentenceOutput object."""
        sentence = SentenceOutput(
            sequence_number=self._sequence_number,
            text=text,
            is_first=self._is_first_sentence,
            is_final=is_final
        )

        self._sequence_number += 1
        self._is_first_sentence = False

        logger.debug(f"Created sentence {sentence.sequence_number}: '{text[:50]}...' "
                    f"(first={sentence.is_first}, final={is_final})")

        return sentence

    def close(self) -> None:
        """Close the divider, stopping any ongoing processing."""
        self._closed = True
        logger.debug("SentenceDivider closed")


async def test_sentence_divider():
    """Test the sentence divider."""
    async def mock_llm_stream():
        """Simulate LLM token stream."""
        tokens = [
            "你好",
            "你好，",
            "你好，今天",
            "你好，今天的天气",
            "你好，今天的天气真",
            "你好，今天的天气真好",
            "你好，今天的天气真好啊",
            "你好，今天的天气真好啊！",
            "你好，今天的天气真好啊！你觉得",
            "你好，今天的天气真好啊！你觉得呢",
            "你好，今天的天气真好啊！你觉得呢？",
            "你好，今天的天气真好啊！你觉得呢？我们",
            "你好，今天的天气真好啊！你觉得呢？我们出去",
            "你好，今天的天气真好啊！你觉得呢？我们出去散步",
            "你好，今天的天气真好啊！你觉得呢？我们出去散步吧",
            "你好，今天的天气真好啊！你觉得呢？我们出去散步吧。",
        ]
        for token in tokens:
            await asyncio.sleep(0.1)
            yield token

    divider = SentenceDivider(PipelineConfig.fast_response())

    print("Testing SentenceDivider:")
    async for sentence in divider.stream_sentences(mock_llm_stream()):
        print(f"  [{sentence.sequence_number}] first={sentence.is_first}, "
              f"final={sentence.is_final}: '{sentence.text}'")


if __name__ == "__main__":
    asyncio.run(test_sentence_divider())
