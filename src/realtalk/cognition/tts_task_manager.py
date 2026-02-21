"""Parallel TTS task manager with ordered delivery.

参考 Open-LLM-VTuber 的 TTSTaskManager 实现:
- 并行生成: 多个句子同时启动TTS请求
- 有序交付: 按序列号顺序 yield 结果，即使先完成的后到
- 任务队列: 控制并发数，防止资源耗尽

Data flow: SentenceOutput → TTSTask → Ordered TTSResult Stream
"""
import asyncio
from typing import AsyncIterator, Callable, Dict, List, Optional

from ..logging_config import setup_logger
from .streaming_types import PipelineConfig, SentenceOutput, TTSTask
from .tts import BaseTTS, TTSResult

logger = setup_logger("realtalk.tts_task_manager")


class TTSTaskManager:
    """Manages parallel TTS generation with ordered delivery.

    Key features:
    1. Parallel generation: Multiple sentences processed concurrently
    2. Sequence tracking: Each task assigned incrementing sequence number
    3. Ordered delivery: Buffer out-of-order completions, yield in sequence
    4. Concurrency control: Limit max concurrent TTS requests
    5. Cancellation: Clean shutdown with pending task cancellation

    Usage:
        manager = TTSTaskManager(tts, config)
        async for tts_result in manager.process_sentences(sentence_stream):
            play_audio(tts_result.audio)
    """

    def __init__(self, tts: BaseTTS, config: Optional[PipelineConfig] = None):
        self.tts = tts
        self.config = config or PipelineConfig()

        # Task tracking
        self._tasks: Dict[int, TTSTask] = {}  # sequence_number -> TTSTask
        self._completed_tasks: Dict[int, TTSTask] = {}  # Completed but not yet yielded
        self._next_sequence_to_yield = 0
        self._max_sequence_submitted = -1

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self._running = False
        self._cancelled = False

        # Stats
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "yielded": 0,
            "errors": 0,
        }

        logger.debug(f"TTSTaskManager initialized: max_concurrent={self.config.max_concurrent_tasks}")

    async def process_sentences(
        self,
        sentence_stream: AsyncIterator[SentenceOutput]
    ) -> AsyncIterator[TTSResult]:
        """Process sentence stream with parallel TTS and ordered delivery.

        Args:
            sentence_stream: Async iterator of SentenceOutput from SentenceDivider

        Yields:
            TTSResult objects in sequence order (0, 1, 2, ...)
        """
        self._running = True
        self._cancelled = False
        self._reset_state()

        # Result queue for completed TTS tasks
        result_queue: asyncio.Queue[tuple[int, Optional[TTSResult], Optional[Exception]]] = asyncio.Queue()

        # Track active tasks
        active_tasks: dict[int, asyncio.Task] = {}
        completed_results: dict[int, TTSResult] = {}
        next_to_yield = 0

        async def generate_tts(sentence: SentenceOutput) -> None:
            """Generate TTS for a sentence and put result in queue."""
            seq = sentence.sequence_number
            text = sentence.effective_text

            logger.info(
                f"[TTS-MANAGER] task STARTED seq={seq} text={repr(text[:50])}"
            )

            try:
                if not text:
                    logger.info(f"[TTS-MANAGER] task SKIP (empty) seq={seq}")
                    await result_queue.put((seq, None, None))
                    return

                result = await asyncio.wait_for(
                    self.tts.synthesize(text),
                    timeout=self.config.task_timeout_seconds
                )
                result.sequence_number = seq
                logger.info(
                    f"[TTS-MANAGER] task DONE seq={seq} "
                    f"audio_bytes={len(result.audio) if result.audio else 0}"
                )
                await result_queue.put((seq, result, None))

            except asyncio.TimeoutError:
                logger.error(f"[TTS-MANAGER] task TIMEOUT seq={seq}")
                await result_queue.put((seq, None, TimeoutError(f"TTS timeout")))
            except Exception as e:
                logger.error(f"[TTS-MANAGER] task ERROR seq={seq}: {e}")
                await result_queue.put((seq, None, e))

        async def feed_sentences() -> None:
            """Feed sentences to worker tasks."""
            async for sentence in sentence_stream:
                if self._cancelled:
                    break

                self._stats["submitted"] += 1
                self._max_sequence_submitted = max(
                    self._max_sequence_submitted,
                    sentence.sequence_number
                )

                logger.info(
                    f"[TTS-MANAGER] sentence SUBMITTED seq={sentence.sequence_number} "
                    f"text={repr(sentence.effective_text[:50])}"
                )

                # Create task for this sentence
                task = asyncio.create_task(generate_tts(sentence))
                active_tasks[sentence.sequence_number] = task

                # Limit concurrent tasks
                while len(active_tasks) >= self.config.max_concurrent_tasks:
                    await asyncio.sleep(0.01)
                    # Clean up completed tasks
                    done = [seq for seq, t in active_tasks.items() if t.done()]
                    for seq in done:
                        try:
                            await active_tasks[seq]
                        except Exception:
                            pass
                        del active_tasks[seq]

            # Mark end of feeding
            self._running = False

        # Start feeding sentences
        feeder_task = asyncio.create_task(feed_sentences())

        try:
            # Process results as they complete
            while True:
                # Check if we're done
                if not self._running and len(active_tasks) == 0 and result_queue.empty():
                    break

                # Wait for results with timeout
                try:
                    seq, result, error = await asyncio.wait_for(
                        result_queue.get(),
                        timeout=0.1
                    )

                    if error:
                        self._stats["errors"] += 1
                        logger.warning(f"[TTS-MANAGER] result ERROR seq={seq}: {error}")
                    elif result:
                        self._stats["completed"] += 1
                        completed_results[seq] = result
                        logger.info(
                            f"[TTS-MANAGER] result BUFFERED seq={seq} "
                            f"next_to_yield={next_to_yield} "
                            f"buffered={sorted(completed_results.keys())}"
                        )

                    # Yield in order
                    while next_to_yield in completed_results:
                        to_yield = completed_results.pop(next_to_yield)
                        self._stats["yielded"] += 1
                        logger.info(
                            f"[TTS-MANAGER] YIELDING seq={next_to_yield} "
                            f"audio_bytes={len(to_yield.audio) if to_yield.audio else 0}"
                        )
                        next_to_yield += 1
                        yield to_yield

                except asyncio.TimeoutError:
                    # No results yet, continue
                    pass

                # Clean up completed tasks
                done = [seq for seq, t in active_tasks.items() if t.done()]
                for seq in done:
                    try:
                        await active_tasks[seq]
                    except Exception:
                        pass
                    del active_tasks[seq]

            # Yield any remaining results in order
            while completed_results:
                if next_to_yield in completed_results:
                    to_yield = completed_results.pop(next_to_yield)
                    self._stats["yielded"] += 1
                    next_to_yield += 1
                    yield to_yield
                else:
                    # Gap in sequence, skip
                    next_to_yield += 1
                    if next_to_yield > max(completed_results.keys(), default=-1) + 10:
                        break

        except asyncio.CancelledError:
            logger.debug("TTSTaskManager processing cancelled")
            raise
        finally:
            # Cleanup
            feeder_task.cancel()
            try:
                await feeder_task
            except asyncio.CancelledError:
                pass

            for task in active_tasks.values():
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks.values(), return_exceptions=True)

            logger.info(f"TTSTaskManager stats: {self._stats}")

    async def _sentence_worker(
        self,
        queue: asyncio.Queue[Optional[SentenceOutput]]
    ) -> None:
        """Worker that consumes sentences and generates TTS."""
        while True:
            try:
                sentence = await queue.get()

                if sentence is None:
                    # Poison pill - exit
                    queue.task_done()
                    break

                if self._cancelled:
                    queue.task_done()
                    break

                # Process sentence with concurrency limit
                async with self._semaphore:
                    await self._generate_tts(sentence)

                queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    async def _generate_tts(self, sentence: SentenceOutput) -> None:
        """Generate TTS for a single sentence."""
        seq = sentence.sequence_number
        text = sentence.effective_text

        if not text:
            logger.warning(f"Empty TTS text for sequence {seq}")
            self._mark_completed(seq, None, None)
            return

        # Create task record
        task = TTSTask(sequence_number=seq, text=text, is_started=True)
        self._tasks[seq] = task

        try:
            logger.debug(f"Starting TTS for sequence {seq}: '{text[:40]}...'")

            # Use non-streaming TTS for single sentence (more reliable)
            # For true streaming, use stream_synthesize and collect chunks
            if self.config.enable_parallel_tts:
                # Use synthesize for simplicity with parallel execution
                result = await asyncio.wait_for(
                    self.tts.synthesize(text),
                    timeout=self.config.task_timeout_seconds
                )
                # Ensure sequence number is set correctly
                result.sequence_number = seq
            else:
                # Sequential mode: stream and collect
                chunks = []
                async for chunk in self.tts.stream_synthesize(text):
                    if chunk.audio:
                        chunks.append(chunk.audio)
                audio = b"".join(chunks) if chunks else None
                result = TTSResult(
                    audio=audio,
                    sample_rate=32000,
                    is_final=True,
                    text=text,
                    sequence_number=seq
                )

            task.audio_data = result.audio
            task.is_completed = True
            self._stats["completed"] += 1

            logger.debug(f"TTS completed for sequence {seq}: "
                        f"{len(result.audio) if result.audio else 0} bytes")

            # Mark as completed for delivery
            self._mark_completed(seq, result, None)

        except asyncio.TimeoutError:
            logger.error(f"TTS timeout for sequence {seq}")
            task.error = TimeoutError(f"TTS timeout after {self.config.task_timeout_seconds}s")
            self._stats["errors"] += 1
            self._mark_completed(seq, None, task.error)
        except Exception as e:
            logger.error(f"TTS error for sequence {seq}: {e}")
            task.error = e
            self._stats["errors"] += 1
            self._mark_completed(seq, None, e)

    def _mark_completed(
        self,
        seq: int,
        result: Optional[TTSResult],
        error: Optional[Exception]
    ) -> None:
        """Mark a task as completed and store result."""
        if seq in self._tasks:
            self._tasks[seq].is_completed = True
            if error:
                self._tasks[seq].error = error

        if not error and result:
            self._completed_tasks[seq] = result
        elif error:
            # Store error marker
            self._completed_tasks[seq] = error

    async def _ordered_delivery(self) -> None:
        """Background task that ensures ordered delivery of results."""
        while self._running or self._completed_tasks:
            try:
                # Check if next sequence is available
                while self._next_sequence_to_yield in self._completed_tasks:
                    result = self._completed_tasks.pop(self._next_sequence_to_yield)
                    self._next_sequence_to_yield += 1

                    if isinstance(result, Exception):
                        logger.warning(f"Skipping failed TTS {self._next_sequence_to_yield - 1}: {result}")
                        continue

                    self._stats["yielded"] += 1

                    # Yield via queue or directly (we use direct method)
                    # This task just manages state; actual yielding happens in main loop

                await asyncio.sleep(0.01)  # Small delay to prevent busy-wait

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ordered delivery error: {e}")

    async def _drain_completed(self) -> AsyncIterator[TTSResult]:
        """Drain completed results in order."""
        while self._next_sequence_to_yield in self._completed_tasks:
            result = self._completed_tasks.pop(self._next_sequence_to_yield)
            self._next_sequence_to_yield += 1

            if isinstance(result, Exception):
                logger.warning(f"Skipping failed TTS {self._next_sequence_to_yield - 1}")
                continue

            self._stats["yielded"] += 1
            yield result

    async def _drain_all_remaining(self) -> AsyncIterator[TTSResult]:
        """Drain all remaining completed results in order."""
        # Sort by sequence number to ensure order
        sequences = sorted(self._completed_tasks.keys())

        for seq in sequences:
            if seq < self._next_sequence_to_yield:
                continue  # Already yielded

            result = self._completed_tasks[seq]
            self._next_sequence_to_yield = seq + 1

            if isinstance(result, Exception):
                logger.warning(f"Skipping failed TTS {seq}")
                continue

            self._stats["yielded"] += 1
            yield result

        self._completed_tasks.clear()

    def cancel(self) -> None:
        """Cancel all pending TTS tasks."""
        self._cancelled = True
        self._running = False
        logger.info("TTSTaskManager cancelled")

    def _reset_state(self) -> None:
        """Reset internal state for new processing session."""
        self._tasks.clear()
        self._completed_tasks.clear()
        self._next_sequence_to_yield = 0
        self._max_sequence_submitted = -1
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "yielded": 0,
            "errors": 0,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self._stats.copy()
