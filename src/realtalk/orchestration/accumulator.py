"""Context Accumulator for the Orchestration Layer."""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ..logging_config import setup_logger

logger = setup_logger("realtalk.accumulator")


@dataclass
class AccumulatedSegment:
    """A segment of accumulated user speech."""
    text: str
    timestamp: datetime
    energy: float = 0.0
    is_interrupt: bool = False


class ContextAccumulator:
    """Accumulates user speech segments for context.

    This handles the "Content Accumulation" feature where:
    - When VAD detects silence but intent classifier判断用户情绪激动
    - AI keeps silent, sending backchannel cues
    - Multiple speech segments are accumulated until timeout or explicit trigger
    """

    def __init__(
        self,
        max_segments: int = 10,
        max_duration_ms: int = 10000,
        auto_flush_timeout_ms: int = 2000
    ):
        self.max_segments = max_segments
        self.max_duration_ms = max_duration_ms
        self.auto_flush_timeout_ms = auto_flush_timeout_ms
        self._segments: List[AccumulatedSegment] = []
        self._start_time: Optional[datetime] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_event = asyncio.Event()
        self._callbacks: List[callable] = []

    def add_segment(
        self,
        text: str,
        energy: float = 0.0,
        is_interrupt: bool = False
    ) -> None:
        """Add a speech segment to the accumulator."""
        now = datetime.now()

        if self._start_time is None:
            self._start_time = now

        segment = AccumulatedSegment(
            text=text,
            timestamp=now,
            energy=energy,
            is_interrupt=is_interrupt
        )
        self._segments.append(segment)

        logger.info(f"Accumulated segment: {text[:50]}... (total: {len(self._segments)})")

        # Reset flush timer
        self._reset_flush_timer()

        # Check if we should auto-flush based on segments
        if len(self._segments) >= self.max_segments:
            self.flush()

    def _reset_flush_timer(self) -> None:
        """Reset the auto-flush timer."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

        # Only create task if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            self._flush_task = loop.create_task(self._auto_flush())
        except RuntimeError:
            # No running event loop - skip timer in sync context
            pass

    async def _auto_flush(self) -> None:
        """Auto-flush after timeout."""
        try:
            await asyncio.sleep(self.auto_flush_timeout_ms / 1000)
            if self._segments:
                logger.info("Auto-flushing accumulated context")
                self.flush()
        except asyncio.CancelledError:
            pass

    def flush(self) -> str:
        """Flush all accumulated segments and return combined text."""
        if not self._segments:
            return ""

        # Combine all segments
        combined_text = " ".join(seg.text for seg in self._segments)

        logger.info(f"Flushing {len(self._segments)} segments: {combined_text[:100]}...")

        # Notify callbacks
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(combined_text))
            else:
                callback(combined_text)

        # Clear
        self._segments.clear()
        self._start_time = None

        return combined_text

    def get_combined_text(self) -> str:
        """Get combined text without flushing."""
        return " ".join(seg.text for seg in self._segments)

    def get_segments(self) -> List[AccumulatedSegment]:
        """Get all accumulated segments."""
        return self._segments.copy()

    def get_interrupt_segments(self) -> List[AccumulatedSegment]:
        """Get segments marked as interrupts."""
        return [seg for seg in self._segments if seg.is_interrupt]

    def get_regular_segments(self) -> List[AccumulatedSegment]:
        """Get non-interrupt segments."""
        return [seg for seg in self._segments if not seg.is_interrupt]

    def is_empty(self) -> bool:
        """Check if accumulator is empty."""
        return len(self._segments) == 0

    def count(self) -> int:
        """Get number of segments."""
        return len(self._segments)

    def on_flush(self, callback: callable) -> None:
        """Register a callback to be called on flush."""
        self._callbacks.append(callback)

    async def wait_for_flush(self) -> str:
        """Wait for a flush to occur."""
        await self._flush_event.wait()
        self._flush_event.clear()
        return self.get_combined_text()

    def clear(self) -> None:
        """Clear all accumulated segments."""
        self._segments.clear()
        self._start_time = None
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        logger.info("Accumulator cleared")

    def __len__(self) -> int:
        return len(self._segments)

    def __bool__(self) -> bool:
        return len(self._segments) > 0


class StubbornnessController:
    """Controls AI's stubbornness level for interrupt handling.

    Stubbornness levels:
    - 0-30: Polite mode - immediately stop TTS on user speech
    - 70-100: Argument/Stubborn mode - ignore short interruptions
    """

    def __init__(self, level: int = 50):
        self._level = max(0, min(100, level))

    @property
    def level(self) -> int:
        """Get current stubbornness level."""
        return self._level

    @level.setter
    def level(self, value: int) -> None:
        """Set stubbornness level (0-100)."""
        self._level = max(0, min(100, value))
        logger.info(f"Stubbornness level set to {self._level}")

    def should_ignore_interrupt(self, interrupt_duration_ms: int, interrupt_text: str) -> bool:
        """Determine if an interrupt should be ignored based on stubbornness.

        Args:
            interrupt_duration_ms: How long the user spoke during AI's TTS
            interrupt_text: What the user said

        Returns:
            True if the interrupt should be ignored
        """
        if self._level < 30:
            # Polite mode - never ignore
            return False

        if self._level < 70:
            # Medium - ignore very short non-command interruptions
            if interrupt_duration_ms < 500:
                return True
            return False

        # High stubbornness (70-100)
        # Ignore short interruptions (< 1 second) unless strong command
        strong_commands = ["闭嘴", "停下", "别说了", "stop", "quiet", "shut up"]
        has_command = any(cmd in interrupt_text for cmd in strong_commands)

        if has_command:
            return False  # Don't ignore strong commands

        if interrupt_duration_ms < 1000:
            return True  # Ignore short interruptions

        return False

    def get_counter_argument_prompt(self, interrupt_text: str) -> str:
        """Get a prompt for generating counter-argument based on interrupt.

        This is used when stubbornness is high and AI continues speaking,
        then generates a response to the interruption after finishing.
        """
        if self._level < 50:
            return ""

        return (
            f"User interrupted with: '{interrupt_text}'. "
            f"Generate a brief counter-response to address this interruption. "
            f"Keep it short and natural."
        )

    def adjust_level(self, delta: int) -> None:
        """Adjust stubbornness level by delta."""
        self.level = self._level + delta
