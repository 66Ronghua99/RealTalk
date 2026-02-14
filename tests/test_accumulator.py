"""Tests for orchestration.accumulator."""
import pytest

from realtalk.orchestration.accumulator import ContextAccumulator, StubbornnessController


class TestContextAccumulator:
    """Test ContextAccumulator."""

    @pytest.fixture
    def accumulator(self):
        """Create an accumulator instance."""
        return ContextAccumulator(
            max_segments=5,
            auto_flush_timeout_ms=100000  # Long timeout to avoid async issues in sync tests
        )

    def test_add_segment(self, accumulator):
        """Test adding segments."""
        accumulator.add_segment("Hello", energy=0.5)
        assert len(accumulator) == 1
        assert accumulator.get_combined_text() == "Hello"

    def test_add_multiple_segments(self, accumulator):
        """Test adding multiple segments."""
        accumulator.add_segment("Hello")
        accumulator.add_segment("world")
        accumulator.add_segment("today")

        assert len(accumulator) == 3
        assert accumulator.get_combined_text() == "Hello world today"

    def test_flush(self, accumulator):
        """Test flushing."""
        accumulator.add_segment("Hello")
        accumulator.add_segment("world")

        text = accumulator.flush()

        assert text == "Hello world"
        assert len(accumulator) == 0

    def test_clear(self, accumulator):
        """Test clearing."""
        accumulator.add_segment("Hello")
        accumulator.add_segment("world")

        accumulator.clear()

        assert len(accumulator) == 0
        assert accumulator.get_combined_text() == ""

    def test_interrupt_segments(self, accumulator):
        """Test tracking interrupt segments."""
        accumulator.add_segment("Hello", is_interrupt=False)
        accumulator.add_segment("Stop!", is_interrupt=True)
        accumulator.add_segment("world", is_interrupt=False)

        interrupts = accumulator.get_interrupt_segments()
        regular = accumulator.get_regular_segments()

        assert len(interrupts) == 1
        assert len(regular) == 2
        assert interrupts[0].text == "Stop!"


class TestStubbornnessController:
    """Test StubbornnessController."""

    def test_low_stubbornness_never_ignores(self):
        """Low stubbornness (0-30) never ignores interrupts."""
        controller = StubbornnessController(level=20)

        assert not controller.should_ignore_interrupt(500, "hello")
        assert not controller.should_ignore_interrupt(100, "stop")

    def test_high_stubbornness_ignores_short(self):
        """High stubbornness ignores short non-command interruptions."""
        controller = StubbornnessController(level=80)

        # Short interruption without command
        assert controller.should_ignore_interrupt(500, "hello")
        assert controller.should_ignore_interrupt(800, "嗯")

        # Strong commands are not ignored
        assert not controller.should_ignore_interrupt(500, "闭嘴")
        assert not controller.should_ignore_interrupt(500, "别说了")

    def test_high_stubbornness_long_interrupt(self):
        """High stubbornness doesn't ignore long interruptions."""
        controller = StubbornnessController(level=90)

        # Long interruption (>1 second)
        assert not controller.should_ignore_interrupt(1500, "hello")

    def test_counter_argument_prompt_low(self):
        """Low stubbornness doesn't generate counter prompts."""
        controller = StubbornnessController(level=30)

        prompt = controller.get_counter_argument_prompt("hello")
        assert prompt == ""

    def test_counter_argument_prompt_high(self):
        """High stubbornness generates counter prompts."""
        controller = StubbornnessController(level=70)

        prompt = controller.get_counter_argument_prompt("我不这么认为")
        assert "User interrupted" in prompt

    def test_adjust_level(self):
        """Test adjusting stubbornness level."""
        controller = StubbornnessController(level=50)

        controller.adjust_level(20)
        assert controller.level == 70

        controller.adjust_level(-50)
        assert controller.level == 20

        # Clamping
        controller.adjust_level(100)
        assert controller.level == 100
