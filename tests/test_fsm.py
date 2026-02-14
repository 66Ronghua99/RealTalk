"""Tests for orchestration.fsm."""
import pytest

from realtalk.orchestration.fsm import (
    Event,
    FiniteStateMachine,
    State,
    create_default_fsm,
)


class TestFiniteStateMachine:
    """Test FiniteStateMachine."""

    @pytest.fixture
    def fsm(self):
        """Create an FSM instance."""
        return create_default_fsm()

    @pytest.mark.asyncio
    async def test_initial_state(self, fsm):
        """Test initial state."""
        assert fsm.current_state == State.IDLE

    @pytest.mark.asyncio
    async def test_transition_idle_to_listening(self, fsm):
        """Test IDLE -> LISTENING transition."""
        await fsm.start()
        result = await fsm.transition(Event.USER_START_SPEAKING)

        assert result is True
        assert fsm.current_state == State.LISTENING

    @pytest.mark.asyncio
    async def test_invalid_transition(self, fsm):
        """Test invalid transition."""
        await fsm.start()

        # IDLE -> TTS_COMPLETE is not valid
        result = await fsm.transition(Event.TTS_COMPLETE)

        assert result is False
        assert fsm.current_state == State.IDLE

    @pytest.mark.asyncio
    async def test_state_handler(self, fsm):
        """Test state handler is called."""
        handler_called = False

        async def on_listening():
            nonlocal handler_called
            handler_called = True

        fsm.add_state_handler(State.LISTENING, on_listening)
        await fsm.start()
        await fsm.transition(Event.USER_START_SPEAKING)

        assert handler_called is True


class TestFSMEvents:
    """Test FSM event handling."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a full conversation flow."""
        fsm = create_default_fsm()
        await fsm.start()

        # User starts speaking -> LISTENING
        await fsm.transition(Event.USER_START_SPEAKING)
        assert fsm.current_state == State.LISTENING

        # Gatekeeper decides to reply -> PROCESSING
        await fsm.transition(Event.GATEKEEPER_DECISION)
        assert fsm.current_state == State.PROCESSING

        # LLM responds -> SPEAKING
        await fsm.transition(Event.LLM_RESPONSE)
        assert fsm.current_state == State.SPEAKING

        # TTS completes -> LISTENING
        await fsm.transition(Event.TTS_COMPLETE)
        assert fsm.current_state == State.LISTENING

    @pytest.mark.asyncio
    async def test_interrupt_flow(self):
        """Test interrupt flow."""
        fsm = create_default_fsm()
        await fsm.start()

        # Start speaking
        await fsm.transition(Event.USER_START_SPEAKING)
        assert fsm.current_state == State.LISTENING

        # LLM responds -> SPEAKING
        await fsm.transition(Event.GATEKEEPER_DECISION)
        await fsm.transition(Event.LLM_RESPONSE)
        assert fsm.current_state == State.SPEAKING

        # User interrupts -> INTERRUPTED
        await fsm.transition(Event.USER_INTERRUPT)
        assert fsm.current_state == State.INTERRUPTED

        # User speaks again -> LISTENING
        await fsm.transition(Event.USER_START_SPEAKING)
        assert fsm.current_state == State.LISTENING
