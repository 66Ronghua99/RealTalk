"""Finite State Machine (FSM) for the Orchestration Layer."""
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..logging_config import setup_logger

logger = setup_logger("realtalk.fsm")


class State(Enum):
    """System states."""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"
    ACCUMULATING = "accumulating"
    INTERRUPTED = "interrupted"


class Event(Enum):
    """System events."""
    USER_START_SPEAKING = "user_start_speaking"
    USER_STOP_SPEAKING = "user_stop_speaking"
    USER_INTERRUPT = "user_interrupt"
    GATEKEEPER_DECISION = "gatekeeper_decision"
    ASR_RESULT = "asr_result"
    LLM_RESPONSE = "llm_response"
    TTS_COMPLETE = "tts_complete"
    TIMEOUT = "timeout"


@dataclass
class FSMTransition:
    """FSM transition definition."""
    from_state: State
    event: Event
    to_state: State
    action: Optional[Callable] = None
    guard: Optional[Callable[[], bool]] = None


class FiniteStateMachine:
    """Finite State Machine for orchestrating the voice interaction."""

    def __init__(self, initial_state: State = State.IDLE):
        self.current_state = initial_state
        self._transitions: list[FSMTransition] = []
        self._state_handlers: dict[State, Callable] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    def add_transition(
        self,
        from_state: State,
        event: Event,
        to_state: State,
        action: Optional[Callable] = None,
        guard: Optional[Callable[[], bool]] = None
    ) -> None:
        """Add a state transition."""
        transition = FSMTransition(
            from_state=from_state,
            event=event,
            to_state=to_state,
            action=action,
            guard=guard
        )
        self._transitions.append(transition)

    def add_state_handler(self, state: State, handler: Callable) -> None:
        """Add a handler for entering a state."""
        self._state_handlers[state] = handler

    def can_transition(self, event: Event) -> bool:
        """Check if there's a valid transition for this event."""
        for transition in self._transitions:
            if (
                transition.from_state == self.current_state
                and transition.event == event
            ):
                if transition.guard is None or transition.guard():
                    return True
        return False

    async def transition(self, event: Event) -> bool:
        """Process an event and transition state."""
        for transition in self._transitions:
            if (
                transition.from_state == self.current_state
                and transition.event == event
            ):
                if transition.guard is not None and not transition.guard():
                    continue

                old_state = self.current_state
                new_state = transition.to_state

                # Execute action if present
                if transition.action:
                    if asyncio.iscoroutinefunction(transition.action):
                        await transition.action()
                    else:
                        transition.action()

                # Update state
                self.current_state = new_state
                logger.info(f"FSM: {old_state.value} -> {new_state.value} via {event.value}")

                # Call state handler
                if new_state in self._state_handlers:
                    handler = self._state_handlers[new_state]
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()

                return True

        logger.warning(f"No valid transition: {event.value} from {self.current_state.value}")
        return False

    async def process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self.transition(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def emit(self, event: Event) -> None:
        """Add an event to the queue."""
        self._event_queue.put_nowait(event)

    async def emit_wait(self, event: Event) -> None:
        """Add an event and wait for it to be processed."""
        future = asyncio.Future()
        self._event_queue.put_nowait((event, future))
        await future

    async def start(self) -> None:
        """Start the FSM."""
        self._running = True
        asyncio.create_task(self.process_events())
        logger.info("FSM started")

    async def stop(self) -> None:
        """Stop the FSM."""
        self._running = False
        logger.info("FSM stopped")


def create_default_fsm() -> FiniteStateMachine:
    """Create the default FSM for RealTalk."""
    fsm = FiniteStateMachine(initial_state=State.IDLE)

    # IDLE -> LISTENING: User starts speaking
    fsm.add_transition(
        State.IDLE,
        Event.USER_START_SPEAKING,
        State.LISTENING
    )

    # LISTENING -> LISTENING: More speech detected
    fsm.add_transition(
        State.LISTENING,
        Event.USER_START_SPEAKING,
        State.LISTENING
    )

    # LISTENING -> PROCESSING: User stops, gatekeeper decides to reply
    fsm.add_transition(
        State.LISTENING,
        Event.GATEKEEPER_DECISION,
        State.PROCESSING
    )

    # LISTENING -> ACCUMULATING: User stops, gatekeeper decides to accumulate
    fsm.add_transition(
        State.LISTENING,
        Event.GATEKEEPER_DECISION,
        State.ACCUMULATING
    )

    # PROCESSING -> SPEAKING: LLM response ready
    fsm.add_transition(
        State.PROCESSING,
        Event.LLM_RESPONSE,
        State.SPEAKING
    )

    # SPEAKING -> INTERRUPTED: User interrupts
    fsm.add_transition(
        State.SPEAKING,
        Event.USER_INTERRUPT,
        State.INTERRUPTED
    )

    # SPEAKING -> LISTENING: Finished speaking
    fsm.add_transition(
        State.SPEAKING,
        Event.TTS_COMPLETE,
        State.LISTENING
    )

    # INTERRUPTED -> LISTENING: After interruption
    fsm.add_transition(
        State.INTERRUPTED,
        Event.USER_START_SPEAKING,
        State.LISTENING
    )

    # ACCUMULATING -> PROCESSING: Timeout or explicit trigger
    fsm.add_transition(
        State.ACCUMULATING,
        Event.TIMEOUT,
        State.PROCESSING
    )

    # Any state -> IDLE: System reset
    fsm.add_transition(
        State.IDLE,
        Event.TIMEOUT,
        State.IDLE
    )

    return fsm
