"""Main Orchestrator - Ties all components together."""
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from ..cognition.llm import BaseLLM, Message
from ..cognition.tts import BaseTTS
from ..config import get_config
from ..exceptions import OrchestrationError
from ..logging_config import setup_logger
from ..perception.asr import BaseASR
from ..perception.vad import BaseVAD
from ..orchestration.accumulator import ContextAccumulator, StubbornnessController
from ..orchestration.fsm import Event, FiniteStateMachine, State, create_default_fsm
from ..orchestration.gatekeeper import Action, GatekeeperInput, create_gatekeeper

logger = setup_logger("realtalk.orchestrator")


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    system_prompt: str = "You are a helpful and conversational AI assistant."
    enable_accumulation: bool = True
    enable_interrupt_handling: bool = True


class VoiceOrchestrator:
    """Main orchestrator for the RealTalk voice interaction system.

    Coordinates:
    - VAD (Voice Activity Detection)
    - ASR (Automatic Speech Recognition)
    - Gatekeeper (Intent Classification)
    - LLM (Language Model)
    - TTS (Text-to-Speech)
    - Context Accumulator
    - Stubbornness Controller
    """

    def __init__(
        self,
        vad: BaseVAD,
        asr: BaseASR,
        llm: BaseLLM,
        tts: BaseTTS,
        config: Optional[OrchestratorConfig] = None
    ):
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts

        self.config = config or OrchestratorConfig()
        cfg = get_config()

        # Core components
        self.fsm = create_default_fsm()
        self.gatekeeper = None  # Initialized in start()
        self.accumulator = ContextAccumulator() if self.config.enable_accumulation else None
        self.stubbornness = StubbornnessController(
            level=cfg.orchestration.stubbornness_level
        )

        # State tracking
        self._running = False
        self._audio_task: Optional[asyncio.Task] = None
        self._llm_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None

        # Event callbacks
        self._on_asr_result: Optional[callable] = None
        self._on_tts_audio: Optional[callable] = None
        self._on_state_change: Optional[callable] = None
        self._on_error: Optional[callable] = None

        # Current context
        self._current_text: str = ""
        self._silence_duration_ms: int = 0
        self._last_speech_time: Optional[float] = None

    async def start(self) -> None:
        """Start the orchestrator."""
        logger.info("Starting VoiceOrchestrator")
        self._running = True

        # Initialize gatekeeper
        self.gatekeeper = await create_gatekeeper()

        # Set up FSM state handlers
        self.fsm.add_state_handler(State.LISTENING, self._on_listening)
        self.fsm.add_state_handler(State.SPEAKING, self._on_speaking)
        self.fsm.add_state_handler(State.PROCESSING, self._on_processing)
        self.fsm.add_state_handler(State.ACCUMULATING, self._on_accumulating)

        # Start FSM
        await self.fsm.start()

        # Add accumulator callback
        if self.accumulator:
            self.accumulator.on_flush(self._on_accumulator_flush)

        # Set initial state
        await self.fsm.transition(Event.USER_START_SPEAKING)

        logger.info("VoiceOrchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping VoiceOrchestrator")
        self._running = False

        # Cancel running tasks
        for task in [self._audio_task, self._llm_task, self._tts_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop components
        await self.fsm.stop()
        await self.vad.close()
        await self.asr.close()
        await self.tts.close()

        logger.info("VoiceOrchestrator stopped")

    async def process_audio(self, audio_chunk: bytes) -> None:
        """Process incoming audio chunk.

        This is the main entry point for audio data from the transport layer.
        """
        if not self._running:
            return

        # VAD detection
        import numpy as np
        audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
        vad_result = await self.vad.detect(audio_array)

        # Track silence
        current_time = asyncio.get_event_loop().time()
        if self._last_speech_time is not None:
            self._silence_duration_ms = int((current_time - self._last_speech_time) * 1000)

        if vad_result.is_speech:
            self._last_speech_time = current_time
            await self.fsm.transition(Event.USER_START_SPEAKING)

            # Process ASR
            asr_result = await self.asr.recognize(audio_chunk)
            if asr_result.text:
                self._current_text = asr_result.text

                # Notify callbacks
                if self._on_asr_result:
                    self._on_asr_result(asr_result)

                # Check for interrupt if speaking
                if self.fsm.current_state == State.SPEAKING:
                    await self._handle_interrupt(asr_result.text, vad_result.confidence)
                else:
                    # Run gatekeeper decision
                    await self._run_gatekeeper(asr_result.text, vad_result.confidence)

        else:
            # No speech detected
            if self._last_speech_time is not None:
                silence_duration = int((current_time - self._last_speech_time) * 1000)
                if silence_duration > 100:
                    await self.fsm.transition(Event.USER_STOP_SPEAKING)

    async def _handle_interrupt(self, text: str, energy: float) -> None:
        """Handle user interruption during AI speech."""
        if not self.config.enable_interrupt_handling:
            await self.fsm.transition(Event.USER_INTERRUPT)
            await self.tts.stop()
            return

        # Check stubbornness
        should_ignore = self.stubbornness.should_ignore_interrupt(
            interrupt_duration_ms=500,  # Approximate
            interrupt_text=text
        )

        if should_ignore:
            # Accumulate the interrupt for later response
            if self.accumulator:
                self.accumulator.add_segment(text, energy, is_interrupt=True)
            logger.info(f"Ignoring interrupt due to stubbornness: {text[:30]}")
        else:
            # Actually interrupt
            await self.fsm.transition(Event.USER_INTERRUPT)
            await self.tts.stop()
            logger.info(f"Interrupted by user: {text[:30]}")

    async def _run_gatekeeper(self, text: str, energy: float) -> None:
        """Run gatekeeper decision."""
        gatekeeper_input = GatekeeperInput(
            text=text,
            silence_duration_ms=self._silence_duration_ms,
            audio_energy=energy,
            is_speaking=self.fsm.current_state == State.LISTENING
        )

        decision = await self.gatekeeper.decide(gatekeeper_input)
        logger.info(f"Gatekeeper decision: {decision.action.value} (confidence: {decision.confidence})")

        # Handle decision
        if decision.action == Action.REPLY:
            await self.fsm.transition(Event.GATEKEEPER_DECISION)
            await self._generate_response()
        elif decision.action == Action.ACCUMULATE:
            if self.accumulator:
                self.accumulator.add_segment(text, energy)
            await self.fsm.transition(Event.GATEKEEPER_DECISION)
        elif decision.action == Action.INTERRUPT:
            await self.fsm.transition(Event.USER_INTERRUPT)
            await self.tts.stop()

        # Reset silence tracking after decision
        self._silence_duration_ms = 0
        self._last_speech_time = None

    async def _generate_response(self) -> None:
        """Generate LLM response and speak it."""
        # Get text to respond to
        if self.accumulator and len(self.accumulator) > 0:
            # Use accumulated context
            context = self.accumulator.get_combined_text()
            self.accumulator.clear()
        else:
            context = self._current_text

        # Build messages
        messages = [Message(role="user", content=context)]

        # Get streaming response
        await self.fsm.transition(Event.LLM_RESPONSE)

        response_text = ""
        async for chunk in self.llm.stream_chat(messages, system_prompt=self.config.system_prompt):
            response_text = chunk.content

            # Stream TTS
            await self.fsm.transition(Event.LLM_RESPONSE)
            self._speaking_task = asyncio.create_task(self._speak(response_text))

        self._current_text = ""

    async def _speak(self, text: str) -> None:
        """Speak the given text."""
        await self.fsm.transition(Event.LLM_RESPONSE)

        # Stream TTS
        async for tts_result in self.tts.stream_synthesize(text):
            if self._on_tts_audio and tts_result.audio:
                self._on_tts_audio(tts_result.audio)

        await self.fsm.transition(Event.TTS_COMPLETE)

    # State handlers
    async def _on_listening(self) -> None:
        """Handler for LISTENING state."""
        logger.debug("Entered LISTENING state")
        if self._on_state_change:
            self._on_state_change(State.LISTENING)

    async def _on_speaking(self) -> None:
        """Handler for SPEAKING state."""
        logger.debug("Entered SPEAKING state")
        if self._on_state_change:
            self._on_state_change(State.SPEAKING)

    async def _on_processing(self) -> None:
        """Handler for PROCESSING state."""
        logger.debug("Entered PROCESSING state")
        if self._on_state_change:
            self._on_state_change(State.PROCESSING)

    async def _on_accumulating(self) -> None:
        """Handler for ACCUMULATING state."""
        logger.debug("Entered ACCUMULATING state")
        if self._on_state_change:
            self._on_state_change(State.ACCUMULATING)

    async def _on_accumulator_flush(self, text: str) -> None:
        """Handle accumulator flush - generate response to accumulated context."""
        logger.info(f"Accumulator flushed with: {text[:50]}...")
        await self._generate_response()

    # Callbacks
    def set_asr_callback(self, callback: callable) -> None:
        """Set callback for ASR results."""
        self._on_asr_result = callback

    def set_tts_callback(self, callback: callable) -> None:
        """Set callback for TTS audio output."""
        self._on_tts_audio = callback

    def set_state_change_callback(self, callback: callable) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def set_error_callback(self, callback: callable) -> None:
        """Set callback for errors."""
        self._on_error = callback

    def get_state(self) -> State:
        """Get current FSM state."""
        return self.fsm.current_state

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self.fsm.current_state == State.SPEAKING


async def create_orchestrator(
    vad: Optional[BaseVAD] = None,
    asr: Optional[BaseASR] = None,
    llm: Optional[BaseLLM] = None,
    tts: Optional[BaseTTS] = None,
    config: Optional[OrchestratorConfig] = None
) -> VoiceOrchestrator:
    """Factory function to create and initialize the orchestrator."""
    from ..perception.vad import create_vad
    from ..perception.asr import create_asr
    from ..cognition.llm import create_llm
    from ..cognition.tts import create_tts

    # Create components if not provided
    vad = vad or await create_vad()
    asr = asr or await create_asr()
    llm = llm or await create_llm()
    tts = tts or await create_tts()

    orchestrator = VoiceOrchestrator(
        vad=vad,
        asr=asr,
        llm=llm,
        tts=tts,
        config=config
    )

    await orchestrator.start()

    return orchestrator
