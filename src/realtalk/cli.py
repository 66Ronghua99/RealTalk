"""CLI application for voice interaction testing."""
import asyncio
import io
import signal
import sys
from typing import Optional, List

import numpy as np
import sounddevice as sd

from ..cognition.llm import Message
from ..config import get_config
from ..logging_config import setup_logger
from ..orchestration.fsm import State
from ..orchestration.gatekeeper import Action, GatekeeperInput, create_gatekeeper
from ..perception.asr import create_asr, BaseASR
from ..perception.vad import create_vad, BaseVAD
from ..cognition.llm import create_llm, BaseLLM
from ..cognition.tts import create_tts, BaseTTS
from ..orchestration.accumulator import ContextAccumulator

logger = setup_logger("realtalk.cli")


class AudioHandler:
    """Handles microphone input and speaker output."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 1600,  # 100ms at 16kHz
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._audio_callback = None
        self._playback_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._audio_buffer: List[bytes] = []
        self._last_speech_time: Optional[float] = None
        self._silence_threshold_ms = 500  # Silence threshold to detect speech end

    def set_audio_callback(self, callback):
        """Set callback for incoming audio data."""
        self._audio_callback = callback

    def start(self):
        """Start audio input/output streams."""
        self._running = True

        # Input stream (microphone)
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.block_size,
            callback=self._audio_callback_wrapper,
        )
        self._input_stream.start()

        # Output stream (speaker) - use higher sample rate for TTS
        self._output_stream = sd.OutputStream(
            samplerate=32000,  # TTS sample rate
            channels=1,
            dtype=np.float32,
            blocksize=self.block_size,
        )
        self._output_stream.start()

        logger.info(f"Audio started: mic={self.sample_rate}Hz, speaker=32000Hz")

    def stop(self):
        """Stop audio streams."""
        self._running = False

        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None

        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

        logger.info("Audio stopped")

    def _audio_callback_wrapper(self, indata, frames, time_info, status):
        """Wrapper for audio callback."""
        if status:
            logger.warning(f"Audio input status: {status}")

        # Convert to PCM16
        audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()

        if self._audio_callback:
            self._audio_callback(audio_bytes, time_info.inputBufferAdcTime)

    async def play_audio(self, audio_data: bytes, target_sample_rate: int = 32000):
        """Play audio data."""
        if not self._output_stream or not self._running:
            return

        try:
            # Use soundfile to read WAV
            import soundfile as sf

            # Read audio data
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data), dtype='float32')

            # Mono to stereo if needed
            if len(audio_array.shape) == 1:
                pass  # Already mono
            else:
                audio_array = audio_array[:, 0]  # Take first channel

            # If sample rate doesn't match, simple linear resample
            if sample_rate != target_sample_rate:
                num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                indices = np.linspace(0, len(audio_array) - 1, num_samples)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

            # Write in chunks
            block_size = 1600
            for i in range(0, len(audio_array), block_size):
                if not self._running:
                    break
                chunk = audio_array[i:i + block_size]
                if len(chunk) < block_size:
                    chunk = np.pad(chunk, (0, block_size - len(chunk)))
                try:
                    self._output_stream.write(chunk)
                except:
                    pass

        except Exception as e:
            logger.error(f"Playback error: {e}")


class CLI:
    """CLI application for voice interaction."""

    def __init__(self):
        self.vad: Optional[BaseVAD] = None
        self.asr: Optional[BaseASR] = None
        self.llm: Optional[BaseLLM] = None
        self.tts: Optional[BaseTTS] = None
        self.gatekeeper = None
        self.audio_handler: Optional[AudioHandler] = None
        self.accumulator = ContextAccumulator()

        self._running = False
        self._audio_buffer: List[bytes] = []
        self._is_speaking = False
        self._last_speech_time: Optional[float] = None
        self._silence_threshold_ms = 500
        self._current_state = State.IDLE
        self._stop_event = asyncio.Event()

    async def run(self):
        """Run the CLI application."""
        logger.info("=" * 50)
        logger.info("RealTalk CLI - Voice Interaction Test")
        logger.info("=" * 50)
        logger.info("Starting... Please speak!")
        logger.info("Press Ctrl+C to exit")
        logger.info("=" * 50)

        cfg = get_config()

        # Initialize components
        logger.info("Initializing VAD...")
        self.vad = await create_vad()

        logger.info("Initializing ASR...")
        self.asr = await create_asr()

        logger.info("Initializing LLM...")
        self.llm = await create_llm()

        logger.info("Initializing TTS...")
        self.tts = await create_tts()

        logger.info("Initializing Gatekeeper...")
        self.gatekeeper = await create_gatekeeper()

        # Create audio handler
        self.audio_handler = AudioHandler(
            sample_rate=cfg.asr.sample_rate,
            block_size=cfg.asr.sample_rate // 10,
        )
        self.audio_handler.set_audio_callback(self._on_audio_input)
        self.audio_handler.start()

        self._running = True
        logger.info("Ready! Say something...")

        # Start audio processing loop
        while self._running:
            await asyncio.sleep(0.1)
            await self._check_silence()

        await self.shutdown()

    def _on_audio_input(self, audio_chunk: bytes, timestamp: float):
        """Handle incoming audio from microphone."""
        if not self._running:
            return

        # Convert to numpy for VAD
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # Run VAD detection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            vad_result = loop.run_until_complete(self.vad.detect(audio_array))
        finally:
            loop.close()

        if vad_result.is_speech:
            # User is speaking
            self._audio_buffer.append(audio_chunk)
            self._last_speech_time = timestamp
            if not self._is_speaking:
                self._is_speaking = True
                logger.info("[Listening...]")
        else:
            # No speech
            if self._last_speech_time is not None:
                silence_duration = (timestamp - self._last_speech_time) * 1000  # ms
                if silence_duration > self._silence_threshold_ms and self._audio_buffer:
                    # User stopped speaking, process audio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._process_audio_end())
                    finally:
                        loop.close()

    async def _check_silence(self):
        """Background task to check for silence."""
        pass

    async def _process_audio_end(self):
        """Process accumulated audio when user stops speaking."""
        if not self._audio_buffer:
            return

        logger.info(f"Processing {len(self._audio_buffer)} audio chunks...")

        # Combine all audio chunks
        combined_audio = b"".join(self._audio_buffer)
        self._audio_buffer.clear()
        self._is_speaking = False
        self._last_speech_time = None

        # Run ASR
        try:
            asr_result = await self.asr.recognize(combined_audio)
            logger.info(f"[ASR] {asr_result.text}")

            if not asr_result.text:
                logger.info("No speech detected, waiting...")
                return

            # Run gatekeeper decision
            gatekeeper_input = GatekeeperInput(
                text=asr_result.text,
                silence_duration_ms=self._silence_threshold_ms + 100,
                audio_energy=0.5,
                is_speaking=False
            )

            decision = await self.gatekeeper.decide(gatekeeper_input)
            logger.info(f"[Gatekeeper] {decision.action.value}")

            if decision.action == Action.REPLY:
                await self._generate_response(asr_result.text)
            elif decision.action == Action.ACCUMULATE:
                self.accumulator.add_segment(asr_result.text, 0.5)
                logger.info("[Accumulating context...]")

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()

    async def _generate_response(self, text: str):
        """Generate LLM response and speak it."""
        # Get context from accumulator if available
        if len(self.accumulator) > 0:
            context = self.accumulator.get_combined_text()
            self.accumulator.clear()
        else:
            context = text

        logger.info("[Thinking...]")

        # Build messages
        messages = [
            Message(role="user", content=context)
        ]

        system_prompt = "You are a helpful and conversational AI assistant. Keep your responses short and natural."

        try:
            # Get streaming response
            response_text = ""
            async for chunk in self.llm.stream_chat(messages, system_prompt=system_prompt):
                response_text = chunk.content
                logger.debug(f"[LLM] {chunk.content}")

            logger.info(f"[LLM] {response_text}")

            if response_text:
                await self._speak(response_text)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()

    async def _speak(self, text: str):
        """Speak the given text."""
        logger.info(f"[Speaking] {text[:50]}...")

        try:
            async for tts_result in self.tts.stream_synthesize(text):
                if tts_result.audio and self.audio_handler:
                    # Play audio directly (TTS returns WAV)
                    await self.audio_handler.play_audio(tts_result.audio)

            logger.info("[Done speaking]")
        except Exception as e:
            logger.error(f"Error speaking: {e}")
            import traceback
            traceback.print_exc()

    async def shutdown(self):
        """Shutdown the application."""
        logger.info("Shutting down...")

        self._running = False

        if self.audio_handler:
            self.audio_handler.stop()

        if self.vad:
            await self.vad.close()
        if self.asr:
            await self.asr.close()
        if self.tts:
            await self.tts.close()

        logger.info("Goodbye!")


async def run_cli():
    """Run the CLI application."""
    cli = CLI()
    await cli.run()


def main():
    """CLI entry point."""
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")


if __name__ == "__main__":
    main()
