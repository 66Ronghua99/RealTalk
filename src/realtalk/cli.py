"""CLI application for voice interaction testing."""
import asyncio
import hashlib
import io
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd

from realtalk.cognition.llm import Message
from realtalk.config import get_config
from realtalk.logging_config import setup_logger
from realtalk.orchestration.fsm import State
from realtalk.orchestration.gatekeeper import Action, GatekeeperInput, create_gatekeeper
from realtalk.perception.asr import create_asr, BaseASR
from realtalk.perception.vad import create_vad, BaseVAD
from realtalk.cognition.llm import create_llm, BaseLLM
from realtalk.cognition.tts import create_tts, BaseTTS
from realtalk.orchestration.accumulator import ContextAccumulator
from realtalk.core.response_generator import ResponseGenerator, GenerationConfig

logger = setup_logger("realtalk.cli", level="DEBUG")


class AudioHandler:
    """Handles microphone input and speaker output.

    Full-duplex mode: microphone input is NOT suppressed during playback.
    Instead, interrupt detection uses energy comparison + high-threshold VAD
    to distinguish human speech from echo.
    """

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
        self._silence_threshold_ms = 1500
        self._is_playing = False
        self._playback_lock = asyncio.Lock()

        # Full-duplex interrupt detection state
        self._peak_speaker_rms: float = 0.0   # Peak-hold RMS of playback (decays slowly, never drops to 0 mid-play)
        self._post_play_cooldown_until: float = 0.0  # monotonic timestamp: echo suppression active until this time
        self._interrupt_requested: bool = False  # Signal to stop playback mid-stream

    def set_audio_callback(self, callback):
        """Set callback for incoming audio data."""
        self._audio_callback = callback

    def is_currently_playing(self) -> bool:
        """Check if audio is currently being played."""
        return self._is_playing

    def get_speaker_rms(self) -> float:
        """Get the peak-hold RMS energy of the speaker output (for echo estimation).

        Returns the peak-hold value rather than the instantaneous value so that
        the echo gate does not collapse to zero between 100ms playback blocks,
        which would otherwise allow any mic sound through the energy filter.
        """
        return self._peak_speaker_rms

    def is_in_echo_suppression_window(self) -> bool:
        """Return True if we are inside the echo-risk period.

        Echo risk includes:
        - Active TTS playback (_is_playing == True)
        - The 500ms cooldown after playback ends (room reverb / hardware tail)
        """
        return self._is_playing or time.monotonic() < self._post_play_cooldown_until

    def request_interrupt(self) -> None:
        """Signal to abort the current TTS playback immediately.

        Called when human speech is detected during playback. The play_audio()
        loop checks this flag every 100ms block.
        """
        if self._is_playing:
            self._interrupt_requested = True
            logger.info("[INTERRUPT] Playback interrupt requested")

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
        """Wrapper for audio callback.

        Full-duplex: mic input is always enqueued, even during playback.
        The interrupt detection logic in CLI._process_audio_chunk decides
        whether it's human speech or echo.
        """
        if status:
            logger.warning(f"Audio input status: {status}")

        # Convert to PCM16
        audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()

        if self._audio_callback:
            self._audio_callback(audio_bytes, time_info.inputBufferAdcTime)

    async def play_audio(self, audio_data: bytes, target_sample_rate: int = 32000):
        """Play audio data with real-time interrupt support.

        While playing, continuously updates _speaker_rms so that the VAD
        interrupt detector can compare mic energy against the expected echo level.
        Stops immediately if request_interrupt() is called.
        """
        if not self._output_stream or not self._running:
            return

        async with self._playback_lock:
            self._is_playing = True
            self._interrupt_requested = False

            interrupted = False
            try:
                import soundfile as sf

                audio_array, sample_rate = sf.read(io.BytesIO(audio_data), dtype='float32')

                if len(audio_array.shape) > 1:
                    audio_array = audio_array[:, 0]  # Take first channel

                # Resample if needed
                if sample_rate != target_sample_rate:
                    num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
                    indices = np.linspace(0, len(audio_array) - 1, num_samples)
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

                block_size = 1600
                loop = asyncio.get_running_loop()

                for i in range(0, len(audio_array), block_size):
                    if not self._running or self._interrupt_requested:
                        interrupted = self._interrupt_requested
                        break
                    chunk = audio_array[i:i + block_size]
                    if len(chunk) < block_size:
                        chunk = np.pad(chunk, (0, block_size - len(chunk)))

                    # Peak-hold speaker RMS: take max of new chunk vs decayed old peak.
                    # Decay factor 0.85 per 100ms block (~10dB/s), so the gate stays
                    # active for ~300-400ms even in silence gaps between chunks.
                    chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
                    self._peak_speaker_rms = max(chunk_rms, self._peak_speaker_rms * 0.85)

                    try:
                        await loop.run_in_executor(None, self._output_stream.write, chunk)
                    except Exception:
                        pass

                if interrupted:
                    logger.info("[INTERRUPT] Playback stopped due to user speech")

            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                # Shorter post-playback gap when interrupted (user is already speaking)
                post_silence = 0.15 if interrupted else 0.3
                await asyncio.sleep(post_silence)
                self._is_playing = False
                self._interrupt_requested = False
                # Set a 500ms cooldown window after playback ends.
                # During this window, mic input is still suppressed to absorb
                # room reverb and hardware echo tail.
                cooldown_s = 0.3 if interrupted else 0.5
                self._post_play_cooldown_until = time.monotonic() + cooldown_s
                self._peak_speaker_rms = 0.0  # Reset peak only after cooldown is set


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
        self.response_generator: Optional[ResponseGenerator] = None

        self._running = False
        self._audio_buffer: List[bytes] = []
        self._is_speaking = False
        self._last_speech_time: Optional[float] = None
        self._silence_threshold_ms = 1500  # 1.5s for more responsive capture
        self._current_state = State.IDLE
        self._stop_event = asyncio.Event()

        # VAD continuous detection counters to prevent premature trigger
        self._consecutive_silence_count = 0
        self._required_silence_frames = 8  # 0.8s / 0.1s per frame = 8 frames (more responsive)

        # Full-duplex interrupt detection parameters
        # _max_echo_ratio: estimated upper bound of mic/speaker ratio due to room acoustics.
        # Mac room echo is typically 80-106% of speaker level (empirically measured).
        # We model the max echo as speaker_rms * 1.1 to be slightly conservative.
        self._max_echo_ratio: float = 1.1
        # _min_user_energy: minimum "excess" energy above the estimated echo level
        # required to consider the microphone input as real human speech.
        # Formula: user_energy = mic_rms - speaker_rms * _max_echo_ratio
        # This subtracts the expected echo contribution and measures only what the
        # user adds on top. Keeps triggering possible even when speaker is loud.
        self._min_user_energy: float = 0.012
        # VAD threshold during active playback - very high to filter echo false positives.
        # Mac speakers produce room echo that Silero VAD scores at 0.97-1.00, so we need
        # a threshold above the typical echo confidence range.
        self._interrupt_vad_threshold: float = 0.95
        # Number of consecutive frames that must pass both filters before triggering interrupt
        self._interrupt_required_frames: int = 3
        self._interrupt_frame_count: int = 0  # Current consecutive interrupt frames

        # Audio processing queue for thread-safe async processing
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._audio_processor_task: Optional[asyncio.Task] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Audio deduplication to prevent duplicate playback
        self._played_audio_hashes: set = set()
        self._max_hash_history = 10

        # Debug logging
        self._debug_session_id = f"{int(time.time())}"
        self._debug_counter = 0
        self._debug_dir = Path(f"debug_cli_{self._debug_session_id}")
        self._debug_dir.mkdir(exist_ok=True)
        logger.info(f"Debug files will be saved to: {self._debug_dir.absolute()}")

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

        # Create response generator with conversation history
        logger.info("Initializing ResponseGenerator...")
        config = GenerationConfig(
            system_prompt="你是一个自然、有亲和力的AI对话伙伴。请遵循以下原则：\n"
                         "1. 回复简洁自然，像日常对话一样\n"
                         "2. 适当使用口语化表达，避免过于书面\n"
                         "3. 记住对话上下文，保持连贯性\n"
                         "4. 回复控制在2-3句话以内，除非需要详细解释",
            enable_streaming_tts=True
        )
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            tts=self.tts,
            config=config
        )
        # Set up callbacks
        self._current_call_id: Optional[str] = None  # Track current call ID for audio playback
        self.response_generator.set_callbacks(
            on_sentence_start=lambda s: logger.info(f"[Speaking] {s[:50]}..."),
            on_audio_chunk=lambda audio: asyncio.create_task(self._play_audio_chunk(audio)),
            on_complete=lambda text: logger.info(f"[Complete] {text[:50]}...")
        )

        # Create audio handler
        self.audio_handler = AudioHandler(
            sample_rate=cfg.asr.sample_rate,
            block_size=cfg.asr.sample_rate // 10,
        )
        self.audio_handler.set_audio_callback(self._on_audio_input)
        self.audio_handler.start()

        self._running = True
        self._audio_chunk_counter = 0  # Global counter for audio chunks

        # Store reference to the main event loop for thread-safe callbacks
        self._main_loop = asyncio.get_running_loop()

        logger.info("Ready! Say something...")

        # Start audio processor task
        self._audio_processor_task = asyncio.create_task(self._audio_processor_loop())

        # Main loop - just keep running until stopped
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        finally:
            # Cancel audio processor task
            if self._audio_processor_task and not self._audio_processor_task.done():
                self._audio_processor_task.cancel()
                try:
                    await self._audio_processor_task
                except asyncio.CancelledError:
                    pass
            await self.shutdown()

    def _on_audio_input(self, audio_chunk: bytes, timestamp: float):
        """Handle incoming audio from microphone.

        Full-duplex: audio is ALWAYS enqueued, even during playback.
        The _process_audio_chunk() method handles echo filtering and
        interrupt detection during playback.
        """
        if not self._running or self._main_loop is None:
            return

        current_time = time.monotonic()
        self._main_loop.call_soon_threadsafe(
            self._audio_queue.put_nowait, (audio_chunk, current_time)
        )

    async def _audio_processor_loop(self):
        """Main loop to process audio chunks from the queue."""
        logger.info("Audio processor started")
        try:
            while self._running:
                try:
                    audio_chunk, timestamp = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                    await self._process_audio_chunk(audio_chunk, timestamp)
                except asyncio.TimeoutError:
                    # Check for silence timeout when no audio received
                    await self._check_silence_timeout()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
        except asyncio.CancelledError:
            logger.info("Audio processor cancelled")

    async def _process_audio_chunk(self, audio_chunk: bytes, timestamp: float):
        """Process a single audio chunk.

        Full-duplex behavior:
        - During TTS playback: run interrupt detection (energy + high-threshold VAD)
        - Normal state: standard VAD for speech collection
        """
        # Convert to numpy for VAD
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        in_echo_window = self.audio_handler and self.audio_handler.is_in_echo_suppression_window()
        is_actively_playing = self.audio_handler and self.audio_handler.is_currently_playing()

        if in_echo_window:
            if not is_actively_playing:
                # ── POST-PLAY COOLDOWN ──────────────────────────────────────────────
                # Playback finished but we are still in the reverb/echo tail window.
                # Discard mic input entirely — no interrupt detection, no VAD.
                self._interrupt_frame_count = 0
                return

            # ── ACTIVE PLAYBACK: INTERRUPT DETECTION MODE ──────────────────────────
            # We're playing TTS audio. Run a two-stage filter to check if human is speaking:
            # Stage 1: Energy filter - mic must be significantly louder than expected echo.
            # Uses peak-hold speaker_rms so the gate never collapses to 0 between 100ms blocks.
            mic_rms = float(np.sqrt(np.mean(audio_array ** 2)))
            speaker_rms = self.audio_handler.get_speaker_rms()

            # Energy gate: subtract the estimated echo contribution from mic_rms.
            # Echo is at most ~110% of speaker (empirical), so anything beyond
            # that is assumed to be actual user voice added on top of the echo.
            # This approach scales naturally: works for both loud and quiet speaker levels.
            estimated_echo = speaker_rms * self._max_echo_ratio
            user_energy = mic_rms - estimated_echo
            energy_ok = user_energy > self._min_user_energy

            if energy_ok:
                # Stage 2: VAD confidence filter - must be VERY high confidence to avoid echo.
                # Mac room echo from Silero VAD often scores 0.97-0.99; threshold is 0.95.
                vad_result = await self.vad.detect(audio_array)
                vad_ok = vad_result.confidence >= self._interrupt_vad_threshold

                logger.debug(
                    f"[INTERRUPT-CHECK] mic_rms={mic_rms:.4f} speaker_rms={speaker_rms:.4f} "
                    f"est_echo={estimated_echo:.4f} user_energy={user_energy:.4f} vad_conf={vad_result.confidence:.3f}"
                )

                if vad_ok:
                    self._interrupt_frame_count += 1
                    logger.info(
                        f"[INTERRUPT] Frame {self._interrupt_frame_count}/{self._interrupt_required_frames} "
                        f"mic={mic_rms:.4f} speaker={speaker_rms:.4f} vad={vad_result.confidence:.3f}"
                    )
                    if self._interrupt_frame_count >= self._interrupt_required_frames:
                        logger.info("[INTERRUPT] 🛑 Human speech detected! Stopping TTS playback.")
                        await self._handle_interrupt(audio_chunk, timestamp)
                        return
                else:
                    self._interrupt_frame_count = 0
            else:
                self._interrupt_frame_count = 0
            return  # Do not accumulate audio for ASR during playback (until interrupt confirmed)

        # --- NORMAL MODE: standard VAD speech collection ---
        self._interrupt_frame_count = 0  # reset when not playing

        # Run VAD detection
        vad_result = await self.vad.detect(audio_array)

        if vad_result.is_speech:
            # User is speaking
            if not self._is_speaking:
                logger.info(f"[VAD] 🎤 SPEECH START (confidence: {vad_result.confidence:.3f})")
            self._audio_buffer.append(audio_chunk)
            self._last_speech_time = timestamp
            self._consecutive_silence_count = 0
            if not self._is_speaking:
                self._is_speaking = True
        else:
            # No speech detected
            self._consecutive_silence_count += 1

    async def _handle_interrupt(self, trigger_chunk: bytes, timestamp: float) -> None:
        """Handle a detected human speech interruption during TTS playback.

        Steps:
        1. Stop the current TTS generation pipeline
        2. Abort the current audio playback
        3. Drain the audio queue (discard queued echo)
        4. Reset VAD state and begin collecting the user's new speech
        """
        logger.info("[INTERRUPT] 🎙 Handling interruption: stopping TTS and collecting new speech")

        # 1. Stop LLM/TTS generation pipeline
        if self.response_generator and self.response_generator.is_generating():
            self.response_generator.stop()
            logger.info("[INTERRUPT] ResponseGenerator stopped")

        # 2. Request immediate abort of audio playback
        if self.audio_handler:
            self.audio_handler.request_interrupt()

        # 3. Drain queued audio to avoid processing buffered echo
        drained = await self._drain_audio_queue()
        if drained:
            logger.info(f"[INTERRUPT] Drained {drained} queued audio chunks")

        # 4. Reset VAD state and start collecting the interruption speech
        self._audio_buffer.clear()
        self._is_speaking = True
        self._consecutive_silence_count = 0
        self._interrupt_frame_count = 0

        # Add the triggering frame itself as the start of new speech
        self._audio_buffer.append(trigger_chunk)
        self._last_speech_time = timestamp
        logger.info("[INTERRUPT] ✅ Ready to collect user's interruption speech")

    async def _check_silence_timeout(self):
        """Check if continuous silence threshold exceeded and process audio."""
        # Only process if we have audio and have detected continuous silence
        if (self._consecutive_silence_count >= self._required_silence_frames and
            self._audio_buffer and self._is_speaking):
            silence_duration = (time.monotonic() - self._last_speech_time) * 1000 if self._last_speech_time else 0
            logger.info(f"[VAD] Continuous silence detected: {self._consecutive_silence_count} frames ({silence_duration:.0f}ms), processing audio")
            await self._process_audio_end()
            self._consecutive_silence_count = 0

    async def _process_audio_end(self):
        """Process accumulated audio when user stops speaking."""
        if not self._audio_buffer:
            return

        self._debug_counter += 1
        call_id = f"{self._debug_session_id}_{self._debug_counter}"
        self._current_call_id = call_id  # Store for audio playback callback

        logger.info(f"[DEBUG-{call_id}] === NEW TURN STARTED ===")
        logger.info(f"[DEBUG-{call_id}] Processing {len(self._audio_buffer)} audio chunks...")

        # Combine all audio chunks
        combined_audio = b"".join(self._audio_buffer)
        total_bytes = len(combined_audio)
        duration_ms = (total_bytes / 2) / 16  # 16-bit PCM at 16kHz = 2 bytes per sample
        logger.info(f"[DEBUG-{call_id}] Total audio: {total_bytes} bytes, ~{duration_ms:.0f}ms")

        # Save input audio for debugging
        input_audio_path = self._debug_dir / f"input_{call_id}.pcm"
        input_audio_path.write_bytes(combined_audio)
        logger.info(f"[DEBUG-{call_id}] Input audio saved to: {input_audio_path}")

        self._audio_buffer.clear()
        self._is_speaking = False
        self._last_speech_time = None

        # Run ASR
        try:
            asr_result = await self.asr.recognize(combined_audio)
            logger.info(f"[DEBUG-{call_id}] [ASR] Result: '{asr_result.text}' (language: {asr_result.language}, confidence: {asr_result.confidence})")

            # Save ASR result to text file
            asr_text_path = self._debug_dir / f"asr_{call_id}.txt"
            asr_text_path.write_text(f"Text: {asr_result.text}\nLanguage: {asr_result.language}\nConfidence: {asr_result.confidence}\n")


            if not asr_result.text:
                logger.info(f"[DEBUG-{call_id}] No speech detected, waiting...")
                return

            # Run gatekeeper decision
            gatekeeper_input = GatekeeperInput(
                text=asr_result.text,
                silence_duration_ms=self._silence_threshold_ms + 100,
                audio_energy=0.5,
                is_speaking=False
            )

            decision = await self.gatekeeper.decide(gatekeeper_input)
            logger.info(f"[DEBUG-{call_id}] [Gatekeeper] Decision: {decision.action.value} (confidence: {decision.confidence:.2f})")

            if decision.action == Action.REPLY:
                # Save the text being sent to LLM
                llm_input_path = self._debug_dir / f"llm_input_{call_id}.txt"
                context_text = asr_result.text
                if len(self.accumulator) > 0:
                    context_text = self.accumulator.get_combined_text() + " " + asr_result.text
                llm_input_path.write_text(context_text)
                logger.info(f"[DEBUG-{call_id}] LLM input saved to: {llm_input_path}")

                await self._generate_response(asr_result.text, call_id)
            elif decision.action == Action.ACCUMULATE:
                self.accumulator.add_segment(asr_result.text, 0.5)
                logger.info(f"[DEBUG-{call_id}] [Accumulating context...] Accumulator size: {len(self.accumulator)}")

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()

    async def _play_audio_chunk(self, audio_data: bytes) -> None:
        """Play an audio chunk from TTS with deduplication and debug saving."""
        if not self.audio_handler or not audio_data:
            return

        call_id = self._current_call_id or "unknown"

        # Global counter to track if multiple chunks are in-flight simultaneously
        self._audio_chunk_counter += 1
        chunk_id = self._audio_chunk_counter
        enqueue_ts = time.monotonic()

        # Deduplication: check if this exact audio was already played
        audio_hash = hashlib.md5(audio_data[:1024]).hexdigest()  # Hash first 1KB for speed

        logger.info(
            f"[AUDIO-TRACE] chunk_id={chunk_id} RECEIVED size={len(audio_data)} "
            f"hash={audio_hash[:8]} call_id={call_id}"
        )

        if audio_hash in self._played_audio_hashes:
            logger.warning(
                f"[AUDIO-TRACE] chunk_id={chunk_id} DUPLICATE SKIP hash={audio_hash[:8]}"
            )
            return

        # Add to played set
        self._played_audio_hashes.add(audio_hash)

        # Limit history size
        if len(self._played_audio_hashes) > self._max_hash_history:
            self._played_audio_hashes.pop()

        # Save TTS audio for debugging
        tts_counter = getattr(self, '_tts_counter', 0)
        self._tts_counter = tts_counter + 1
        tts_audio_path = self._debug_dir / f"tts_{call_id}_{tts_counter:03d}_{audio_hash[:8]}.mp3"
        tts_audio_path.write_bytes(audio_data)
        logger.info(
            f"[AUDIO-TRACE] chunk_id={chunk_id} SAVED {tts_audio_path.name}"
        )

        start_ts = time.monotonic()
        wait_ms = (start_ts - enqueue_ts) * 1000
        logger.info(
            f"[AUDIO-TRACE] chunk_id={chunk_id} PLAY_START "
            f"wait={wait_ms:.1f}ms size={len(audio_data)}"
        )

        await self.audio_handler.play_audio(audio_data)

        end_ts = time.monotonic()
        play_ms = (end_ts - start_ts) * 1000
        logger.info(
            f"[AUDIO-TRACE] chunk_id={chunk_id} PLAY_END "
            f"duration={play_ms:.1f}ms"
        )

    async def _generate_response(self, text: str, call_id: str):
        """Generate LLM response and speak it using ResponseGenerator."""
        if not self.response_generator:
            logger.error(f"[DEBUG-{call_id}] ResponseGenerator not initialized")
            return

        # Wait for previous generation to fully stop (needed after interrupts,
        # since stop() is async and _is_generating resets in the pipeline's finally block)
        if self.response_generator.is_generating():
            logger.info(f"[DEBUG-{call_id}] Waiting for previous generation to stop...")
            for _ in range(10):  # Max ~1 second wait
                await asyncio.sleep(0.1)
                if not self.response_generator.is_generating():
                    break
            if self.response_generator.is_generating():
                logger.warning(f"[DEBUG-{call_id}] Generation still active after 1s, skipping")
                return

        # Get context from accumulator if available
        if len(self.accumulator) > 0:
            context = self.accumulator.get_combined_text()
            self.accumulator.clear()
            logger.info(f"[DEBUG-{call_id}] [Response] Using accumulated context: '{context[:50]}...'")
        else:
            context = text

        logger.info(f"[DEBUG-{call_id}] [Response] Generating for: '{context[:50]}...'")

        try:
            # Generate and speak response
            response = await self.response_generator.generate_response(context)

            if response:
                logger.info(f"[DEBUG-{call_id}] [Response] Complete: '{response[:100]}...'")

                # Save LLM response to text file
                llm_output_path = self._debug_dir / f"llm_output_{call_id}.txt"
                llm_output_path.write_text(response)
                logger.info(f"[DEBUG-{call_id}] LLM output saved to: {llm_output_path}")

                # Drain any audio chunks queued during playback (echo suppression)
                await self._drain_audio_queue()
            else:
                logger.warning(f"[DEBUG-{call_id}] [Response] Generation returned None")

        except Exception as e:
            logger.error(f"[DEBUG-{call_id}] [Response] Generation error: {e}")
            import traceback
            traceback.print_exc()

    async def _drain_audio_queue(self) -> int:
        """Drain all pending audio chunks from the queue to prevent echo.

        During TTS playback, the microphone callback continues enqueueing audio.
        These chunks need to be discarded to prevent the AI from hearing itself.
        """
        drained = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained > 0:
            logger.info(f"Drained {drained} audio chunks from queue to prevent echo")
        return drained

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
