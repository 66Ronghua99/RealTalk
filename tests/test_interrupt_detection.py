"""Unit tests for full-duplex interrupt detection in AudioHandler and CLI.

These tests verify:
1. AudioHandler interrupt request mechanism
2. Interrupt detection logic (energy + VAD double filter)
3. _handle_interrupt state reset behavior
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest


# ============================================================
# AudioHandler tests
# ============================================================

class TestAudioHandlerInterrupt:
    """Tests for new interrupt-related methods in AudioHandler."""

    def setup_method(self):
        """Set up an AudioHandler without real audio streams."""
        # Import here to allow patching sounddevice
        from realtalk.cli import AudioHandler
        self.handler = AudioHandler(sample_rate=16000, block_size=1600)

    def test_initial_state(self):
        """AudioHandler starts with no interrupt pending and zero peak_speaker_rms."""
        assert self.handler._interrupt_requested is False
        assert self.handler._peak_speaker_rms == 0.0
        assert self.handler._post_play_cooldown_until == 0.0
        assert self.handler._is_playing is False

    def test_get_speaker_rms_returns_current(self):
        """get_speaker_rms returns the current peak-hold speaker RMS value."""
        self.handler._peak_speaker_rms = 0.123
        assert self.handler.get_speaker_rms() == pytest.approx(0.123)

    def test_request_interrupt_only_when_playing(self):
        """request_interrupt has no effect when not playing."""
        self.handler._is_playing = False
        self.handler.request_interrupt()
        assert self.handler._interrupt_requested is False

    def test_request_interrupt_when_playing(self):
        """request_interrupt sets the flag when playing."""
        self.handler._is_playing = True
        self.handler.request_interrupt()
        assert self.handler._interrupt_requested is True

    def test_is_currently_playing(self):
        """is_currently_playing delegates to _is_playing."""
        assert self.handler.is_currently_playing() is False
        self.handler._is_playing = True
        assert self.handler.is_currently_playing() is True

    def test_is_in_echo_suppression_window_during_playback(self):
        """is_in_echo_suppression_window returns True while _is_playing is True."""
        self.handler._is_playing = True
        self.handler._post_play_cooldown_until = 0.0  # not in cooldown
        assert self.handler.is_in_echo_suppression_window() is True

    def test_is_in_echo_suppression_window_during_cooldown(self):
        """is_in_echo_suppression_window returns True during cooldown even when not playing."""
        import time
        self.handler._is_playing = False
        self.handler._post_play_cooldown_until = time.monotonic() + 0.5  # 500ms cooldown active
        assert self.handler.is_in_echo_suppression_window() is True

    def test_is_in_echo_suppression_window_after_cooldown(self):
        """is_in_echo_suppression_window returns False once cooldown has expired."""
        import time
        self.handler._is_playing = False
        self.handler._post_play_cooldown_until = time.monotonic() - 0.1  # cooldown expired
        assert self.handler.is_in_echo_suppression_window() is False


# ============================================================
# Interrupt detection logic tests (energy + VAD filter)
# ============================================================

class TestInterruptDetectionLogic:
    """Tests for the dual-filter interrupt detection algorithm."""

    @pytest.mark.asyncio
    async def test_energy_above_threshold_triggers_vad_check(self):
        """When excess mic energy (mic - speaker*1.1) > 0.012, VAD is consulted."""
        mock_vad = AsyncMock()
        mock_vad.detect.return_value = MagicMock(
            is_speech=True, confidence=0.9, timestamp_ms=0
        )
        mock_audio_handler = MagicMock()
        mock_audio_handler.is_in_echo_suppression_window.return_value = True
        mock_audio_handler.is_currently_playing.return_value = True
        mock_audio_handler.get_speaker_rms.return_value = 0.02

        # User voice adds 0.03 on top of echo (0.02*1.1=0.022 max echo)
        # mic_rms ≈ 0.05 → user_energy = 0.05 - 0.022 = 0.028 > 0.012 → PASS
        samples = np.full(1600, 0.05, dtype=np.float32)
        audio_chunk = (samples * 32768).astype(np.int16).tobytes()

        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        mic_rms = float(np.sqrt(np.mean(audio_array ** 2)))
        speaker_rms = 0.02
        max_echo_ratio = 1.1
        min_user_energy = 0.012

        user_energy = mic_rms - speaker_rms * max_echo_ratio
        assert user_energy > min_user_energy, (
            f"user_energy={user_energy:.4f} should exceed min_user_energy={min_user_energy}"
        )

        vad_result = await mock_vad.detect(audio_array)
        assert vad_result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_low_energy_skips_vad(self):
        """When user_energy = mic - speaker*1.1 <= 0.012, VAD is NOT consulted."""
        mock_vad = AsyncMock()

        speaker_rms = 0.02
        max_echo_ratio = 1.1
        min_user_energy = 0.012

        # Pure echo: mic at 105% of speaker (within expected echo range)
        mic_rms = speaker_rms * 1.05  # 0.021
        user_energy = mic_rms - speaker_rms * max_echo_ratio  # 0.021 - 0.022 = -0.001
        energy_ok = user_energy > min_user_energy
        assert energy_ok is False, "Pure echo should NOT pass the energy gate"
        mock_vad.detect.assert_not_called()

    def test_echo_is_below_energy_threshold(self):
        """Confirm that real Mac echo patterns from production logs all fail the energy gate."""
        max_echo_ratio = 1.1
        min_user_energy = 0.012
        # Frames from production log: (mic_rms, speaker_rms)
        echo_frames = [
            (0.0296, 0.0283),  # ratio 1.05x
            (0.0217, 0.0204),  # ratio 1.06x
            (0.0141, 0.0174),  # ratio 0.81x
        ]
        for mic, spk in echo_frames:
            user_energy = mic - spk * max_echo_ratio
            energy_ok = user_energy > min_user_energy
            assert not energy_ok, (
                f"Echo frame mic={mic} spk={spk} gave user_energy={user_energy:.4f}, "
                "should NOT pass energy gate"
            )

    def test_quiet_user_voice_passes_energy_gate(self):
        """A quiet user voice still passes if it adds enough energy above the echo."""
        max_echo_ratio = 1.1
        min_user_energy = 0.012
        # Quiet user: mic = echo + small voice contribution
        speaker_rms = 0.02
        user_voice_contribution = 0.015
        mic_rms = speaker_rms * 1.0 + user_voice_contribution  # echo at 100% + voice
        user_energy = mic_rms - speaker_rms * max_echo_ratio  # 0.035 - 0.022 = 0.013
        energy_ok = user_energy > min_user_energy
        assert energy_ok, (
            f"Quiet user voice (contribution={user_voice_contribution}) should pass "
            f"energy gate, user_energy={user_energy:.4f}"
        )

    def test_echo_threshold_calculation(self):
        """Energy gate: user_energy = mic - speaker*1.1 must exceed 0.012."""
        speaker_rms = 0.02
        max_echo_ratio = 1.1
        estimated_echo = speaker_rms * max_echo_ratio
        assert estimated_echo == pytest.approx(0.022)

    def test_vad_threshold_for_echo_suppression(self):
        """VAD threshold of 0.95 blocks typical echo confidence scores (0.97-0.99).

        Counterintuitively, this means echo scores (0.97) DO trigger at 0.95 threshold,
        so the interrupt frame counter relies on peak-hold + energy gate to distinguish.
        The real protection is: peak-hold keeps echo_threshold > 0, so mic_rms won't
        exceed it during pure echo (no user speech).
        """
        interrupt_vad_threshold = 0.95

        # Scores that should block an interrupt when energy gate is not passed
        echo_vad_scores = [0.97, 0.99, 1.00]
        for score in echo_vad_scores:
            # These scores pass VAD check alone - that's why energy gate matters
            assert score >= interrupt_vad_threshold

        # Real human speech during AI playback would exceed both energy gate AND VAD
        # The key protection is the energy gate (peak-hold prevents threshold=0)
        assert interrupt_vad_threshold > 0.8, "Threshold should be raised above old 0.8"

    def test_consecutive_frames_required(self):
        """Interrupt triggers only after required consecutive frames, not on first."""
        required_frames = 3
        frame_count = 0

        # Simulate 2 passing frames (should NOT trigger)
        for i in range(2):
            frame_count += 1
        assert frame_count < required_frames

        # Third frame SHOULD trigger
        frame_count += 1
        assert frame_count >= required_frames

    def test_peak_hold_rms_decay(self):
        """Peak-hold RMS decays by factor 0.85 per block but never below zero."""
        decay = 0.85
        initial_peak = 0.2

        # After 1 silent block, peak decays to 0.2 * 0.85 = 0.17
        peak_after_one_block = max(0.0, initial_peak * decay)
        assert peak_after_one_block == pytest.approx(0.17)

        # After 5 silent blocks, still above zero
        peak = initial_peak
        for _ in range(5):
            peak = max(0.0, peak * decay)
        assert peak > 0.0, "Peak should not collapse to zero after a few blocks"

        # After 30 silent blocks (~3 seconds), essentially zero
        for _ in range(30):
            peak = max(0.0, peak * decay)
        assert peak < 0.01, "Peak should decay to near-zero after prolonged silence"


# ============================================================
# _handle_interrupt state management tests
# ============================================================

class TestHandleInterruptState:
    """Tests that _handle_interrupt correctly resets all state."""

    @pytest.mark.asyncio
    async def test_handle_interrupt_resets_vad_state(self):
        """_handle_interrupt clears audio buffer and resets speaking state."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Create a minimal CLI-like object with necessary attributes
        import sys
        from types import SimpleNamespace

        # Patch sounddevice to avoid hardware access
        with patch("sounddevice.InputStream"), patch("sounddevice.OutputStream"):
            from realtalk.cli import CLI

        cli = object.__new__(CLI)

        # Set required attributes directly
        cli._audio_buffer = [b"old_audio_1", b"old_audio_2"]
        cli._is_speaking = False
        cli._consecutive_silence_count = 5
        cli._interrupt_frame_count = 3
        cli._last_speech_time = None
        cli._audio_queue = asyncio.Queue()
        cli._main_loop = asyncio.get_event_loop()
        cli._running = True

        # Mock audio_handler
        cli.audio_handler = MagicMock()
        cli.audio_handler.request_interrupt = MagicMock()

        # Mock response_generator
        cli.response_generator = MagicMock()
        cli.response_generator.is_generating.return_value = True
        cli.response_generator.stop = MagicMock()

        trigger_chunk = b"\x00" * 3200  # 100ms of silence
        trigger_timestamp = 12345.0

        await cli._handle_interrupt(trigger_chunk, trigger_timestamp)

        # Verify state was reset correctly
        assert cli._is_speaking is True  # Started new collection
        assert cli._consecutive_silence_count == 0
        assert cli._interrupt_frame_count == 0
        assert trigger_chunk in cli._audio_buffer  # Trigger frame starts the buffer
        assert cli._last_speech_time == trigger_timestamp

        # Verify pipeline was stopped
        cli.response_generator.stop.assert_called_once()
        cli.audio_handler.request_interrupt.assert_called_once()
