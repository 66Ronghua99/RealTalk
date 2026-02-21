"""Voice Activity Detection (VAD) module."""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

from ..config import get_config
from ..exceptions import VADError
from ..logging_config import setup_logger

logger = setup_logger("realtalk.vad")


@dataclass
class VADResult:
    """VAD detection result."""
    is_speech: bool
    confidence: float
    timestamp_ms: int


class BaseVAD(ABC):
    """Base class for VAD implementations."""

    @abstractmethod
    async def detect(self, audio_chunk: np.ndarray) -> VADResult:
        """Detect voice activity in audio chunk."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the VAD model."""
        pass


class SileroVAD(BaseVAD):
    """Silero VAD implementation."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._sample_rate = 16000

    async def load(self) -> None:
        """Load the Silero VAD model."""
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
            logger.info("Silero VAD model loaded successfully")
        except ImportError:
            logger.warning("silero-vad not installed, using fallback")
            self._model = None

    async def detect(self, audio_chunk: np.ndarray) -> VADResult:
        """Detect voice activity.

        New silero-vad v6.x requires 512 samples (32ms @ 16kHz) per inference.
        We split the audio chunk into frames and aggregate results.
        """
        if self._model is None:
            # Fallback: simple energy-based detection
            return self._energy_based_detection(audio_chunk)

        try:
            import torch

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)

            # silero-vad v6.x requires 512 samples per frame (32ms @ 16kHz)
            frame_size = 512
            num_frames = audio_tensor.shape[1] // frame_size

            if num_frames == 0:
                # Audio chunk too small, use energy-based fallback
                return self._energy_based_detection(audio_chunk)

            # Process each frame and collect probabilities
            probs = []
            with torch.no_grad():
                for i in range(num_frames):
                    frame = audio_tensor[:, i * frame_size:(i + 1) * frame_size]
                    if frame.shape[1] == frame_size:
                        prob = self._model(frame, self._sample_rate).item()
                        probs.append(prob)

            if not probs:
                return self._energy_based_detection(audio_chunk)

            # Use max probability as the speech confidence
            max_prob = max(probs)
            avg_prob = sum(probs) / len(probs)

            # Weighted: favor max but consider average
            confidence = 0.7 * max_prob + 0.3 * avg_prob

            return VADResult(
                is_speech=confidence > self.threshold,
                confidence=confidence,
                timestamp_ms=0
            )
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return self._energy_based_detection(audio_chunk)

    def _energy_based_detection(self, audio_chunk: np.ndarray) -> VADResult:
        """Fallback energy-based voice detection."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = rms > 0.02  # Lowered threshold for more sensitive voice detection

        return VADResult(
            is_speech=is_speech,
            confidence=float(min(rms * 20, 1.0)),  # Scale to 0-1 range (adjusted multiplier)
            timestamp_ms=0
        )

    async def close(self) -> None:
        """Close the VAD model."""
        if self._model is not None:
            del self._model
            self._model = None


class WebRTCVAD(BaseVAD):
    """WebRTC VAD implementation."""

    def __init__(self, sample_rate: int = 16000, mode: int = 3):
        self.sample_rate = sample_rate
        self.mode = mode
        self._vad = None

    async def load(self) -> None:
        """Load WebRTC VAD."""
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(mode=self.mode)
            logger.info("WebRTC VAD loaded successfully")
        except Exception as e:
            logger.warning(f"WebRTC VAD not available: {e}")
            self._vad = None

    async def detect(self, audio_chunk: np.ndarray) -> VADResult:
        """Detect voice activity."""
        if self._vad is None:
            # Fallback to energy detection
            return VADResult(is_speech=False, confidence=0.0, timestamp_ms=0)

        try:
            # Convert to 16-bit PCM
            audio_int16 = (audio_chunk * 32767).astype(np.int16).tobytes()

            # Process in 10ms, 20ms, or 30ms frames
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)

            if len(audio_int16) < frame_size * 2:
                return VADResult(is_speech=False, confidence=0.0, timestamp_ms=0)

            is_speech = self._vad.is_speech(
                audio_int16[:frame_size * 2],
                self.sample_rate
            )

            return VADResult(
                is_speech=bool(is_speech),
                confidence=1.0 if is_speech else 0.0,
                timestamp_ms=frame_duration
            )
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return VADResult(is_speech=False, confidence=0.0, timestamp_ms=0)

    async def close(self) -> None:
        """Close the VAD."""
        self._vad = None


async def create_vad(config: Optional[dict] = None) -> BaseVAD:
    """Factory function to create VAD instance."""
    cfg = get_config()
    vad_config = cfg.vad if config is None else config

    if vad_config.model_name == "silero":
        vad = SileroVAD(threshold=vad_config.threshold)
        await vad.load()
        return vad
    elif vad_config.model_name == "webrtc":
        vad = WebRTCVAD()
        await vad.load()
        return vad
    else:
        # Default to Silero
        vad = SileroVAD(threshold=vad_config.threshold)
        await vad.load()
        return vad
