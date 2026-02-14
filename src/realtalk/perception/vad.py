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
            import torch
            from silero import vad

            self._model, _ = vad.load(model_id="silero_v5", language="en")
            logger.info("Silero VAD model loaded successfully")
        except ImportError:
            logger.warning("silero-vad not installed, using fallback")
            self._model = None

    async def detect(self, audio_chunk: np.ndarray) -> VADResult:
        """Detect voice activity."""
        if self._model is None:
            # Fallback: simple energy-based detection
            return self._energy_based_detection(audio_chunk)

        try:
            import torch

            audio_tensor = torch.from_numpy(audio_chunk).float()
            audio_tensor = audio_tensor.unsqueeze(0)

            with torch.no_grad():
                speech_prob = self._model(audio_tensor).item()

            return VADResult(
                is_speech=speech_prob > self.threshold,
                confidence=speech_prob,
                timestamp_ms=0
            )
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return VADResult(is_speech=False, confidence=0.0, timestamp_ms=0)

    def _energy_based_detection(self, audio_chunk: np.ndarray) -> VADResult:
        """Fallback energy-based voice detection."""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = rms > 0.01  # Simple threshold

        return VADResult(
            is_speech=is_speech,
            confidence=float(rms * 10),  # Scale to 0-1 range
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
