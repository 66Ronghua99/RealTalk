"""Streaming ASR (Automatic Speech Recognition) module using Minimax API."""
import asyncio
import base64
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp

from ..config import get_config
from ..exceptions import ASRError
from ..logging_config import setup_logger

logger = setup_logger("realtalk.asr")


@dataclass
class ASRResult:
    """ASR recognition result."""
    text: str
    is_final: bool
    language: Optional[str] = None
    confidence: float = 0.0


class BaseASR(ABC):
    """Base class for ASR implementations."""

    @abstractmethod
    async def recognize(self, audio_chunk: bytes) -> ASRResult:
        """Recognize speech from audio chunk."""
        pass

    @abstractmethod
    async def stream_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """Process streaming audio."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the ASR."""
        pass


class MinimaxASR(BaseASR):
    """Minimax ASR implementation."""

    def __init__(
        self,
        api_key: str,
        group_id: str,
        language: str = "auto",
        sample_rate: int = 16000
    ):
        self.api_key = api_key
        self.group_id = group_id
        self.language = language
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None

        # For Minimax Filler API (streaming ASR)
        self._base_url = "https://api.minimax.chat/v1"
        self._task_id: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _generate_signature(self, timestamp: int) -> str:
        """Generate API signature."""
        import hmac
        import hashlib

        message = f"{self.group_id}{timestamp}"
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()

    async def recognize(self, audio_chunk: bytes) -> ASRResult:
        """Recognize speech from a single audio chunk."""
        timestamp = int(time.time())
        signature = self._generate_signature(timestamp)

        url = f"{self._base_url}/audio/filler"

        headers = {
            "Authorization": f"Bearer; {signature}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        audio_base64 = base64.b64encode(audio_chunk).decode()

        payload = {
            "model": "filler-ease",
            "group_id": self.group_id,
            "timestamp": timestamp,
            "audio": audio_base64,
            "language": self.language,
            "sample_rate": self.sample_rate,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ASR API error: {response.status} - {error_text}")
                    return ASRResult(text="", is_final=False)

                result = await response.json()
                text = result.get("data", {}).get("text", "")
                is_final = result.get("data", {}).get("is_final", True)

                return ASRResult(
                    text=text,
                    is_final=is_final,
                    language=result.get("data", {}).get("language"),
                    confidence=result.get("data", {}).get("confidence", 0.0)
                )

        except asyncio.TimeoutError:
            logger.error("ASR request timeout")
            raise ASRError("ASR request timeout")
        except Exception as e:
            logger.error(f"ASR recognition error: {e}")
            raise ASRError(f"ASR recognition failed: {e}")

    async def stream_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """Process streaming audio and yield results."""
        buffer = b""

        async for chunk in audio_stream:
            buffer += chunk

            # Process when we have enough audio (e.g., 100ms)
            min_chunk_size = self.sample_rate * 2 * 0.1  # 100ms at 16kHz mono
            if len(buffer) >= min_chunk_size:
                try:
                    result = await self.recognize(buffer)
                    if result.text:
                        yield result
                    buffer = b""
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")

    async def close(self) -> None:
        """Close the ASR."""
        if self._session and not self._session.closed:
            await self._session.close()


class FasterWhisperASR(BaseASR):
    """Faster-Whisper ASR implementation (local)."""

    def __init__(
        self,
        model_name: str = "small",
        language: str = "auto",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self._model = None

    async def load(self) -> None:
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            logger.info(f"Faster-Whisper model '{self.model_name}' loaded")
        except ImportError:
            logger.error("faster-whisper not installed")
            raise ASRError("faster-whisper not installed")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ASRError(f"Failed to load Whisper model: {e}")

    async def recognize(self, audio_chunk: bytes) -> ASRResult:
        """Recognize speech from audio chunk."""
        if self._model is None:
            await self.load()

        try:
            import numpy as np
            from io import BytesIO
            import soundfile as sf

            # Convert bytes to numpy array
            audio_array, _ = sf.read(BytesIO(audio_chunk))

            segments, info = self._model.transcribe(
                audio_array,
                language=self.language if self.language != "auto" else None,
                beam_size=5,
                vad_filter=True
            )

            text = " ".join([segment.text for segment in segments])

            return ASRResult(
                text=text,
                is_final=True,
                language=info.language if info.language else self.language,
                confidence=info.language_probability
            )

        except Exception as e:
            logger.error(f"Whisper recognition error: {e}")
            raise ASRError(f"Whisper recognition failed: {e}")

    async def stream_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """Process streaming audio."""
        buffer = b""

        async for chunk in audio_stream:
            buffer += chunk

            # Process in chunks
            if len(buffer) > 16000 * 2:  # 1 second
                result = await self.recognize(buffer)
                if result.text:
                    yield result
                buffer = b""

    async def close(self) -> None:
        """Close the ASR."""
        self._model = None


async def create_asr(config: Optional[dict] = None) -> MinimaxASR:
    """Factory function to create ASR instance."""
    cfg = get_config()

    if config and config.get("model_name") == "faster-whisper":
        asr = FasterWhisperASR(
            model_name=config.get("model_name", "small"),
            language=config.get("language", "auto")
        )
        await asr.load()
        return asr
    else:
        # Default to Minimax
        return MinimaxASR(
            api_key=cfg.api.minimax_api_key,
            group_id=cfg.api.minimax_group_id,
            language=cfg.asr.language,
            sample_rate=cfg.asr.sample_rate
        )
