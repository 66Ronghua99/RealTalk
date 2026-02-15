"""Streaming ASR (Automatic Speech Recognition) module using Minimax API."""
import asyncio
import base64
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import aiohttp
import numpy as np

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
                logger.info(f"ASR API response: {result}")
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


class SherpaOnnxASR(BaseASR):
    """Sherpa-ONNX ASR implementation using SenseVoice (local, fast)."""

    def __init__(
        self,
        num_threads: int = 4,
        sample_rate: int = 16000,
        use_itn: bool = True,
    ):
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.use_itn = use_itn
        self._recognizer = None
        self._stream = None

    async def load(self) -> None:
        """Load the Sherpa-ONNX SenseVoice ASR model."""
        try:
            import sherpa_onnx

            # Download model if not exists
            model_dir = Path.home() / ".cache" / "realtalk" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            # SenseVoice model files
            sense_voice_model = model_dir / "sense-voice" / "model.onnx"
            tokens_file = model_dir / "sense-voice" / "tokens.txt"

            if not sense_voice_model.exists():
                logger.info("Downloading SenseVoice model...")
                import urllib.request
                import tarfile
                import io

                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"

                response = urllib.request.urlopen(url)
                tar_data = io.BytesIO(response.read())
                with tarfile.open(fileobj=tar_data) as tar:
                    for member in tar.getmembers():
                        if "model.onnx" in member.name:
                            sense_voice_model.parent.mkdir(parents=True, exist_ok=True)
                            sense_voice_model.write_bytes(tar.extractfile(member).read())
                        if member.name.endswith("tokens.txt"):
                            tokens_file.parent.mkdir(parents=True, exist_ok=True)
                            tokens_file.write_bytes(tar.extractfile(member).read())

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(sense_voice_model),
                tokens=str(tokens_file),
                num_threads=self.num_threads,
                use_itn=self.use_itn,
            )
            logger.info("Sherpa-ONNX SenseVoice ASR loaded")
        except Exception as e:
            logger.error(f"Failed to load Sherpa-ONNX: {e}")
            raise ASRError(f"Failed to load Sherpa-ONNX: {e}")

    async def recognize(self, audio_chunk: bytes) -> ASRResult:
        """Recognize speech from audio chunk."""
        if self._recognizer is None:
            await self.load()

        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

            # Create stream and decode
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_array)
            self._recognizer.decode_streams([stream])

            text = stream.result.text
            return ASRResult(
                text=text,
                is_final=True,
                language="auto",
                confidence=1.0 if text else 0.0
            )
        except Exception as e:
            logger.error(f"Sherpa-ONNX recognition error: {e}")
            raise ASRError(f"Recognition failed: {e}")

    async def stream_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """Process streaming audio."""
        buffer = b""
        async for chunk in audio_stream:
            buffer += chunk
            # Process every 1 second of audio
            if len(buffer) >= self.sample_rate * 2:
                result = await self.recognize(buffer)
                if result.text:
                    yield result
                buffer = b""

    async def close(self) -> None:
        """Close the ASR."""
        self._recognizer = None
        self._stream = None


async def create_asr(config: Optional[dict] = None) -> SherpaOnnxASR:
    """Factory function to create ASR instance."""
    if config and config.get("model_name") == "minimax":
        asr = MinimaxASR(
            api_key=config.get("api_key", ""),
            group_id=config.get("group_id", ""),
            language=config.get("language", "auto"),
            sample_rate=config.get("sample_rate", 16000)
        )
        return asr
    else:
        # Default to Sherpa-ONNX SenseVoice (local, fast)
        asr = SherpaOnnxASR(
            num_threads=4,
            sample_rate=16000,
            use_itn=True
        )
        await asr.load()
        return asr
