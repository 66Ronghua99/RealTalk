"""Text-to-Speech (TTS) module using Minimax API."""
import asyncio
import base64
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import numpy as np

from ..config import get_config
from ..exceptions import TTSError
from ..logging_config import setup_logger

logger = setup_logger("realtalk.tts")


@dataclass
class TTSResult:
    """TTS synthesis result."""
    audio: Optional[bytes]
    sample_rate: int
    is_final: bool
    text: str


class BaseTTS(ABC):
    """Base class for TTS implementations."""

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text."""
        pass

    @abstractmethod
    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """Synthesize speech with streaming."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop ongoing synthesis."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the TTS."""
        pass


class MinimaxTTS(BaseTTS):
    """Minimax TTS implementation with streaming support."""

    def __init__(
        self,
        api_key: str,
        group_id: str,
        voice_id: str = "male-qn-qingse",
        sample_rate: int = 32000
    ):
        self.api_key = api_key
        self.group_id = group_id
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = "https://api.minimax.chat/v1"
        self._current_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _generate_signature(self, timestamp: int) -> str:
        """Generate API signature."""
        message = f"{self.group_id}{timestamp}"
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text."""
        timestamp = int(time.time())
        signature = self._generate_signature(timestamp)

        url = f"{self._base_url}/t2a_v2"

        headers = {
            "Authorization": f"Bearer; {signature}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        payload = {
            "model": "speech-01-turbo",
            "group_id": self.group_id,
            "timestamp": timestamp,
            "text": text,
            "voice_setting": {
                "voice_id": self.voice_id,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "wav",
            },
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS API error: {response.status} - {error_text}")
                    raise TTSError(f"TTS API error: {response.status}")

                result = await response.json()

                if "data" in result and "audio" in result["data"]:
                    audio_base64 = result["data"]["audio"]
                    audio_data = base64.b64decode(audio_base64)

                    return TTSResult(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        is_final=True,
                        text=text
                    )
                else:
                    logger.warning("No audio in TTS response")
                    return TTSResult(
                        audio=None,
                        sample_rate=self.sample_rate,
                        is_final=True,
                        text=text
                    )

        except TTSError:
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise TTSError(f"TTS synthesis failed: {e}")

    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """Synthesize speech with streaming output."""
        timestamp = int(time.time())
        signature = self._generate_signature(timestamp)

        url = f"{self._base_url}/t2a_v2"

        headers = {
            "Authorization": f"Bearer; {signature}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": "speech-01-turbo",
            "group_id": self.group_id,
            "timestamp": timestamp,
            "text": text,
            "voice_setting": {
                "voice_id": self.voice_id,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "wav",
            },
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS stream error: {response.status} - {error_text}")
                    return

                # Read streaming response
                async for line in response.content:
                    if self._stop_event.is_set():
                        break

                    line = line.decode().strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "data" in data and "audio" in data["data"]:
                                audio_base64 = data["data"]["audio"]
                                audio_data = base64.b64decode(audio_base64)
                                is_final = data.get("is_final", False)

                                yield TTSResult(
                                    audio=audio_data,
                                    sample_rate=self.sample_rate,
                                    is_final=is_final,
                                    text=text
                                )
                        except json.JSONDecodeError:
                            continue

        except asyncio.CancelledError:
            logger.info("TTS stream cancelled")
        except Exception as e:
            logger.error(f"TTS stream error: {e}")
            raise TTSError(f"TTS stream failed: {e}")

    async def stop(self) -> None:
        """Stop ongoing synthesis."""
        self._stop_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        self._stop_event.clear()

    async def close(self) -> None:
        """Close the TTS."""
        await self.stop()
        if self._session and not self._session.closed:
            await self._session.close()


class EdgeTTS(BaseTTS):
    """Edge TTS implementation (alternative)."""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
        self._current_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, self.voice)
            audio_chunks = []

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            audio_data = b"".join(audio_chunks) if audio_chunks else None

            return TTSResult(
                audio=audio_data,
                sample_rate=48000,
                is_final=True,
                text=text
            )
        except ImportError:
            raise TTSError("edge-tts not installed")
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            raise TTSError(f"Edge TTS failed: {e}")

    async def stream_synthesize(self, text: str) -> AsyncIterator[TTSResult]:
        """Synthesize speech with streaming."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, self.voice)

            async for chunk in communicate.stream():
                if self._stop_event.is_set():
                    break

                if chunk["type"] == "audio":
                    yield TTSResult(
                        audio=chunk["data"],
                        sample_rate=48000,
                        is_final=False,
                        text=text
                    )

            yield TTSResult(
                audio=None,
                sample_rate=48000,
                is_final=True,
                text=text
            )

        except Exception as e:
            logger.error(f"Edge TTS stream error: {e}")
            raise TTSError(f"Edge TTS stream failed: {e}")

    async def stop(self) -> None:
        """Stop ongoing synthesis."""
        self._stop_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        self._stop_event.clear()

    async def close(self) -> None:
        """Close the TTS."""
        await self.stop()


async def create_tts(config: Optional[dict] = None) -> MinimaxTTS:
    """Factory function to create TTS instance."""
    cfg = get_config()

    if config and config.get("model_name") == "edge-tts":
        return EdgeTTS(voice=config.get("voice_id", "zh-CN-XiaoxiaoNeural"))
    else:
        # Default to Minimax
        return MinimaxTTS(
            api_key=cfg.api.minimax_api_key,
            group_id=cfg.api.minimax_group_id,
            voice_id=cfg.tts.voice_id,
            sample_rate=cfg.tts.sample_rate
        )
