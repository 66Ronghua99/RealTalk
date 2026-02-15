"""Text-to-Speech (TTS) module using Minimax API."""
import asyncio
import base64
import json
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
        self._base_url = "https://api.minimaxi.com/v1"
        self._current_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        logger.info(f"MinimaxTTS initialized: group_id={group_id}, voice_id={voice_id}, sample_rate={sample_rate}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize(self, text: str) -> TTSResult:
        """Synthesize speech from text."""
        url = f"{self._base_url}/t2a_v2"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "speech-2.8-hd",
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            },
            "subtitle_enable": False,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS API error: {response.status} - {error_text}")
                    raise TTSError(f"TTS API error: {response.status}")

                result = await response.json()
                logger.info(f"TTS response keys: {result.keys()}")

                # Check for API error
                if "base_resp" in result and result["base_resp"].get("status_code") != 0:
                    error_msg = result["base_resp"].get("status_msg", "Unknown error")
                    logger.error(f"TTS API error: {error_msg}")
                    raise TTSError(f"TTS API error: {error_msg}")

                if "data" in result and "audio" in result["data"]:
                    audio_data = result["data"]["audio"]
                    # Note: API returns raw binary, not base64 encoded
                    if isinstance(audio_data, str):
                        audio_data = audio_data.encode('latin-1')  # Convert to bytes
                    logger.info(f"TTS audio field type: {type(audio_data)}, length: {len(audio_data) if audio_data else 0}")

                    # Debug: save audio to file
                    debug_path = "/Users/cory/codes/RealTalk/debug_tts.mp3"
                    with open(debug_path, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"TTS audio saved to {debug_path}, size: {len(audio_data)} bytes")

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
        url = f"{self._base_url}/t2a_v2"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "speech-2.8-hd",
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": 1,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            },
            "subtitle_enable": False,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"TTS stream error: {response.status} - {error_text}")
                    return

                logger.info(f"TTS API response status: {response.status}")
                logger.info(f"TTS Content-Type: {response.headers.get('Content-Type', 'unknown')}")

                # Read streaming response
                content_bytes = await response.read()
                logger.info(f"TTS response length: {len(content_bytes)} bytes")
                logger.info(f"TTS response (first 500): {content_bytes[:500]}")

                # Check if it's JSON instead of SSE
                try:
                    data = json.loads(content_bytes)
                    logger.info(f"TTS response is JSON: {data.keys()}")
                    # Handle non-streaming response
                    if "data" in data and "audio" in data["data"]:
                        audio_base64 = data["data"]["audio"]
                        audio_data = base64.b64decode(audio_base64)
                        yield TTSResult(
                            audio=audio_data,
                            sample_rate=self.sample_rate,
                            is_final=True,
                            text=text
                        )
                    return
                except json.JSONDecodeError:
                    pass

                # Parse as SSE
                for line_str in content_bytes.decode().split('\n'):
                    if self._stop_event.is_set():
                        break

                    line_str = line_str.strip()
                    if not line_str:
                        continue

                    logger.debug(f"TTS line: {line_str[:100]}...")

                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            logger.info("TTS stream done")
                            break

                        try:
                            data = json.loads(data_str)
                            logger.info(f"TTS data keys: {data.keys()}")

                            # Check for API error
                            if "base_resp" in data and data["base_resp"].get("status_code") != 0:
                                error_msg = data["base_resp"].get("status_msg", "Unknown error")
                                logger.error(f"TTS API error: {error_msg}")
                                raise TTSError(f"TTS API error: {error_msg}")

                            if "data" in data and "audio" in data["data"]:
                                audio_data = data["data"]["audio"]
                                # Note: API returns raw binary, not base64 encoded
                                if isinstance(audio_data, str):
                                    audio_data = audio_data.encode('latin-1')  # Convert to bytes
                                logger.info(f"TTS audio field type: {type(audio_data)}, length: {len(audio_data) if audio_data else 0}")
                                is_final = data.get("is_final", False)

                                yield TTSResult(
                                    audio=audio_data,
                                    sample_rate=self.sample_rate,
                                    is_final=is_final,
                                    text=text
                                )
                            else:
                                logger.warning(f"TTS data format unexpected: {data}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"TTS JSON decode error: {e}")
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
