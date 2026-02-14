"""WebRTC Transport Layer for RealTalk."""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

from ..config import get_config
from ..exceptions import TransportError
from ..logging_config import setup_logger

logger = setup_logger("realtalk.transport")


@dataclass
class AudioPacket:
    """Audio packet for transport."""
    audio: bytes
    sample_rate: int
    timestamp_ms: int


class BaseTransport(ABC):
    """Base class for transport implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the transport."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport."""
        pass

    @abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Send audio data."""
        pass

    @abstractmethod
    async def receive_audio(self) -> AsyncIterator[AudioPacket]:
        """Receive audio data."""
        pass

    @abstractmethod
    async def send_message(self, message: dict) -> None:
        """Send a control message."""
        pass


class WebRTCTransport(BaseTransport):
    """WebRTC transport implementation using DataChannel.

    This provides bidirectional, low-latency audio streaming
    required for real-time voice interaction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self._connected = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        """Connect via WebRTC.

        Note: In a real implementation, this would:
        1. Create an RTCPeerConnection
        2. Set up audio tracks
        3. Exchange SDP offers/answers
        4. Establish DataChannel for control messages
        """
        logger.info("WebRTC transport connecting...")
        # Placeholder - real implementation would use aiortc or similar
        self._connected = True
        logger.info("WebRTC transport connected")

    async def disconnect(self) -> None:
        """Disconnect from WebRTC."""
        logger.info("WebRTC transport disconnecting...")
        self._connected = False
        # Clear queues
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info("WebRTC transport disconnected")

    async def send_audio(self, audio: bytes) -> None:
        """Send audio data via WebRTC."""
        if not self._connected:
            raise TransportError("Not connected")

        # In real implementation, this would send via:
        # - AudioTrack for actual audio
        # - DataChannel for metadata
        pass

    async def receive_audio(self) -> AsyncIterator[AudioPacket]:
        """Receive audio data from WebRTC."""
        while self._connected:
            try:
                packet = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=1.0
                )
                yield packet
            except asyncio.TimeoutError:
                continue

    async def send_message(self, message: dict) -> None:
        """Send a control message via DataChannel."""
        if not self._connected:
            raise TransportError("Not connected")

        # In real implementation, this would send via DataChannel
        logger.debug(f"Sending message: {message}")


class MockTransport(BaseTransport):
    """Mock transport for testing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self._connected = False
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._mock_sending = False

    async def connect(self) -> None:
        """Connect (mock)."""
        self._connected = True
        logger.info("Mock transport connected")

    async def disconnect(self) -> None:
        """Disconnect (mock)."""
        self._connected = False
        logger.info("Mock transport disconnected")

    async def send_audio(self, audio: bytes) -> None:
        """Send audio (mock - just logs)."""
        if not self._connected:
            raise TransportError("Not connected")
        # In mock, we could loopback audio for testing
        # audio_array = np.frombuffer(audio, dtype=np.float32)
        # await self._audio_queue.put(AudioPacket(audio, self.sample_rate, 0))

    async def receive_audio(self) -> AsyncIterator[AudioPacket]:
        """Receive audio (mock)."""
        while self._connected:
            try:
                packet = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.1
                )
                yield packet
            except asyncio.TimeoutError:
                continue

    async def send_message(self, message: dict) -> None:
        """Send message (mock)."""
        logger.debug(f"Mock send: {message}")

    async def inject_audio(self, audio: bytes, timestamp_ms: int = 0) -> None:
        """Inject audio for testing."""
        packet = AudioPacket(
            audio=audio,
            sample_rate=self.sample_rate,
            timestamp_ms=timestamp_ms
        )
        await self._audio_queue.put(packet)


class WebSocketTransport(BaseTransport):
    """WebSocket transport as fallback (not recommended for production)."""

    def __init__(
        self,
        url: str,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        self.url = url
        self.sample_rate = sample_rate
        self.channels = channels
        self._connected = False
        self._ws: Optional[asyncio.WebSocketServer] = None

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        try:
            import aiohttp

            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.url)
            self._connected = True
            logger.info(f"WebSocket transport connected to {self.url}")
        except Exception as e:
            raise TransportError(f"WebSocket connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._connected = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("WebSocket transport disconnected")

    async def send_audio(self, audio: bytes) -> None:
        """Send audio via WebSocket."""
        if not self._connected:
            raise TransportError("Not connected")

        import base64

        audio_b64 = base64.b64encode(audio).decode()
        await self._ws.send_json({
            "type": "audio",
            "data": audio_b64,
            "sample_rate": self.sample_rate
        })

    async def receive_audio(self) -> AsyncIterator[AudioPacket]:
        """Receive audio from WebSocket."""
        while self._connected:
            try:
                msg = await asyncio.wait_for(
                    self._ws.receive(),
                    timeout=1.0
                )

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    if data.get("type") == "audio":
                        import base64

                        audio_b64 = data.get("data", "")
                        audio = base64.b64decode(audio_b64)
                        sample_rate = data.get("sample_rate", self.sample_rate)
                        timestamp_ms = data.get("timestamp", 0)

                        yield AudioPacket(
                            audio=audio,
                            sample_rate=sample_rate,
                            timestamp_ms=timestamp_ms
                        )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                break

    async def send_message(self, message: dict) -> None:
        """Send control message via WebSocket."""
        if not self._connected:
            raise TransportError("Not connected")

        await self._ws.send_json(message)


async def create_transport(
    transport_type: str = "webrtc",
    **kwargs
) -> BaseTransport:
    """Factory function to create transport."""
    if transport_type == "webrtc":
        return WebRTCTransport(**kwargs)
    elif transport_type == "mock":
        return MockTransport(**kwargs)
    elif transport_type == "websocket":
        return WebSocketTransport(**kwargs)
    else:
        raise TransportError(f"Unknown transport type: {transport_type}")
