"""Simple web server for RealTalk testing."""
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional, Union

import aiohttp
from aiohttp import web
import numpy as np

from ..cognition.llm import OpenRouterLLM, Message as LLMMessage
from ..cognition.tts import MinimaxTTS
from ..config import get_config
from ..logging_config import setup_logger
from ..messages import (
    AudioStart,
    ClearContext,
    GatekeeperDecision,
    InterruptRequest,
    LLMChunk,
    MicAudioData,
    MicAudioEnd,
    ServerMessage,
    State as StateEnum,
    StateUpdate,
    StatusMessage,
    StubbornnessUpdate,
    TextInput,
    Transcript,
    TTSAudio,
    deserialize_message,
)
from ..orchestration.accumulator import ContextAccumulator, StubbornnessController
from ..orchestration.fsm import Event, State
from ..orchestration.gatekeeper import RuleBasedGatekeeper
from ..perception.vad import WebRTCVAD
from ..perception.asr import SherpaOnnxASR

logger = setup_logger("realtalk.web")


class RealTalkWebHandler:
    """Web handler for RealTalk web interface."""

    def __init__(self):
        self._ws: Optional[web.WebSocketResponse] = None
        self._llm: Optional[OpenRouterLLM] = None
        self._tts: Optional[MinimaxTTS] = None
        self._vad: Optional[WebRTCVAD] = None
        self._asr: Optional[SherpaOnnxASR] = None
        self._gatekeeper: Optional[RuleBasedGatekeeper] = None
        self._accumulator: Optional[ContextAccumulator] = None
        self._stubbornness: Optional[StubbornnessController] = None

        self._current_state = State.IDLE
        self._current_text = ""
        self._is_speaking = False
        self._audio_buffer: list = []

    async def init(self):
        """Initialize components."""
        cfg = get_config()

        self._llm = OpenRouterLLM(
            api_key=cfg.api.openrouter_api_key,
            model_name=cfg.llm.model_name
        )
        self._tts = MinimaxTTS(
            api_key=cfg.api.minimax_api_key,
            group_id=cfg.api.minimax_group_id
        )
        self._vad = WebRTCVAD()
        self._asr = SherpaOnnxASR(
            num_threads=4,
            sample_rate=16000,
            use_itn=True
        )
        await self._asr.load()
        self._gatekeeper = RuleBasedGatekeeper()
        self._accumulator = ContextAccumulator()
        self._stubbornness = StubbornnessController(level=cfg.orchestration.stubbornness_level)

        logger.info("Web handler initialized")

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws = ws

        logger.info("WebSocket client connected")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws = None
            logger.info("WebSocket client disconnected")

        return ws

    async def _handle_message(self, data: dict):
        """Handle incoming messages using Pydantic models."""
        try:
            message = deserialize_message(data)
        except ValueError as e:
            logger.error(f"Failed to deserialize message: {e}")
            return

        # Handle using validated Pydantic models
        if isinstance(message, TextInput):
            await self._process_text(message.text)

        elif isinstance(message, StubbornnessUpdate):
            self._stubbornness.level = message.level
            await self._send_to_client(
                StatusMessage(message=f"Stubbornness level: {message.level}")
            )

        elif isinstance(message, InterruptRequest):
            await self._handle_interrupt(message.text or "")

        elif isinstance(message, ClearContext):
            self._accumulator.clear()
            await self._send_to_client(StatusMessage(message="Context cleared"))

        elif isinstance(message, MicAudioData):
            await self._process_audio_float(message.audio)

        elif isinstance(message, MicAudioEnd):
            logger.info("Received mic-audio-end")
            await self._process_audio_end()

        elif isinstance(message, AudioStart):
            logger.info("Received audio_start")
            self._current_state = State.LISTENING
            self._audio_buffer.clear()
            await self._send_to_client(
                StateUpdate(state=StateEnum.LISTENING)
            )

    async def _process_text(self, text: str):
        """Process user text input."""
        self._current_text = text
        self._current_state = State.LISTENING

        await self._send_to_client(
            Transcript(text=text, is_final=True)
        )

        # Gatekeeper decision
        from ..orchestration.gatekeeper import GatekeeperInput
        gatekeeper_input = GatekeeperInput(
            text=text,
            silence_duration_ms=500,
            audio_energy=0.5,
            is_speaking=False
        )

        decision = await self._gatekeeper.decide(gatekeeper_input)

        await self._send_to_client(
            GatekeeperDecision(
                action=decision.action.value,
                confidence=decision.confidence
            )
        )

        if decision.action.value == "reply":
            await self._generate_response(text)
        elif decision.action.value == "accumulate":
            self._accumulator.add_segment(text)
            await self._send_to_client(
                StatusMessage(message="Accumulated for context")
            )

    async def _process_audio(self, audio_base64: str):
        """Process incoming audio chunk from browser (legacy base64 format)."""
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            # logger.info(f"Decoded audio chunk: {len(audio_bytes)} bytes, buffer size now: {len(self._audio_buffer) + 1}")
            self._audio_buffer.append(audio_bytes)

            # Convert to numpy for VAD
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run VAD
            vad_result = await self._vad.detect(audio_array)

            await self._send_to_client({
                "type": "vad",
                "is_speaking": vad_result.is_speech,
                "energy": vad_result.confidence
            })
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    async def _process_audio_float(self, audio_list: list):
        """Process incoming audio chunk from browser (float array from frontend VAD)."""
        try:
            # Convert float array to numpy and then to bytes
            audio_array = np.array(audio_list, dtype=np.float32)
            # Clip values to prevent overflow/distortion (P0 fix)
            audio_clipped = np.clip(audio_array, -1.0, 1.0)
            audio_bytes = (audio_clipped * 32767).astype(np.int16).tobytes()

            # logger.info(f"Processed float audio: {len(audio_bytes)} bytes, buffer size now: {len(self._audio_buffer) + 1}")
            self._audio_buffer.append(audio_bytes)

            # Frontend VAD handles speech detection, no need to run backend VAD
        except Exception as e:
            logger.error(f"Error processing float audio: {e}")

    async def _process_audio_end(self):
        """Process accumulated audio when user stops speaking."""
        if not self._audio_buffer:
            logger.warning("No audio buffer to process")
            await self._send_to_client(
                StatusMessage(message="No audio recorded")
            )
            return

        # Combine all audio chunks
        combined_audio = b"".join(self._audio_buffer)
        logger.info(f"Processing audio: {len(combined_audio)} bytes, {len(self._audio_buffer)} chunks")

        # Run ASR
        try:
            asr_result = await self._asr.recognize(combined_audio)
            logger.info(f"ASR result: '{asr_result.text}', is_final={asr_result.is_final}")

            if asr_result.text:
                await self._process_text(asr_result.text)
            else:
                await self._send_to_client(
                    StatusMessage(message="No speech detected")
                )
        except Exception as e:
            logger.error(f"ASR error: {e}")
            await self._send_to_client(
                StatusMessage(message=f"ASR error: {e}")
            )
        finally:
            self._audio_buffer.clear()

    async def _generate_response(self, text: str):
        """Generate LLM response and send TTS."""
        # Check accumulator first
        if len(self._accumulator) > 0:
            context = self._accumulator.get_combined_text()
            self._accumulator.clear()
        else:
            context = text

        self._current_state = State.PROCESSING
        await self._send_to_client({"type": "state", "state": "processing"})

        # Build messages
        messages = [
            LLMMessage(role="system", content="You are a helpful and conversational AI assistant."),
            LLMMessage(role="user", content=context)
        ]

        # Stream LLM response
        response_text = ""
        async for chunk in self._llm.stream_chat(messages):
            response_text = chunk.content
            await self._send_to_client(
                LLMChunk(text=response_text)
            )

        # Synthesize TTS
        self._current_state = State.SPEAKING
        self._is_speaking = True
        await self._send_to_client(
            StateUpdate(state=StateEnum.SPEAKING)
        )

        logger.info(f"Starting TTS for text: {response_text[:50]}...")
        try:
            chunk_count = 0
            async for tts_result in self._tts.stream_synthesize(response_text):
                logger.info(f"TTS result: audio={tts_result.audio is not None}, is_final={tts_result.is_final}")
                if tts_result.audio:
                    chunk_count += 1
                    logger.info(f"TTS audio chunk {chunk_count}: {len(tts_result.audio)} bytes")
                    audio_b64 = base64.b64encode(tts_result.audio).decode()
                    await self._send_to_client(
                        TTSAudio(
                            audio=audio_b64,
                            is_final=tts_result.is_final,
                            chunk_index=chunk_count - 1
                        )
                    )
            logger.info(f"TTS complete, total chunks: {chunk_count}")
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            await self._send_to_client(
                StatusMessage(message=f"TTS error: {e}")
            )

        self._current_state = State.LISTENING
        self._is_speaking = False
        await self._send_to_client(
            StateUpdate(state=StateEnum.LISTENING)
        )

    async def _handle_interrupt(self, text: str):
        """Handle user interruption."""
        if not self._is_speaking:
            return

        # Check stubbornness
        should_ignore = self._stubbornness.should_ignore_interrupt(500, text)

        if should_ignore:
            self._accumulator.add_segment(text, is_interrupt=True)
            await self._send_to_client(
                StatusMessage(message="Ignored (stubborn mode)")
            )
        else:
            await self._tts.stop()
            self._is_speaking = False
            self._current_state = State.INTERRUPTED
            await self._send_to_client(
                StateUpdate(state=StateEnum.INTERRUPTED)
            )

    async def _send_to_client(self, message: Union[ServerMessage, dict]):
        """Send message to WebSocket client.

        Args:
            message: Either a Pydantic ServerMessage model or legacy dict
        """
        if not self._ws or self._ws.closed:
            return

        if isinstance(message, ServerMessage):
            # Add timestamp before sending
            message.timestamp = int(time.time() * 1000)
            await self._ws.send_json(message.to_dict())
        else:
            # Legacy dict support (for backwards compatibility)
            await self._ws.send_json(message)


async def create_app() -> web.Application:
    """Create aiohttp application (async version for manual use)."""
    handler = RealTalkWebHandler()
    await handler.init()

    app = web.Application()
    app["handler"] = handler

    # WebSocket endpoint
    app.router.add_get("/ws", handler.handle_websocket)

    # Static files (CSS, JS)
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static", static_path)

    # Index page - serve from template file
    async def index(request: web.Request) -> web.Response:
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            html = template_path.read_text()
        else:
            # Fallback to embedded HTML for backwards compatibility
            html = get_index_html()
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", index)

    return app


def create_app_sync() -> web.Application:
    """Create aiohttp application (sync version for uvicorn)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        app = loop.run_until_complete(create_app())
    finally:
        loop.close()
    return app


def get_index_html() -> str:
    """Get the index HTML page (fallback when template file doesn't exist)."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text()
    # Minimal fallback HTML
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RealTalk - Voice Interaction</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <h1>RealTalk - Voice Interaction</h1>
        <p>Error: Templates not found. Please ensure templates/index.html exists.</p>
    </div>
</body>
</html>
"""


async def run_server(host: str = "localhost", port: int = 8080):
    """Run the web server."""
    app = await create_app()
    logger.info(f"Starting server on {host}:{port}")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info(f"Server running at http://{host}:{port}")

    # Keep server running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Server shutting down")
        await runner.cleanup()


def main():
    """CLI entry point."""
    asyncio.run(run_server(), debug=True)


if __name__ == "__main__":
    main()
