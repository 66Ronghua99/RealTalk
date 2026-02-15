"""Simple web server for RealTalk testing."""
import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

import aiohttp
from aiohttp import web
import numpy as np

from ..cognition.llm import OpenRouterLLM
from ..cognition.tts import MinimaxTTS
from ..config import get_config
from ..logging_config import setup_logger
from ..orchestration.accumulator import ContextAccumulator, StubbornnessController
from ..orchestration.fsm import Event, State
from ..orchestration.gatekeeper import RuleBasedGatekeeper
from ..perception.vad import WebRTCVAD
from ..perception.asr import MinimaxASR

logger = setup_logger("realtalk.web")


class RealTalkWebHandler:
    """Web handler for RealTalk web interface."""

    def __init__(self):
        self._ws: Optional[web.WebSocketResponse] = None
        self._llm: Optional[OpenRouterLLM] = None
        self._tts: Optional[MinimaxTTS] = None
        self._vad: Optional[WebRTCVAD] = None
        self._asr: Optional[MinimaxASR] = None
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
        self._asr = MinimaxASR(
            api_key=cfg.api.minimax_api_key,
            group_id=cfg.api.minimax_group_id
        )
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
        """Handle incoming messages."""
        msg_type = data.get("type")

        if msg_type == "text":
            # User typed text - treat as user speech
            text = data.get("text", "")
            await self._process_text(text)

        elif msg_type == "stubbornness":
            # Update stubbornness level
            level = data.get("level", 50)
            self._stubbornness.level = level
            await self._send_to_client({
                "type": "status",
                "message": f"Stubbornness level: {level}"
            })

        elif msg_type == "interrupt":
            # Simulate user interruption
            await self._handle_interrupt(data.get("text", ""))

        elif msg_type == "clear":
            # Clear accumulator
            self._accumulator.clear()
            await self._send_to_client({
                "type": "status",
                "message": "Context cleared"
            })

        elif msg_type == "audio":
            # Process audio from microphone
            audio_data = data.get("data", "")
            await self._process_audio(audio_data)

        elif msg_type == "audio_start":
            # User started speaking
            self._current_state = State.LISTENING
            self._audio_buffer.clear()
            await self._send_to_client({"type": "state", "state": "listening"})

        elif msg_type == "audio_end":
            # User stopped speaking - process accumulated audio
            await self._process_audio_end()

    async def _process_text(self, text: str):
        """Process user text input."""
        self._current_text = text
        self._current_state = State.LISTENING

        await self._send_to_client({
            "type": "transcript",
            "text": text,
            "is_final": True
        })

        # Gatekeeper decision
        from ..orchestration.gatekeeper import GatekeeperInput
        gatekeeper_input = GatekeeperInput(
            text=text,
            silence_duration_ms=500,
            audio_energy=0.5,
            is_speaking=False
        )

        decision = await self._gatekeeper.decide(gatekeeper_input)

        await self._send_to_client({
            "type": "gatekeeper",
            "action": decision.action.value,
            "confidence": decision.confidence
        })

        if decision.action.value == "reply":
            await self._generate_response(text)
        elif decision.action.value == "accumulate":
            self._accumulator.add_segment(text)
            await self._send_to_client({
                "type": "status",
                "message": "Accumulated for context"
            })

    async def _process_audio(self, audio_base64: str):
        """Process incoming audio chunk from browser."""
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64)
            self._audio_buffer.append(audio_bytes)

            # Convert to numpy for VAD
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run VAD
            vad_result = await self._vad.detect(audio_array)

            await self._send_to_client({
                "type": "vad",
                "is_speaking": vad_result.is_speaking,
                "energy": vad_result.energy
            })
        except Exception as e:
            logger.error(f"Error processing audio: {e}")

    async def _process_audio_end(self):
        """Process accumulated audio when user stops speaking."""
        if not self._audio_buffer:
            return

        # Combine all audio chunks
        combined_audio = b"".join(self._audio_buffer)

        # Run ASR
        try:
            asr_result = await self._asr.recognize(combined_audio)

            if asr_result.text:
                await self._send_to_client({
                    "type": "transcript",
                    "text": asr_result.text,
                    "is_final": True
                })
                await self._process_text(asr_result.text)
            else:
                await self._send_to_client({
                    "type": "status",
                    "message": "No speech detected"
                })
        except Exception as e:
            logger.error(f"ASR error: {e}")
            await self._send_to_client({
                "type": "status",
                "message": f"ASR error: {e}"
            })
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
            {"role": "system", "content": "You are a helpful and conversational AI assistant."},
            {"role": "user", "content": context}
        ]

        # Stream LLM response
        response_text = ""
        async for chunk in self._llm.stream_chat(messages):
            response_text = chunk.content
            await self._send_to_client({
                "type": "llm_chunk",
                "text": response_text
            })

        # Synthesize TTS
        self._current_state = State.SPEAKING
        self._is_speaking = True
        await self._send_to_client({"type": "state", "state": "speaking"})

        async for tts_result in self._tts.stream_synthesize(response_text):
            if tts_result.audio:
                import base64
                audio_b64 = base64.b64encode(tts_result.audio).decode()
                await self._send_to_client({
                    "type": "tts_audio",
                    "audio": audio_b64
                })

        self._current_state = State.LISTENING
        self._is_speaking = False
        await self._send_to_client({"type": "state", "state": "listening"})

    async def _handle_interrupt(self, text: str):
        """Handle user interruption."""
        if not self._is_speaking:
            return

        # Check stubbornness
        should_ignore = self._stubbornness.should_ignore_interrupt(500, text)

        if should_ignore:
            self._accumulator.add_segment(text, is_interrupt=True)
            await self._send_to_client({
                "type": "status",
                "message": "Ignored (stubborn mode)"
            })
        else:
            await self._tts.stop()
            self._is_speaking = False
            self._current_state = State.INTERRUPTED
            await self._send_to_client({
                "type": "state",
                "state": "interrupted"
            })

    async def _send_to_client(self, data: dict):
        """Send message to WebSocket client."""
        if self._ws and not self._ws.closed:
            await self._ws.send_json(data)


async def create_app() -> web.Application:
    """Create aiohttp application."""
    handler = RealTalkWebHandler()
    await handler.init()

    app = web.Application()
    app["handler"] = handler

    # WebSocket endpoint
    app.router.add_get("/ws", handler.handle_websocket)

    # Static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.router.add_static("/static", static_path)

    # Index page
    async def index(request: web.Request) -> web.Response:
        html = get_index_html()
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", index)

    return app


def get_index_html() -> str:
    """Get the index HTML page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RealTalk - Voice Interaction</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; color: #00d9ff; }
        .status-bar {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        .status-item { text-align: center; }
        .status-label { font-size: 12px; color: #888; }
        .status-value { font-size: 18px; font-weight: bold; color: #00d9ff; }
        .chat-container {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            min-height: 300px;
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        .message.user {
            background: #007bff;
            margin-left: auto;
        }
        .message.assistant {
            background: #28a745;
        }
        .message.system {
            background: rgba(255,255,255,0.1);
            color: #888;
            font-size: 12px;
            text-align: center;
            max-width: 100%;
        }
        .input-area {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        input[type="text"]:focus { outline: 2px solid #00d9ff; }
        button {
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-send { background: #00d9ff; color: #000; }
        .btn-send:hover { background: #00b8d4; }
        .btn-interrupt { background: #ff4757; color: #fff; }
        .btn-interrupt:hover { background: #ff3344; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .control-group {
            flex: 1;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        .control-group label { display: block; margin-bottom: 10px; }
        input[type="range"] { width: 100%; }
        .stubbornness-value { color: #00d9ff; font-weight: bold; }
        .mic-button {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #ff4757;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 24px;
        }
        .mic-button.recording {
            background: #ff0000;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RealTalk - Voice Interaction</h1>

        <button class="mic-button" id="micBtn" onclick="toggleMic()">🎤</button>

        <div class="status-bar">
            <div class="status-item">
                <div class="status-label">State</div>
                <div class="status-value" id="state">IDLE</div>
            </div>
            <div class="status-item">
                <div class="status-label">Gatekeeper</div>
                <div class="status-value" id="gatekeeper">-</div>
            </div>
            <div class="status-item">
                <div class="status-label">Accumulated</div>
                <div class="status-value" id="accumulated">0</div>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Stubbornness Level: <span class="stubbornness-value" id="stubbornness-val">50</span></label>
                <input type="range" id="stubbornness" min="0" max="100" value="50">
            </div>
            <div class="control-group">
                <button class="btn-interrupt" onclick="interrupt()">Interrupt</button>
                <button onclick="clearContext()">Clear Context</button>
            </div>
        </div>

        <div class="chat-container" id="chat"></div>

        <div class="input-area">
            <input type="text" id="message" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button class="btn-send" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let ws = null;
        let accumulatedCount = 0;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                addSystemMessage('Connected to RealTalk');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                addSystemMessage('Disconnected');
                setTimeout(connect, 3000);
            };
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'transcript':
                    addMessage(data.text, 'user');
                    break;
                case 'llm_chunk':
                    // Could stream LLM response
                    break;
                case 'tts_audio':
                    // Play audio
                    if (data.audio) {
                        const audioData = atob(data.audio);
                        const arrayBuffer = new ArrayBuffer(audioData.length);
                        const uint8Array = new Uint8Array(arrayBuffer);
                        for (let i = 0; i < audioData.length; i++) {
                            uint8Array[i] = audioData.charCodeAt(i);
                        }
                        const audio = new Audio(URL.createObjectURL(new Blob([uint8Array], {type: 'audio/wav'})));
                        audio.play();
                    }
                    break;
                case 'state':
                    document.getElementById('state').textContent = data.state.toUpperCase();
                    if (data.state === 'speaking') {
                        addMessage(data.text || 'AI is speaking...', 'assistant');
                    }
                    break;
                case 'gatekeeper':
                    document.getElementById('gatekeeper').textContent = data.action.toUpperCase();
                    break;
                case 'status':
                    addSystemMessage(data.message);
                    if (data.message.includes('Accumulated')) {
                        accumulatedCount++;
                        document.getElementById('accumulated').textContent = accumulatedCount;
                    }
                    break;
                case 'vad':
                    // Update VAD status indicator
                    const micBtn = document.getElementById('micBtn');
                    if (data.is_speaking) {
                        micBtn.style.background = '#00ff00';
                    } else {
                        micBtn.style.background = '#ff4757';
                    }
                    break;
            }
        }

        function sendMessage() {
            const input = document.getElementById('message');
            const text = input.value.trim();
            if (!text || !ws) return;

            ws.send(JSON.stringify({type: 'text', text}));
            input.value = '';
        }

        function interrupt() {
            if (!ws) return;
            ws.send(JSON.stringify({type: 'interrupt', text: 'User interrupted'}));
        }

        function clearContext() {
            if (!ws) return;
            ws.send(JSON.stringify({type: 'clear'}));
            accumulatedCount = 0;
            document.getElementById('accumulated').textContent = '0';
        }

        function handleKeyPress(e) {
            if (e.key === 'Enter') sendMessage();
        }

        function addMessage(text, role) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addSystemMessage(text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message system';
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        // Stubbornness slider
        document.getElementById('stubbornness').addEventListener('input', (e) => {
            document.getElementById('stubbornness-val').textContent = e.target.value;
            if (ws) {
                ws.send(JSON.stringify({type: 'stubbornness', level: parseInt(e.target.value)}));
            }
        });

        // Microphone audio capture
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let audioContext = null;
        let analyser = null;
        let micStream = null;

        async function toggleMic() {
            const btn = document.getElementById('micBtn');

            if (!isRecording) {
                // Start recording with PCM capture and client-side VAD
                try {
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }});

                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioContext.createMediaStreamSource(micStream);

                    // Create script processor for PCM capture
                    const bufferSize = 4096;
                    const sampleRate = 16000;

                    // Use AudioWorklet if available, otherwise ScriptProcessor (deprecated but works)
                    if (audioContext.audioWorklet) {
                        // For simplicity, use ScriptProcessor here
                        const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

                        let silenceCount = 0;
                        const silenceThreshold = 0.01;
                        const silenceFramesNeeded = 30; // ~2 seconds of silence to stop
                        let isSpeaking = false;
                        let audioBuffer = [];

                        processor.onaudioprocess = (event) => {
                            const inputData = event.inputBuffer.getChannelData(0);
                            const outputData = event.outputBuffer.getChannelData(0);

                            // Copy input to output
                            outputData.set(inputData);

                            // Calculate RMS
                            let sum = 0;
                            for (let i = 0; i < inputData.length; i++) {
                                sum += inputData[i] * inputData[i];
                            }
                            const rms = Math.sqrt(sum / inputData.length);

                            // Convert to 16-bit PCM
                            const pcmData = new Int16Array(inputData.length);
                            for (let i = 0; i < inputData.length; i++) {
                                pcmData[i] = Math.max(-32768, Math.min(32767, Math.round(inputData[i] * 32767)));
                            }

                            // Send audio to server
                            const bytes = new Uint8Array(pcmData.buffer);
                            let binary = '';
                            for (let i = 0; i < bytes.length; i++) {
                                binary += String.fromCharCode(bytes[i]);
                            }
                            const base64 = btoa(binary);

                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({
                                    type: 'audio',
                                    data: base64,
                                    isSpeaking: rms > silenceThreshold
                                }));
                            }

                            // VAD: detect silence
                            if (rms > silenceThreshold) {
                                isSpeaking = true;
                                silenceCount = 0;
                                btn.style.background = '#00ff00';
                            } else {
                                silenceCount++;
                                if (isSpeaking && silenceCount > silenceFramesNeeded) {
                                    // Auto-stop recording after silence
                                    stopRecording();
                                }
                                btn.style.background = '#ff4757';
                            }
                        };

                        source.connect(processor);
                        processor.connect(audioContext.destination);

                        // Store for cleanup
                        mediaRecorder = processor;
                        isRecording = true;
                        btn.classList.add('recording');
                        btn.textContent = '⏹';

                        // Notify server
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({type: 'audio_start'}));
                        }

                    } else {
                        // Fallback to MediaRecorder
                        mediaRecorder = new MediaRecorder(micStream);
                        audioChunks = [];

                        mediaRecorder.ondataavailable = (event) => {
                            if (event.data.size > 0) {
                                audioChunks.push(event.data);
                                const reader = new FileReader();
                                reader.onload = () => {
                                    const arrayBuffer = reader.result;
                                    const uint8Array = new Uint8Array(arrayBuffer);
                                    let binary = '';
                                    for (let i = 0; i < uint8Array.length; i++) {
                                        binary += String.fromCharCode(uint8Array[i]);
                                    }
                                    const base64 = btoa(binary);
                                    if (ws && ws.readyState === WebSocket.OPEN) {
                                        ws.send(JSON.stringify({type: 'audio', data: base64}));
                                    }
                                };
                                reader.readAsArrayBuffer(event.data);
                            }
                        };

                        mediaRecorder.onstop = () => {
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({type: 'audio_end'}));
                            }
                        };

                        mediaRecorder.start(100);
                        isRecording = true;
                        btn.classList.add('recording');
                        btn.textContent = '⏹';

                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({type: 'audio_start'}));
                        }
                    }

                } catch (err) {
                    addSystemMessage('Microphone error: ' + err.message);
                }
            } else {
                stopRecording();
            }
        }

        function stopRecording() {
            const btn = document.getElementById('micBtn');

            if (micStream) {
                micStream.getTracks().forEach(track => track.stop());
                micStream = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (mediaRecorder) {
                if (mediaRecorder instanceof AudioWorkletNode || mediaRecorder.onaudioprocess) {
                    // Disconnect script processor
                    mediaRecorder.disconnect();
                } else if (mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                mediaRecorder = null;
            }

            isRecording = false;
            btn.classList.remove('recording');
            btn.textContent = '🎤';
            btn.style.background = '#ff4757';

            // Notify server recording ended
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'audio_end'}));
            }
        }

        connect();
    </script>
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
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
