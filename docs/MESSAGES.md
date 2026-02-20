# RealTalk Message Protocol (v1.0.0)

This document describes the WebSocket message protocol used by RealTalk for real-time voice interaction.

## Overview

The protocol uses JSON messages over WebSocket. All messages include:
- `type`: Message type discriminator (required)
- `api_version`: API version string (default: "1.0.0")
- `timestamp`: Unix timestamp in milliseconds (optional, added by server)

## Message Types

### Client → Server

#### `text` - User Text Input
Sent when user types a message in the text input field.

```json
{
  "type": "text",
  "text": "Hello, how are you?"
}
```

**Fields:**
- `text` (string, required): User input text (min 1 character)

---

#### `stubbornness` - Update Stubbornness Level
Sent when user adjusts the stubbornness slider.

```json
{
  "type": "stubbornness",
  "level": 50
}
```

**Fields:**
- `level` (integer, required): 0-100 (0=polite, 100=argumentative)

---

#### `interrupt` - User Interruption
Sent when user interrupts AI speech.

```json
{
  "type": "interrupt",
  "text": "Wait, that's not right"
}
```

**Fields:**
- `text` (string, optional): Interruption text/keywords

---

#### `clear` - Clear Context
Sent when user wants to clear accumulated context.

```json
{
  "type": "clear"
}
```

---

#### `mic-audio-data` - Audio Data
Sent continuously while user is speaking (frontend VAD).

```json
{
  "type": "mic-audio-data",
  "audio": [0.0, 0.01, -0.02, ...]
}
```

**Fields:**
- `audio` (array of float, required): Audio samples (-1.0 to 1.0)

---

#### `mic-audio-end` - End of Speech
Sent when frontend VAD detects silence.

```json
{
  "type": "mic-audio-end"
}
```

---

#### `audio_start` - Start of Speech
Sent when frontend VAD detects speech.

```json
{
  "type": "audio_start"
}
```

### Server → Client

#### `transcript` - ASR Transcript
Sent when ASR produces transcription.

```json
{
  "type": "transcript",
  "text": "Hello world",
  "is_final": true
}
```

**Fields:**
- `text` (string, required): Transcribed text
- `is_final` (boolean): Always true for this implementation

---

#### `llm_chunk` - LLM Streaming Response
Sent as streaming chunks from the LLM.

```json
{
  "type": "llm_chunk",
  "text": "Hello! I'm doing well",
  "is_final": false
}
```

**Fields:**
- `text` (string, required): Current accumulated response text
- `is_final` (boolean, optional): Whether this is the final chunk

---

#### `tts_audio` - TTS Audio Chunk
Sent as audio chunks are generated.

```json
{
  "type": "tts_audio",
  "audio": "//NEx...",
  "is_final": false,
  "chunk_index": 0
}
```

**Fields:**
- `audio` (string, required): Base64-encoded MP3 data
- `is_final` (boolean, required): Whether this is the final chunk
- `chunk_index` (integer, optional): 0-based chunk index

---

#### `state` - FSM State Update
Sent when the orchestration state changes.

```json
{
  "type": "state",
  "state": "listening"
}
```

**States:**
- `idle`: Not actively listening or speaking
- `listening`: Waiting for user input
- `processing`: Generating LLM response
- `speaking`: Playing TTS audio
- `interrupted`: AI was interrupted
- `accumulating`: Collecting multi-segment context

---

#### `gatekeeper` - Gatekeeper Decision
Sent when the gatekeeper makes a decision.

```json
{
  "type": "gatekeeper",
  "action": "reply",
  "confidence": 0.85
}
```

**Fields:**
- `action` (enum): `wait`, `accumulate`, `interrupt`, `reply`
- `confidence` (float): 0.0 to 1.0

---

#### `status` - Status/Notification
Sent for various status updates.

```json
{
  "type": "status",
  "message": "Context cleared"
}
```

**Fields:**
- `message` (string, required): Status message text

---

#### `vad` - Voice Activity Detection
Sent during audio processing (legacy mode).

```json
{
  "type": "vad",
  "is_speaking": true,
  "energy": 0.75
}
```

**Fields:**
- `is_speaking` (boolean): Whether speech is detected
- `energy` (float): Audio energy level (0.0 to 1.0)

## Python API

### Using Pydantic Models

```python
from realtalk.messages import TextInput, StateUpdate, deserialize_message

# Create a message
msg = TextInput(text="Hello")

# Serialize to dict
data = msg.to_dict()  # {"type": "text", "text": "Hello", ...}

# Serialize to JSON
json_str = msg.to_json()

# Deserialize
msg = deserialize_message({"type": "text", "text": "Hello"})
```

### Message Bus Pattern

```python
from realtalk.web.message_bus import create_message_bus
from realtalk.messages import TextInput

bus = create_message_bus()

@bus.subscribe(MessageType.TEXT)
async def handle_text(msg: TextInput) -> None:
    print(f"Received: {msg.text}")

# Publish a message
await bus.publish(TextInput(text="Hello"))
```

## JSON Schema

Generate JSON Schemas for all message types:

```python
from realtalk.messages import get_all_schemas
import json

schemas = get_all_schemas()
print(json.dumps(schemas, indent=2))
```

## Migration Notes

### From Legacy Protocol

1. **Removed legacy messages**: `audio`, `audio_end` (base64 audio)
2. **Use `mic-audio-data`**: Float array audio instead of base64
3. **Validation is stricter**: Invalid messages are now rejected with validation errors
4. **Timestamps**: Server now adds timestamps automatically to outgoing messages

## Version History

- **v1.0.0** (2026-02-20): Initial formalized protocol with Pydantic models
