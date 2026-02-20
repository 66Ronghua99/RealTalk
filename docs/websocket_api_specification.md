# RealTalk WebSocket API Specification

**Version:** 1.0.0
**Date:** 2026-02-20
**Protocol:** WebSocket (ws:// or wss://)
**Endpoint:** `/ws`

---

## Overview

This document specifies the WebSocket message protocol between RealTalk frontend clients and the backend server. All messages are JSON-encoded.

---

## Client → Server Messages

### 1. `text` - User Text Input

Sent when user types a message (alternative to voice input).

**Schema:**
```json
{
  "type": "text",
  "text": "string (required) - User input text"
}
```

**Example:**
```json
{
  "type": "text",
  "text": "Hello, how are you?"
}
```

**Response:** Server responds with `transcript`, `gatekeeper`, `llm_chunk`, `tts_audio`, and `state` messages.

---

### 2. `mic-audio-data` - Microphone Audio Chunk

Sent continuously while user is speaking. Audio data is transmitted as float array (inefficient - see notes).

**Schema:**
```json
{
  "type": "mic-audio-data",
  "audio": [0.0, 0.01, -0.005, ...]  // Array of float32 samples
}
```

**Notes:**
- Sample rate: 16kHz (configured but not enforced)
- Format: Float32 array (range: -1.0 to 1.0)
- Chunk size: ~4096 samples (depends on ScriptProcessorNode buffer size)
- **TODO:** Migrate to binary ArrayBuffer transmission for efficiency

---

### 3. `mic-audio-end` - End of User Speech

Sent when frontend VAD detects silence after speech.

**Schema:**
```json
{
  "type": "mic-audio-end"
}
```

**Behavior:** Server processes accumulated audio through ASR and triggers gatekeeper decision.

---

### 4. `audio_start` - User Started Speaking

Sent when frontend VAD detects speech start.

**Schema:**
```json
{
  "type": "audio_start"
}
```

**Behavior:** Server transitions to LISTENING state.

---

### 5. `stubbornness` - Update Stubbornness Level

Adjusts AI's interrupt resistance level.

**Schema:**
```json
{
  "type": "stubbornness",
  "level": 50  // Integer 0-100
}
```

**Levels:**
- 0-30: Polite mode - stop immediately on interrupt
- 30-70: Medium - ignore short non-command interruptions
- 70-100: Argument mode - ignore short interruptions, counter-respond

**Response:** Server sends `status` confirmation.

---

### 6. `interrupt` - Simulate User Interruption

Manually triggers an interruption (for testing).

**Schema:**
```json
{
  "type": "interrupt",
  "text": "User interrupted"  // Optional context
}
```

**Behavior:** Server stops TTS and transitions to INTERRUPTED state.

---

### 7. `clear` - Clear Accumulated Context

Clears the context accumulator.

**Schema:**
```json
{
  "type": "clear"
}
```

**Response:** Server sends `status` confirmation.

---

### 8. `audio` / `audio_end` (Legacy)

**DEPRECATED:** Legacy base64-encoded audio messages. Use `mic-audio-data` instead.

---

## Server → Client Messages

### 1. `transcript` - ASR Recognition Result

Sent when ASR successfully recognizes speech.

**Schema:**
```json
{
  "type": "transcript",
  "text": "Recognized speech text",
  "is_final": true
}
```

**UI Behavior:** Display as user message bubble.

---

### 2. `gatekeeper` - Gatekeeper Decision

Sent after ASR result with turn-taking decision.

**Schema:**
```json
{
  "type": "gatekeeper",
  "action": "wait|accumulate|interrupt|reply",
  "confidence": 0.85  // Float 0-1
}
```

**Actions:**
- `wait`: Continue listening
- `accumulate`: Add to context accumulator (high emotion scenario)
- `interrupt`: User interrupted AI speech
- `reply`: Generate AI response

**UI Behavior:** Update gatekeeper status indicator.

---

### 3. `llm_chunk` - Streaming LLM Response

Sent continuously as LLM generates response.

**Schema:**
```json
{
  "type": "llm_chunk",
  "text": "Partial response..."
}
```

**Behavior:** Multiple chunks form complete response. Final chunk has complete text.

**UI Behavior:** Update last assistant message with streaming text.

---

### 4. `tts_audio` - TTS Audio Chunk

Sent when TTS generates audio data.

**Schema:**
```json
{
  "type": "tts_audio",
  "audio": "base64-encoded-mp3-audio",
  "is_final": true  // Indicates last chunk of utterance
}
```

**Notes:**
- Format: MP3 (Minimax TTS)
- Sample rate: 32kHz (Minimax) or 48kHz (Edge TTS)
- **CRITICAL:** Frontend must accumulate all chunks until `is_final=true` before playing
- **TODO:** Audio is hex-encoded from Minimax, then converted to bytes, then base64-encoded for JSON

**UI Behavior:** Accumulate chunks, play combined audio when `is_final=true`.

---

### 5. `state` - State Machine Update

Sent when FSM state changes.

**Schema:**
```json
{
  "type": "state",
  "state": "idle|listening|processing|speaking|accumulating|interrupted"
}
```

**States:**
- `idle`: Waiting for user
- `listening`: User is speaking
- `processing`: Generating LLM response
- `speaking`: Playing TTS audio
- `accumulating`: Collecting context segments
- `interrupted`: User interrupted AI

**UI Behavior:** Update state indicator text/animation.

---

### 6. `status` - System Status Message

General status/information messages.

**Schema:**
```json
{
  "type": "status",
  "message": "Status message text"
}
```

**Common Messages:**
- "Accumulated for context"
- "Context cleared"
- "Stubbornness level: X"
- "No speech detected"
- "TTS error: ..."

**UI Behavior:** Display as transient system message.

---

### 7. `vad` - Voice Activity Detection Update

Sent during audio processing (legacy, primarily for debug).

**Schema:**
```json
{
  "type": "vad",
  "is_speaking": true,
  "energy": 0.45  // Float 0-1
}
```

**UI Behavior:** Change microphone button color (green when speaking).

---

## Message Flow Examples

### Normal Conversation Flow

```
Client                                    Server
  |                                         |
  |-------- mic-audio-data (continuous) --->|
  |-------- mic-audio-end ----------------->|
  |                                         | (ASR processing)
  |<------- transcript --------------------|
  |                                         | (Gatekeeper decision)
  |<------- gatekeeper (action: reply) ----|
  |                                         | (LLM streaming)
  |<------- llm_chunk --------------------|
  |<------- llm_chunk --------------------|
  |<------- llm_chunk --------------------|
  |                                         | (TTS synthesis)
  |<------- state (speaking) --------------|
  |<------- tts_audio (is_final=false) ---|
  |<------- tts_audio (is_final=true) ----|
  |<------- state (listening) ------------|
```

### Interruption Flow

```
Client                                    Server
  |                                         |
  |<------- state (speaking) --------------|
  |<------- tts_audio (streaming) --------|
  |                                         |
  |-------- interrupt -------------------->|
  |                                         | (Stop TTS)
  |<------- state (interrupted) -----------|
```

### Accumulation Flow (High Emotion)

```
Client                                    Server
  |-------- mic-audio-end ----------------->|
  |                                         | (Gatekeeper decides accumulate)
  |<------- gatekeeper (action: accumulate) |
  |<------- status (Accumulated...) -------|
  |                                         |
  |-------- mic-audio-data (more speech) -->|
  |-------- mic-audio-end ----------------->|
  |<------- transcript --------------------|
  |<------- gatekeeper (action: reply) ----|
```

---

## Known Issues

1. **Audio Transmission Inefficiency:** Audio sent as JSON float arrays. Should use binary WebSocket frames.

2. **TTS Race Condition:** `tts_audio` chunks can overlap if user interrupts and new response starts before previous audio finishes playing.

3. **No Deduplication:** Server can send duplicate `tts_audio` messages if multiple TTS tasks are created (orchestrator bug).

4. **State Synchronization:** No acknowledgment mechanism for state changes; client may miss updates during reconnection.

---

## Future Improvements

### API Version 2.0 Proposals

1. **Binary Audio Transmission**
   ```javascript
   // Use ArrayBuffer for audio data
   ws.send(audioBlob);  // Binary frame
   ```

2. **Message Acknowledgments**
   ```json
   {
     "type": "ack",
     "ref_id": "msg-uuid",
     "status": "received|processed|error"
   }
   ```

3. **Session Management**
   ```json
   {
     "type": "session_init",
     "session_id": "uuid",
     "config": {...}
   }
   ```

4. **Structured Errors**
   ```json
   {
     "type": "error",
     "code": "ASR_FAILED|LLM_TIMEOUT|TTS_ERROR",
     "message": "Human-readable description",
     "recoverable": true
   }
   ```

---

*This specification should be updated when message formats or behaviors change.*
