# RealTalk Audio Pipeline Architecture

**Version:** 1.0.0
**Date:** 2026-02-20

---

## Overview

This document describes the audio data flow through the RealTalk system, including sample rates, formats, and conversion points.

---

## Audio Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (FRONTEND)                              │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ getUserMedia API │───▶│  AudioContext │───▶│  ScriptProcessorNode     │   │
│  │                  │    │  (16kHz*)    │    │  (4096 sample buffer)    │   │
│  └──────────────────┘    └──────────────┘    └──────────────────────────┘   │
│                                                         │                     │
│                                                         ▼                     │
│                                              ┌────────────────────┐         │
│                                              │  Frontend VAD      │         │
│                                              │  (RMS threshold:   │         │
│                                              │   0.01)            │         │
│                                              └────────────────────┘         │
│                                                         │                     │
│                                                         ▼                     │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  JSON.stringify({type: 'mic-audio-data', audio: [0.0, 0.01, ...]})    ║  │
│  ║  ⚠️ INEFFICIENT: Float array → String → JSON → WebSocket            ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                         │                     │
└─────────────────────────────────────────────────────────┼─────────────────────┘
                                                          │ WebSocket
                                                          ▼
┌─────────────────────────────────────────────────────────┼─────────────────────┐
│                           BACKEND                        │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  WebSocket Handler: _process_audio_float()                            │    │
│  │  Location: src/realtalk/web/server.py:208-220                         │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│  ⚠️ CONVERSION 1: JSON array → Float32Array → Int16 (no clipping!)           │
│     audio_array = np.array(audio_list, dtype=np.float32)                      │
│     audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()            │
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  Audio Buffer Accumulation                                            │    │
│  │  Location: src/realtalk/web/server.py:40, 216                         │    │
│  │  Buffer: List[bytes] - accumulates until mic-audio-end                │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  On mic-audio-end: _process_audio_end()                               │    │
│  │  Location: src/realtalk/web/server.py:222-255                         │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  ASR: SherpaOnnxASR (default) or MinimaxASR                          │    │
│  │  Location: src/realtalk/perception/asr.py                             │    │
│  │  Input: Int16 bytes, 16kHz                                            │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│  ⚠️ CONVERSION 2 (SherpaOnnxASR): Int16 → Float32 (-1.0 to 1.0)              │
│     audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  ASR Recognition Result                                               │    │
│  │  Output: Text string                                                  │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│                             [ LLM Processing ]                               │
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐    │
│  │  TTS: MinimaxTTS (default) or EdgeTTS                                │    │
│  │  Location: src/realtalk/cognition/tts.py                              │    │
│  │  Input: Text string                                                   │    │
│  │  Output: MP3 audio (32kHz for Minimax, 48kHz for Edge)               │    │
│  └───────────────────────────────────────────────────────────────────────┘    │
│                                                          │                     │
│  ⚠️ CONVERSION 3 (Minimax): Hex string → bytes → base64 for JSON             │
│     Minimax returns: {"audio": "48656c6c6f..."}  (hex-encoded)               │
│     Server: bytes.fromhex(audio_hex) → base64.b64encode() → JSON             │
│                                                          │                     │
│                                                          ▼                     │
│  ╔═══════════════════════════════════════════════════════════════════════╗    │
│  ║  JSON: {type: 'tts_audio', audio: 'base64-mp3', is_final: true}      ║    │
│  ╚═══════════════════════════════════════════════════════════════════════╝    │
│                                                          │                     │
└─────────────────────────────────────────────────────────┼─────────────────────┘
                                                          │ WebSocket
                                                          ▼
┌─────────────────────────────────────────────────────────┼─────────────────────┐
│                              BROWSER (FRONTEND)          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  handleMessage(): tts_audio case                                      │   │
│  │  Location: src/realtalk/web/server.py:587-658 (JS in HTML)            │   │
│  │  Accumulates chunks in audioChunks[] array                            │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                          │                     │
│  ⚠️ CONVERSION 4: base64 → Uint8Array → Blob → Audio element                 │
│     const binaryString = atob(data.audio);                                    │
│     const bytes = new Uint8Array(binaryString.length);                        │
│     for (let i = 0; i < binaryString.length; i++) {                           │
│         bytes[i] = binaryString.charCodeAt(i);                                │
│     }                                                                         │
│     const blob = new Blob([combined], {type: 'audio/mpeg'});                  │
│     const audio = new Audio(URL.createObjectURL(blob));                       │
│                                                          │                     │
│                                                          ▼                     │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  Audio Playback                                                       │   │
│  │  Sample rate: 32kHz (Minimax) or 48kHz (Edge)                        │   │
│  │  Format: MP3                                                          │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

* Browser may ignore 16kHz constraint and provide 48kHz audio
```

---

## Sample Rate Summary

| Stage | Component | Sample Rate | Format | Notes |
|-------|-----------|-------------|--------|-------|
| Capture | getUserMedia | 16kHz (requested) | Float32 | Browser may provide 48kHz |
| Frontend Processing | AudioContext | 16kHz | Float32 | Assumes input is 16kHz |
| Transmission | WebSocket | - | JSON float array | ⚠️ Inefficient |
| Backend Reception | _process_audio_float | 16kHz (assumed) | Int16 | No validation |
| ASR Input | SherpaOnnxASR | 16kHz | Float32 | Converts Int16 → Float32 |
| ASR Model | SenseVoice | 16kHz | Float32 | Internal processing |
| TTS Output | Minimax | 32kHz | MP3 | Hex-encoded in API |
| TTS Output | EdgeTTS | 48kHz | MP3 | Direct stream |
| Playback | Audio element | 32kHz or 48kHz | MP3 | Browser handles decoding |

---

## Format Conversion Details

### Conversion 1: Frontend Float → Backend Int16
**Location:** `src/realtalk/web/server.py:211-213`

```python
# Input: Float32 array from frontend (-1.0 to 1.0)
audio_array = np.array(audio_list, dtype=np.float32)

# ⚠️ NO CLIPPING PROTECTION!
# If values > 1.0 or < -1.0, overflow occurs
audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
```

**Issue:** Microphone gain can produce values outside [-1.0, 1.0], causing distortion.

**Fix:**
```python
audio_clipped = np.clip(audio_array, -1.0, 1.0)
audio_bytes = (audio_clipped * 32767).astype(np.int16).tobytes()
```

---

### Conversion 2: Int16 → Float32 (ASR)
**Location:** `src/realtalk/perception/asr.py:229` (SherpaOnnxASR)

```python
# Input: Int16 bytes from buffer
# Output: Float32 normalized to [-1.0, 1.0]
audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
```

This is correct normalization but assumes proper Int16 input.

---

### Conversion 3: Minimax Hex → Bytes → Base64
**Location:** `src/realtalk/cognition/tts.py:124-128`

```python
# Minimax API returns hex-encoded string
audio_hex = result["data"]["audio"]  # e.g., "48656c6c6f..."

# Convert to bytes
audio_data = bytes.fromhex(audio_hex)

# For WebSocket JSON transmission, re-encode as base64
audio_base64 = base64.b64encode(audio_data).decode()
```

**Issue:** Double encoding (hex → bytes → base64) adds overhead.

---

### Conversion 4: Base64 → Blob → Audio
**Location:** `src/realtalk/web/server.py:593-626` (JavaScript)

```javascript
// Decode base64 to binary
const binaryString = atob(data.audio);
const bytes = new Uint8Array(binaryString.length);
for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
}

// Create MP3 blob
const blob = new Blob([combined], {type: 'audio/mpeg'});
const audioUrl = URL.createObjectURL(blob);
const audio = new Audio(audioUrl);
```

**Issue:** Manual base64 decoding is slow. Could use `fetch('data:audio/mp3;base64,...')` or ArrayBuffer.

---

## Known Issues

### Issue 1: Sample Rate Mismatch Risk
**Severity:** HIGH

Browsers may ignore the 16kHz constraint in `getUserMedia`. If the browser provides 48kHz audio:
- Frontend processes at wrong sample rate
- Backend assumes 16kHz, causing pitch distortion (3x speedup)
- ASR fails due to wrong frequency characteristics

**Fix:** Validate and resample on backend:
```python
# Detect actual sample rate from audio duration
# Resample using librosa or scipy.signal.resample
```

---

### Issue 2: Buffer Overflow in ASR
**Severity:** HIGH
**Location:** `src/realtalk/perception/asr.py:140-157`

```python
async def stream_audio(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
    buffer = b""
    async for chunk in audio_stream:
        buffer += chunk
        if len(buffer) >= min_chunk_size:
            result = await self.recognize(buffer)
            buffer = b""  # ⚠️ RESET - loses partial words!
```

When buffer reaches threshold (100ms or 1s), it's completely cleared. Partial words at the end are lost.

**Fix:** Implement sliding window:
```python
overlap = 0.1  # Keep last 100ms for next chunk
result = await self.recognize(buffer)
buffer = buffer[-int(overlap * sample_rate * 2):]  # Keep overlap
```

---

### Issue 3: JSON Audio Transmission
**Severity:** MEDIUM
**Location:** Frontend → Backend WebSocket

Sending audio as JSON float arrays:
- Increases size ~3x (float → string)
- Loses precision (float32 → decimal string → float32)
- Causes WebSocket backpressure

**Fix:** Use binary WebSocket frames:
```javascript
// Frontend: Send ArrayBuffer directly
const audioBuffer = new Float32Array(channelData);
ws.send(audioBuffer);  // Binary frame

// Backend: Receive bytes directly
message = await ws.receive_bytes()
```

---

### Issue 4: VAD Frame Truncation
**Severity:** MEDIUM
**Location:** `src/realtalk/perception/vad.py:125-138` (WebRTCVAD)

```python
frame_duration = 30  # ms
frame_size = int(self.sample_rate * frame_duration / 1000)

if len(audio_int16) < frame_size * 2:
    return VADResult(is_speech=False, confidence=0.0, timestamp_ms=0)

is_speech = self._vad.is_speech(audio_int16[:frame_size * 2], self.sample_rate)
```

Only first 30ms of each chunk is analyzed. Remainder is ignored, potentially missing speech.

**Fix:** Process audio in 30ms windows:
```python
for i in range(0, len(audio_int16), frame_size * 2):
    frame = audio_int16[i:i + frame_size * 2]
    if len(frame) < frame_size * 2:
        break
    is_speech = self._vad.is_speech(frame, self.sample_rate)
    # Aggregate results
```

---

### Issue 5: Frontend/Backend VAD Mismatch
**Severity:** LOW

Frontend uses energy-based VAD (RMS threshold):
```javascript
const rms = Math.sqrt(sum / channelData.length);
if (rms > SILENCE_THRESHOLD) {  // 0.01
    // Speech detected
}
```

Backend uses WebRTC or Silero VAD (ML-based).

**Result:** Different speech boundaries, causing:
- Frontend sends audio backend would have rejected
- Frontend stops too early (before ASR has enough context)
- Partial word at start/end

**Fix:** Unify VAD or have backend control recording duration.

---

## Audio Buffer Flow

### Input Buffer (Frontend → Backend)

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend: ScriptProcessorNode                              │
│  Buffer size: 4096 samples (~256ms @ 16kHz)                │
│  Frequency: ~4 times per second                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket (JSON float array)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend: _audio_buffer (List[bytes])                       │
│  Type: List of Int16 byte arrays                            │
│  Cleared: On mic-audio-end or audio_start                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Combined and sent to ASR
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ASR: SherpaOnnxASR                                        │
│  Input: Complete audio buffer as Int16                      │
│  Output: Text                                               │
└─────────────────────────────────────────────────────────────┘
```

### Output Buffer (Backend → Frontend)

```
┌─────────────────────────────────────────────────────────────┐
│  Backend: TTS (Minimax)                                     │
│  Output: MP3 chunks (hex-encoded)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ WebSocket (base64-encoded MP3)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Frontend: audioChunks[] array                              │
│  Type: Array of Uint8Array                                  │
│  Cleared: On audio.onended (RACE CONDITION!)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Combined and played
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Audio element                                              │
│  Format: MP3                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Sample Rates (config.py)

```python
class ASRConfig(BaseModel):
    sample_rate: int = 16000  # 16kHz

class TTSConfig(BaseModel):
    sample_rate: int = 32000  # 32kHz for Minimax
```

### VAD Thresholds

```python
class VADConfig(BaseModel):
    threshold: float = 0.5  # Silero confidence threshold

# Frontend VAD (hardcoded in server.py JS)
const SILENCE_THRESHOLD = 0.01;  // RMS threshold
const SILENCE_FRAMES_NEEDED = 5;  // ~1.3s at 5 frames
```

---

## Testing Audio Pipeline

### Test 1: Sample Rate Validation
```python
# Verify audio duration matches expected at 16kHz
expected_duration = len(audio_bytes) / (2 * 16000)  # 2 bytes per sample
actual_duration = recorded_duration
assert abs(expected_duration - actual_duration) < 0.1
```

### Test 2: Format Conversion Round-trip
```python
# Float → Int16 → Float should preserve values (with quantization)
original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
int16 = (original * 32767).astype(np.int16)
recovered = int16.astype(np.float32) / 32768.0
assert np.allclose(original, recovered, atol=1e-4)
```

### Test 3: Buffer Sliding Window
```python
# Verify overlap is preserved between chunks
chunk1 = b"part1-part2-"
chunk2 = b"-part2-part3"
# After processing chunk1, overlap "-part2" should be kept
# When processing chunk2, result should include "part2"
```

---

*This document should be updated when audio processing changes are made.*
