# RealTalk Memory

## 2026-02-20: P0 Critical Bug Fixes Implemented

### TTS Audio Double Playback - Fixed

**Root Causes Addressed:**

1. **Orchestrator Creating Multiple TTS Tasks** (`src/realtalk/core/orchestrator.py:244-249`)
   - **Problem:** Every LLM streaming chunk created a new TTS speaking task
   - **Fix:** Accumulate full LLM response first, then speak once
   - **Code Change:** Moved `_speak()` call outside the `async for` loop

2. **TTS Stop Event Cleared Too Early** (`src/realtalk/cognition/tts.py:274-283`)
   - **Problem:** `_stop_event.clear()` called immediately in `stop()`, allowing new chunks
   - **Fix:** Moved `_stop_event.clear()` to `finally` block in `stream_synthesize()`
   - **Result:** Stop event persists until stream is fully complete

3. **Frontend Race Condition** (`src/realtalk/web/server.py:551, 601-656`)
   - **Problem:** `audioChunks` array only cleared in `audio.onended`, causing overlap
   - **Fix:** Added `isPlayingAudio` flag, reset chunks when new TTS starts during playback
   - **Code:** Added check at start of `tts_audio` handler

### ASR Accuracy Issues - Fixed

1. **Float-to-Int16 Clipping** (`src/realtalk/web/server.py:211-213`)
   - **Problem:** No clipping before conversion caused distortion when values > 1.0
   - **Fix:** Added `np.clip(audio_array, -1.0, 1.0)` before scaling

2. **Buffer Reset Losing Partial Words** (`src/realtalk/perception/asr.py:140-157, 247-257`)
   - **Problem:** Complete buffer reset after each chunk lost partial words at boundaries
   - **Fix:** Implemented sliding window with 100ms overlap preservation
   - **Code:** Keep last 100ms of buffer for next chunk processing

## Key Files Modified

- `src/realtalk/core/orchestrator.py` - TTS task debouncing
- `src/realtalk/cognition/tts.py` - Stop event handling
- `src/realtalk/web/server.py` - Frontend race condition, ASR clipping
- `src/realtalk/perception/asr.py` - Sliding window buffer

## Testing

All 23 existing tests pass after changes:
```bash
uv run pytest tests/ -v
```
