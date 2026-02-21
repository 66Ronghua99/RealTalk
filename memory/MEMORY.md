# RealTalk Memory

## 2026-02-21: TTS 音频重叠 Bug 根因与修复

- **问题描述**: AI 语音回复出现内容重叠（如"要不你直接打字跟我说吧？"之后听到"打字跟我说吧？"）
- **根本原因**: `llm.py` `OpenRouterLLM.stream_chat()` 中使用两个独立 `if` 分支，导致当 OpenRouter 最后一条 SSE 同时包含 `content` 和 `finish_reason` 时，同一个 delta 内容被 yield 两次。重复内容进入 `SentenceDivider` buffer，在最后标点处被切成"正常句子"+"重复后缀"两条句子，各触发一次 TTS。
- **解决方案**: 将两个 `if` 合并为一次 yield，`finish_reason` 附在同一 `LLMResponse` 对象上
- **预防措施**: LLM `stream_chat` 接口 — 每个 SSE chunk 只能 yield 一次，不能对同一 chunk 做多次 yield；新 LLM 后端接入时必须检查 `content` 与 `finish_reason` 同时存在的 corner case

---

## 2026-02-20: Phase 1 Frontend Extraction Completed

### Static File Separation

**Before:** 57% of server.py was embedded frontend code (545 lines of HTML/CSS/JS in Python strings)
**After:** Clean separation with ~420 line server.py focused purely on backend logic

**Files Created:**
- `src/realtalk/web/templates/index.html` - HTML template structure
- `src/realtalk/web/static/css/style.css` - All CSS styles (114 lines)
- `src/realtalk/web/static/js/app.js` - All JavaScript logic (376 lines)

**Benefits:**
- IDE support for frontend editing (syntax highlighting, linting)
- Browser caching of static assets
- Clearer separation of concerns
- Easier testing and maintenance

**Implementation Notes:**
- Used aiohttp's static file serving (`add_static`)
- Added fallback HTML in `get_index_html()` for backwards compatibility
- Template file loaded with simple `read_text()`, no Jinja2 needed for this use case

---

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
- `src/realtalk/web/server.py` - Frontend race condition, ASR clipping, static file serving
- `src/realtalk/perception/asr.py` - Sliding window buffer

## Testing

All 23 existing tests pass after changes:
```bash
uv run pytest tests/ -v
```
