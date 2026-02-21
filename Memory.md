# RealTalk Development Memory

## Environment Setup Notes

### 2026-02-20: Sherpa-ONNX Library Loading Issue (macOS)

**Problem:**
Sherpa-ONNX fails to load with error:
```
ImportError: dlopen(..._sherpa_onnx.cpython-312-darwin.so): Library not loaded: @rpath/libonnxruntime.1.23.2.dylib
```

**Root Cause:**
- `sherpa-onnx` Python package requires `onnxruntime` but doesn't properly declare it as a dependency
- Even after installing `onnxruntime`, the dynamic library isn't found because it's in `onnxruntime/capi/` but sherpa-onnx looks in its own `lib/` directory

**Solution:**
1. Add `onnxruntime==1.23.2` to `pyproject.toml` dependencies (pinned to match sherpa-onnx requirements)
2. Create a symlink from sherpa_onnx lib directory to onnxruntime:
   ```bash
   ln -sf $(pwd)/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib \
          $(pwd)/.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.1.23.2.dylib
   ```

**Verification:**
```bash
uv run realtalk-cli --help  # Should start without ImportError
uv run pytest              # All 65 tests should pass
```

---

## Code Fixes

### 2026-02-20: CLI Import Error

**Problem:**
`realtalk-cli` command fails with:
```
ImportError: attempted relative import beyond top-level package
```

**Fix:**
Changed relative imports to absolute imports in `src/realtalk/cli.py`:
- `from ..cognition.llm import Message` → `from realtalk.cognition.llm import Message`
- (and similar for all other imports)

**Why:**
When run as a console script via `pyproject.toml` `[project.scripts]`, Python treats the module as top-level, breaking relative imports.

---

### 2026-02-21: TTS Duplicate Playback Issue (Minimax API)

**Problem:**
When using `stream_synthesize()` with Minimax TTS API, the audio plays twice:
1. First time: incremental audio chunks play sequentially
2. Second time: a complete audio file plays

**Root Cause:**
Minimax TTS streaming API returns SSE data in this format:
- First N chunks: incremental audio (~20KB each)
- Last chunk: complete audio (~313KB, much larger)

The original code yielded all chunks including the final complete audio, causing duplicate playback since the incremental chunks already contain all the audio data.

**Solution:**
Modified `MinimaxTTS.stream_synthesize()` in `src/realtalk/cognition/tts.py` to:
1. Parse all audio chunks first into a list
2. Yield only the incremental chunks (`parsed_chunks[:-1]`)
3. Skip the final complete audio chunk
4. Yield a final marker with `audio=None, is_final=True`

```python
# Yield all chunks except the last one (complete audio)
if len(parsed_chunks) > 1:
    for i, audio_data in enumerate(parsed_chunks[:-1]):
        yield TTSResult(audio=audio_data, is_final=False, ...)
    # Skip the last chunk - it's the complete audio

# Final marker
yield TTSResult(audio=None, is_final=True, ...)
```

**Prevention Measures:**
- Always inspect streaming API response format before implementing
- Log chunk sizes to identify patterns (incremental vs complete)
- When API returns both incremental and complete data, choose one approach:
  - **Streaming**: Use incremental chunks only (for low latency)
  - **Non-streaming**: Use the complete audio only (for simplicity)
- Never yield both - they contain the same audio content

---

### 2026-02-21: ResponseGenerator Architecture

**Problem:**
Response generation logic was duplicated between `cli.py` and `orchestrator.py`, leading to:
- Bug fixes needed in multiple places
- Inconsistent behavior between CLI and Web modes
- Hard to add features like conversation history

**Solution:**
Created unified `ResponseGenerator` class that:
1. Encapsulates LLM streaming + TTS logic
2. Manages `ConversationManager` for multi-turn history
3. Handles sentence-level streaming TTS
4. Provides duplicate generation protection

**Key Design Decisions:**
- `ResponseGenerator` owns the `ConversationManager` instance
- Callback-based architecture for audio chunks and completion
- Sentence detection using regex delimiters (。！？.!?)
- Proper handling of both cumulative and incremental LLM streams

**Files Changed:**
- `src/realtalk/cognition/conversation.py` (new)
- `src/realtalk/core/response_generator.py` (new)
- `src/realtalk/cli.py` (modified)
- `src/realtalk/core/orchestrator.py` (modified)

**Prevention Measures:**
- Always abstract duplicated logic into reusable components
- Keep transport layer (CLI/Web) thin, delegate to core classes
- Use factory/callback patterns for extensibility

**Verification:**
```bash
uv run realtalk-cli  # Trigger TTS, should hear audio only once
```

---

### 2026-02-21: Streaming Pipeline Implementation Lessons

**Architecture Pattern: Decorator Chain**
参考 Open-LLM-VTuber 实现多级流式流水线:
```
LLM Token Stream → SentenceDivider → TTSTaskManager → Audio Playback
```

Key Insights:
1. **Fast First Response**: Split first sentence at comma/顿号 for sub-1.5s latency
2. **Parallel TTS**: Multiple sentences generated concurrently with ordered delivery
3. **Sequence Numbers**: Essential for maintaining order when tasks complete out-of-order
4. **Async Generator Pattern**: Use `async for` with proper yield from nested generators

**Common Pitfall - Generator Chaining:**
```python
# WRONG: Doesn't actually yield results
async for result in inner_generator():
    # ... process ...
# Results are consumed but not yielded outward

# CORRECT: Must yield explicitly
async for result in inner_generator():
    yield result
```

**Testing Async Generators:**
```python
results = []
async for item in generator():
    results.append(item)
assert len(results) > 0  # Verify actual consumption
```

**Concurrency Control:**
- Use `asyncio.Semaphore` for limiting concurrent TTS requests
- Use `asyncio.Queue` for result passing between workers
- Cancel tasks properly in finally blocks to avoid warnings

---

### 2026-02-21: Ordered Delivery Implementation

**Problem:** Parallel TTS tasks complete at different times, but audio must play in sentence order.

**Solution:** Buffer + Sequence Number Pattern
```python
completed_results: Dict[int, TTSResult] = {}
next_to_yield = 0

# When task completes:
completed_results[seq] = result

# Yield in order:
while next_to_yield in completed_results:
    yield completed_results.pop(next_to_yield)
    next_to_yield += 1
```

**Key Design Decision:**
- Don't block on missing sequences (handles gaps from failed tasks)
- Use timeouts to prevent infinite waiting
- Clean up buffer periodically to prevent memory growth

---

### 2026-02-21: E2E Testing Bug Fixes

#### Issue 1: ASR Recognizing Chinese as Japanese ("ん")

**Problem:**
User says "你好" but ASR recognizes it as "ん" (Japanese hiragana).

**Root Cause:**
Sherpa-ONNX SenseVoice is a multilingual model (zh/en/ja/ko/yue). Without explicit language hint, it may incorrectly detect short audio as Japanese.

**Solution:**
Force Chinese language in `SherpaOnnxASR.load()`:
```python
self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=str(sense_voice_model),
    tokens=str(tokens_file),
    num_threads=self.num_threads,
    use_itn=self.use_itn,
    language="zh",  # Force Chinese language
)
```

**Alternative Solutions:**
- Use Minimax ASR API (cloud-based, better for Chinese)
- Train/fine-tune SenseVoice on Chinese-only data
- Add language detection pre-processing

---

#### Issue 2: TTS Audio Playing Twice

**Problem:**
TTS audio plays twice even after previous fixes to `stream_synthesize()`.

**Investigation:**
1. Previous fix was for `stream_synthesize()` method
2. But `TTSTaskManager` now uses non-streaming `synthesize()` method
3. The duplicate might come from:
   - Minimax API returning data in unexpected format
   - Callback being triggered twice
   - Audio player issue

**Defensive Solution:**
Add audio deduplication in CLI playback:
```python
# Deduplication: check if this exact audio was already played
audio_hash = hashlib.md5(audio_data[:1024]).hexdigest()

if audio_hash in self._played_audio_hashes:
    logger.debug(f"Skipping duplicate audio chunk")
    return

self._played_audio_hashes.add(audio_hash)
```

**Key Insight:**
When dealing with third-party APIs that may have inconsistent behavior, add defensive checks at the consumption point (audio playback) rather than relying solely on source fixes.

**Verification:**
```bash
uv run realtalk-cli
# Say "你好" → Should hear Chinese response, only once
```

---

### 2026-02-21: E2E Debug Setup & Threshold Tuning

**Problem:**
1. ASR识别断断续续，语音输入捕捉不完整
2. TTS播放两次的问题难以定位源头（ASR/LLM/TTS）

**Debug Solution:**
Added comprehensive debug logging and file saving in `cli.py`:

```python
# Auto-generated debug directory per session
debug_cli_{timestamp}/
├── input_{call_id}.pcm      # Raw microphone input (16-bit PCM)
├── asr_{call_id}.txt        # ASR recognition result
├── llm_input_{call_id}.txt  # Text sent to LLM
├── llm_output_{call_id}.txt # LLM response text
└── tts_{call_id}_{seq}_{hash}.mp3  # Each TTS audio chunk
```

**Log Prefix Format:** `[DEBUG-{call_id}]` - allows grepping specific turn:
```bash
uv run realtalk-cli 2>&1 | grep "DEBUG-1771674395_1"
```

**Threshold Adjustments for Better Capture:**

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| Silence threshold | 1500ms | 800ms | Faster response to speech end |
| Silence frames | 15 (1.5s) | 8 (0.8s) | More responsive processing |
| VAD energy threshold | 0.02 | 0.008 | More sensitive voice detection |
| Confidence multiplier | 10x | 20x | Better confidence scaling |

**Files Modified:**
- `src/realtalk/cli.py` - Debug logging, audio/text file saving
- `src/realtalk/perception/vad.py` - Lowered energy threshold

**Debug Commands:**
```bash
# Run with all debug output visible
uv run realtalk-cli

# Check saved debug files
ls -la debug_cli_*/

# Play back captured input audio
ffplay -f s16le -ar 16000 -ac 1 debug_cli_*/input_*.pcm
```
