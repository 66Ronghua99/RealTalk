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

**Verification:**
```bash
uv run realtalk-cli  # Trigger TTS, should hear audio only once
```
