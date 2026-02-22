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

---

### 2026-02-22: 全双工打断检测（无需AEC库）

**问题描述**：
TTS播放期间麦克风输入被完全丢弃，导致用户无法打断AI说话。

**根本原因**：
为解决回声问题采用了最简单但体验差的方法——直接在 `_on_audio_input()` 底层回调里丢弃输入。

**解决方案**：
不引入 speexdsp/WebRTC AEC 库，使用**双重过滤算法**识别人声 vs 回声：

```python
# Stage 1: Energy gate
energy_ok = mic_rms > speaker_rms * echo_attenuation_factor and mic_rms > 0.01

# Stage 2: High-threshold VAD (much stricter than normal)
if energy_ok:
    vad_result = await self.vad.detect(audio_array)
    vad_ok = vad_result.confidence >= 0.82  # vs normal 0.5

# Stage 3: Frame counter (filter transients)
if vad_ok:
    interrupt_frame_count += 1
    if interrupt_frame_count >= 3:
        trigger_interrupt()
```

**关键设计决策**：
- `AudioHandler.play_audio()` 实时更新 `_speaker_rms`（每个100ms播放块），供打断检测比对
- `request_interrupt()` 设置标志，播放循环在下一个100ms块检测到后立即退出
- 打断后等待仅0.15s（vs 正常的0.3s），因为用户已经在说话了
- `_generate_response()` 的并发保护从"立即skip"改为"等待重试1s"，适配打断后的快速重启

**环境相关参数**（需根据硬件调整）：
- `_echo_attenuation_factor = 0.4`：内置MacBook扬声器+麦克风的估算衰减比
- 耳机使用者：回声极小，可设为 0.1~0.2
- 外接独立扬声器：距离远，可设为 0.5~0.7

**预防措施**：
- 在设计语音类应用时，从一开始就考虑全双工架构，而非事后补丁
- AEC库（speexdsp）在 macOS ARM 上存在编译问题，纯算法方案更可靠
- 调试时在日志中始终打印 `mic_rms` vs `speaker_rms` 的实际比值，便于校准系数


**问题描述**：
日志显示 AI 刚刚完成一句长段落的语音播放（2-3秒），11ms 后 VAD 立刻触发记录，并将 AI 刚说过的话通过 ASR 原封不动地识别了出来，导致死循环进行回答。

**根本原因**：
三个设计缺陷互相叠加：
1. `AudioHandler.play_audio` 中使用了阻塞型的 `output_stream.write(chunk)` 调用。因为在 `asyncio` Task 中执行阻塞操作，导致整个 Event Loop 被卡住，系统无法及时处理队列中的数据。
2. 由于 Event Loop 被卡住，在播放 TTS 的 2-3 秒内，麦克风将这段时间听到的"回音"全部积压在 `asyncio.Queue` 中。
3. 播放一结束，Event Loop 恢复，积压的回音由于 `_is_playing` 在 `finally` 块中立即被置为 `False`，导致它们不仅没有被抑制，反而瞬间全被 VAD 消费，从而触发判定 AI 给自己说话。

**解决方案**：
1. **防止阻塞循环**：改用 `await loop.run_in_executor(None, self._output_stream.write, chunk)` 来执行阻塞的音频输出。
2. **提前丢弃保护**：在最外层的 `_on_audio_input`（声卡底层回调）中，通过 `self.audio_handler.is_currently_playing()` 直接抛弃处于正在播放状态的麦克风输入流，而不放入 `asyncio.Queue` 中。
3. **播放控制**：引入 `asyncio.Lock()` 实现不同并发块中分段 TTS 的强制有序与同步播放。
4. **混响延时**：在播放完成时加入 `await asyncio.sleep(0.3)`，使得房间中的混响和硬件电信号残留能有时间耗散，再将开放录音。

**预防措施**：
- 绝对不要在 async 函数中执行大体积数据的同步 I/O。
- 设计双工交互防回音（Echo Cancellation/Suppression）时，抑制动作应尽可能早地在数据生产源头（底层输入 Callback 侧）阻断，而非在队列消费侧做判断。

---

### 2026-02-22: 外放音量高时 VAD 长时间保持 0.99 导致误打断

**问题描述：**
Mac 外放声音较大时，TTS 播放期间麦克风持续捕获回声，VAD 给出 0.97-1.00 置信度，导致打断检测误触发。即使实现了能量门槛（`energy_ok`），`speaker_rms=0.0` 时 `echo_threshold=0.0`，任何麦克风声音都能通过。

**根本原因（三个叠加 Bug）：**
1. **瞬时 `_speaker_rms` 在块间隙归零**：`play_audio()` 在每个 100ms 块执行时才更新 `_speaker_rms`，块间空隙或最后 `finally` 块清零后，`echo_threshold=0`，能量门槛完全失效
2. **播放后无抑制窗口**：`_is_playing=False` 后立即切换到正常 VAD 模式，房间混响（来不及消散）被当作人声处理
3. **VAD 无法区分回声**：Silero VAD 看到的是麦克风捕获的语音信号，不管是真人还是回声都给高 confidence

**解决方案：**
1. **Peak-hold `_peak_speaker_rms`**：用带衰减的峰值保持（每块乘衰减系数 0.85）替代瞬时值，即使块间隙也不会归零
2. **后播放冷却期（500ms）**：播放结束后设置 `_post_play_cooldown_until = monotonic() + 0.5`，冷却期内直接丢弃麦克风输入
3. **新增 `is_in_echo_suppression_window()`**：统一判断"是否在回声风险期"（播放中 OR 冷却期内）
4. **提高 VAD 阈值至 0.95**：从 `0.8` 提高，过滤更多回声误判（但主要防线是峰值保持的能量门槛）

```python
# AudioHandler: peak-hold RMS
chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
self._peak_speaker_rms = max(chunk_rms, self._peak_speaker_rms * 0.85)

# finally 块：设置冷却期
cooldown_s = 0.3 if interrupted else 0.5
self._post_play_cooldown_until = time.monotonic() + cooldown_s
self._peak_speaker_rms = 0.0  # 冷却期已设，再清零

# CLI._process_audio_chunk
in_echo_window = self.audio_handler.is_in_echo_suppression_window()
if in_echo_window and not is_actively_playing:
    return  # 冷却期：直接丢弃
```

**预防措施：**
- 任何基于"播放状态"的过滤必须同时考虑：① 播放中 ② 播放刚结束的余响期
- 绝不用瞬时值做能量门槛，必须用 peak-hold 或平滑值
- 在 Memory 中记录 `_echo_attenuation_factor=0.72`（MacBook 内置扬声器+麦克风经验值）

