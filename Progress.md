# RealTalk Project Progress

## Project Overview
RealTalk is a human-like full-duplex voice interaction core module enabling natural, real-time voice conversations with AI.

## Current Status
**Date:** 2026-02-21

### ✅ Completed: VAD Dependencies Installation
- Installed `silero-vad>=6.2.0` and `webrtcvad-wheels>=2.0.14`
- Updated `vad.py` to work with new silero-vad v6.x API (requires 512-sample frames @ 16kHz)
- Modified `detect()` method to split audio into 32ms frames and aggregate results

### ✅ Completed: Acoustic Echo Feedback Loop Fix
Implemented half-duplex mode in CLI to prevent AI from responding to its own voice:
- Added `_is_playing` state tracking to `AudioHandler`
- Added `is_currently_playing()` method for echo suppression checks
- VAD processing now skipped during AI playback
- Audio buffer cleared after playback to discard residual echo

### ✅ Completed: P0 Critical Bug Fixes

#### Fix 1: TTS 播放卡顿问题 (stream_synthesize 过滤无效 chunks)
- **文件**: `src/realtalk/cognition/tts.py`
- **问题**: Minimax API 返回的 audio chunks 包含极小 (~621 bytes) 和空 (0 bytes) 数据，导致 playback error 和播放断断续续
- **修复**: 在 `stream_synthesize()` 中过滤小于 1000 bytes 的 chunk
- **代码**: `if len(audio_data) < 1000: continue`

#### Fix 2: 语音输入时间过短问题 (增加静音阈值)
- **文件**: `src/realtalk/cli.py`
- **问题**: `_silence_threshold_ms = 500ms` 对于自然对话太短，用户仅停顿 0.5 秒就触发处理
- **修复**: 增加到 `1500ms`，更符合人类对话习惯
- **位置**: Line 44 (`AudioHandler`) 和 Line 169 (`CLI`)

#### Fix 3: VAD 过于敏感问题 (调整能量阈值)
- **文件**: `src/realtalk/perception/vad.py`
- **问题**: `rms > 0.01` 阈值过低，安静环境下容易误触发语音结束
- **修复**: 增加到 `rms > 0.02`，减少误触发
- **位置**: Line 85 (`_energy_based_detection`)

#### Fix 4: TTS 回复内容截断 (根本原因修复)
- **文件**: `src/realtalk/cognition/tts.py`
- **问题**: Minimax API 返回的 chunks 混合了完整 MP3 文件（有 ID3 头）和原始 MP3 帧（无 ID3 头）
  - Chunk 0: 完整 MP3（4653 bytes，有 ID3 头）
  - Chunk 1-2: 原始 MP3 帧（无 ID3 头，无法直接播放）
  - Chunk 4: 完整 MP3（23604 bytes，有 ID3 头，最完整但被跳过！）
- **修复**: 只 yield 带有 ID3 头的完整 MP3 chunks，选择最后一个（最完整的）
- **关键发现**: 原始 MP3 帧（无 ID3 头）无法被 soundfile 正确解码，导致播放截断

#### Fix 5: ASR 提前结束 (VAD 连续检测)
- **文件**: `src/realtalk/cli.py`
- **问题**: 即使用户持续说话，VAD 音量波动导致短暂返回 `is_speech=False`，累积时间后提前触发处理
- **修复**: 引入连续静音帧计数器（`_consecutive_silence_count`），需要连续 15 帧（1.5秒）静音才触发
- **逻辑**: 检测到语音时重置计数器，只有连续静音帧达标才处理

#### Fix 6: AI 录入自身回复 (TTS Echo Recording)
- **文件**: `src/realtalk/cli.py`
- **问题**: AI 会把刚播放完的大段 TTS 语音原文重新识别为用户输入。原因是播放 TTS 音频的 `write()` 操作阻塞了 Asyncio Event Loop，使得麦克风输入队列堆积。播放完成后，积压的录音（含回音）立刻被 VAD 并发处理，错误地判定为有效输入。
- **修复**: 
  1. 使用 `run_in_executor` 将 `write()` 脱离主事件循环
  2. 加入 `asyncio.Lock()` 强制音频片段串行播放
  3. 在麦克风底层回调中提前检查 `is_currently_playing()` 并直接丢弃音频
  4. 播放完成后加入 300ms 的保护窗口以等待房间残响消散

所有 P0 fixes 已推送至 master（除了 Fix 6 位于本地最新迭代中）。

### ✅ Completed: Phase 1 - Frontend/Backend Separation

### ✅ Completed: Phase 2 - API Formalization

## Phase 1 Completed

### Extracted Files
- `src/realtalk/web/templates/index.html` - HTML template
- `src/realtalk/web/static/css/style.css` - CSS styles (114 lines)
- `src/realtalk/web/static/js/app.js` - JavaScript (376 lines)

### Server.py Changes
- Updated `create_app()` to serve static files and templates
- `get_index_html()` now reads from template file with minimal fallback
- Server reduced from 971 lines to ~420 lines (57% reduction in embedded frontend code)

### Benefits
- Frontend code can now be edited with proper IDE support
- CSS and JS are cached by browsers (better performance)
- Clearer separation of concerns
- Easier to maintain and test

## Phase 2 Completed

### New Files
- `src/realtalk/messages.py` - Pydantic message models (API v1.0.0)
  - 14 message types (7 client→server, 7 server→client)
  - Type-safe validation with bounds checking
  - JSON Schema generation support
  - `deserialize_message()` for automatic type resolution

- `src/realtalk/web/message_bus.py` - Publish-subscribe message bus
  - `MessageBus` class for decoupled message routing
  - Middleware support for logging/transformations
  - Metrics collection
  - `TypedMessageBus` variant for better IDE support

- `tests/test_messages.py` - Comprehensive message tests (42 tests)
- `docs/MESSAGES.md` - Complete protocol documentation

### Server.py Updates
- Migrated from dict-based to Pydantic model-based messages
- All outgoing messages now include timestamps
- Type-safe message handling with proper validation
- Removed legacy message support (`audio`, `audio_end`)

### API Version
- Current: **v1.0.0**
- All messages include `api_version` field
- Deserialization validates message structure

### Benefits
- **Type Safety**: Compile-time checking with Pydantic validation
- **Documentation**: JSON Schema for all message types
- **Testability**: Easy to mock and test message flows
- **Extensibility**: New message types follow established patterns
- **Debugging**: Validation errors provide clear feedback

## Phase 3: Core Dialogue Fixes (Completed)

### Goals
- Fix streaming content accumulation bug
- Add conversation history management
- Prevent duplicate response generation
- Implement sentence-level streaming TTS

### Completed
- Created `ConversationManager` for multi-turn history
- Created `ResponseGenerator` to unify CLI and Orchestrator logic
- Modified `cli.py` to use new ResponseGenerator
- Modified `orchestrator.py` to use new ResponseGenerator
- Added protection against concurrent response generation
- Added 11 tests for ConversationManager

### Key Changes
- `src/realtalk/cognition/conversation.py` - New: Multi-turn conversation history
- `src/realtalk/core/response_generator.py` - New: Unified response generation
- `src/realtalk/cli.py` - Modified: Use ResponseGenerator
- `src/realtalk/core/orchestrator.py` - Modified: Use ResponseGenerator

## Phase 4: Streaming Architecture Optimization (Completed)

### Goals
参考 Open-LLM-VTuber 的多级流式流水线架构，实现:
- 更稳定的 LLM 流式输出 (句子分割+缓冲策略)
- 更高效的 TTS 处理 (并行生成+有序交付)
- 更低的首音延迟 (首句快速响应优化)
- 更好的内存管理 (减少音频数据滞留)

### Completed
- Phase 1: Core data structures (`streaming_types.py`)
  - `SentenceOutput`: 句子级输出，包含序列号和首句标记
  - `TTSTask`: TTS任务包装器
  - `PipelineConfig`: 流水线配置
- Phase 2: SentenceDivider (`sentence_divider.py`)
  - 快速首句响应 (逗号/顿号处断开)
  - 智能边界检测 (中英文结束符+引号处理)
  - 缓冲区管理 (超时强制刷新)
- Phase 3: TTSTaskManager (`tts_task_manager.py`)
  - 任务队列 + 并行生成 (最多4个并发)
  - 序列号分配与有序交付
  - 错误处理 (跳过失败任务，继续处理)
- Phase 4: StreamingPipeline (`streaming_pipeline.py`)
  - 装饰器链设计: LLM → SentenceDivider → TTSTaskManager
  - 统一接口: `generate_response()` 和 `generate_stream()`
  - 向后兼容的 Legacy 模式
- Phase 5: ResponseGenerator 重构
  - 内部使用 StreamingPipeline
  - 保持对外接口不变
- Phase 6: Testing
  - 18个流式管道测试用例
  - 覆盖分句、并行TTS、有序交付、错误处理

### Bug Fixes (2026-02-21)
- **ASR中文识别**: 强制SenseVoice使用中文语言设置 (`language="zh"`)，避免识别为日语假名
- **TTS重复播放**: 添加音频去重机制 (MD5 hash前1KB)，防止同一音频播放两次

### Performance Improvements
- **首音延迟**: 3-5秒 → 预计 <1.5秒 (首句逗号分割优化)
- **多句回复**: 并行TTS减少总时间 30-50%
- **架构升级**: 顺序处理 → 多级流式流水线

### Key Files
| 文件 | 说明 |
|------|------|
| `src/realtalk/cognition/streaming_types.py` | 流式数据类型定义 |
| `src/realtalk/cognition/sentence_divider.py` | 流式分句器 |
| `src/realtalk/cognition/tts_task_manager.py` | 并行TTS管理器 |
| `src/realtalk/cognition/streaming_pipeline.py` | 流水线整合 |
| `src/realtalk/core/response_generator.py` | 更新为使用新流水线 |
| `tests/test_streaming_pipeline.py` | 18个流式测试 |

## E2E Debug Setup (2026-02-21)

### Added Debug Capabilities
- Auto-generated debug directory: `debug_cli_{timestamp}/`
- Saved artifacts per voice interaction:
  - `input_*.pcm` - Raw microphone input (16-bit PCM, 16kHz)
  - `asr_*.txt` - ASR recognition results with confidence
  - `llm_input_*.txt` - Text sent to LLM
  - `llm_output_*.txt` - LLM response text
  - `tts_*.mp3` - Each TTS audio chunk with hash

### Threshold Adjustments for Better Capture
| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| Silence threshold | 1500ms | 800ms | Faster response |
| Silence frames | 15 | 8 | More responsive |
| VAD energy threshold | 0.02 | 0.008 | More sensitive |

### Debug Log Format
All logs now include `[DEBUG-{call_id}]` prefix for filtering:
```bash
uv run realtalk-cli 2>&1 | grep "DEBUG-1771674395_1"
```

## Stats
| Metric | Value |
|--------|-------|
| Lines extracted from server.py (Phase 1) | ~545 |
| Current server.py | ~420 lines (was 971) |
| New files (Phase 1) | 3 |
| New files (Phase 2) | 4 |
| New files (Phase 4) | 5 |
| Total tests | 94 (65 + 29 new) |
| Streaming pipeline tests | 18 |
| Message types | 14 |
| API version | 1.0.0 |
