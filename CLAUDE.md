# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. This project is managed by uv.

## Project Overview

RealTalk is a **human-like full-duplex voice interaction core module** that enables natural, real-time voice conversations with AI. Unlike traditional request-response systems, it supports "边听边想" (thinking while listening), interruption handling, and emotional resonance.

## Development Commands

```bash
# Install dependencies
uv sync

# Run the web server (HTTP/WebSocket on localhost:8080)
uv run realtalk

# Run the CLI (for testing voice interaction with local microphone)
uv run realtalk-cli

# Run tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run a single test
uv run pytest tests/test_gatekeeper.py::TestRuleBasedGatekeeper::test_short_silence_incomplete_sentence -v

# Lint code
uv run ruff check src/

# Type check
uv run mypy src/
```

## Architecture

The system follows a **layered architecture**:

```
┌─────────────────────────────────────────────────┐
│              Transport Layer                     │
│         (WebRTC / WebSocket)                      │
├─────────────────────────────────────────────────┤
│             Orchestration Layer                  │
│    (FSM, Gatekeeper, Context Accumulator)       │
├──────────────┬──────────────────────────────────┤
│  Perception  │         Cognition                 │
│  - VAD       │    - LLM (OpenRouter/Gemini)     │
│  - ASR       │    - TTS (Minimax/Edge)          │
└──────────────┴──────────────────────────────────┘
```

### Key Modules

| Module | Path | Description |
|--------|------|-------------|
| **VAD** | `src/realtalk/perception/vad.py` | Voice Activity Detection (Silero/WebRTC) |
| **ASR** | `src/realtalk/perception/asr.py` | Streaming ASR (Sherpa-ONNX SenseVoice local, or Minimax API) |
| **LLM** | `src/realtalk/cognition/llm.py` | Language Model (OpenRouter/Gemini or local Qwen) |
| **TTS** | `src/realtalk/cognition/tts.py` | Text-to-Speech (Minimax API or Edge-TTS) |
| **Gatekeeper** | `src/realtalk/orchestration/gatekeeper.py` | Intent classifier for turn-taking |
| **FSM** | `src/realtalk/orchestration/fsm.py` | Finite State Machine for orchestration |
| **Accumulator** | `src/realtalk/orchestration/accumulator.py` | Context accumulation for "argument mode" |
| **Orchestrator** | `src/realtalk/core/orchestrator.py` | Main coordinator that ties all components together |
| **Web Server** | `src/realtalk/web/server.py` | HTTP/WebSocket server with built-in frontend |
| **CLI** | `src/realtalk/cli.py` | Command-line interface for local microphone testing |

### Core Concepts

1. **Gatekeeper Decision Logic**: The gatekeeper uses multi-modal fusion to decide when to respond:
   - `WAIT`: Short silence + incomplete semantics → keep listening
   - `REPLY`: Long silence + complete semantics → generate response
   - `INTERRUPT`: User speaks during AI output with high energy/commands
   - `ACCUMULATE`: Long silence but high emotion → keep accumulating context

2. **Stubbornness Level** (0-100): Controls interrupt behavior via `StubbornnessController`:
   - 0-30: Polite mode - stop immediately on user speech
   - 30-70: Medium - ignore very short non-command interruptions
   - 70-100: Argument mode - ignore short interruptions, counter-respond

3. **Context Accumulator**: Handles multi-segment speech accumulation when the gatekeeper detects emotional content but the user pauses. Accumulated context is flushed after a timeout or when max segments reached.

4. **State Machine States**: IDLE → LISTENING → PROCESSING → SPEAKING → (INTERRUPTED → LISTENING)

## Configuration

Configuration is managed via `.env` file with pydantic-settings in `src/realtalk/config.py`:

**Required:**
- `MINIMAX_API_KEY`: For ASR and TTS (if using Minimax)
- `MINIMAX_GROUP_ID`: Minimax group identifier
- `OPENROUTER_API_KEY`: For LLM (if using OpenRouter)

**Configurable via environment:**
- `VAD model`: silero (default) or webrtc
- `ASR model`: sherpa-onnx (default, local) or minimax
- `LLM model`: google/gemini-2.5-flash (default) via OpenRouter
- `TTS voice`: male-qn-qingse (default) via Minimax

## Component Details

### VAD (`src/realtalk/perception/vad.py`)
- `SileroVAD`: ML-based VAD with energy fallback
- `WebRTCVAD`: Lightweight WebRTC-based VAD
- Factory: `create_vad()` - defaults to Silero

### ASR (`src/realtalk/perception/asr.py`)
- `SherpaOnnxASR`: Local SenseVoice model (default), supports zh/en/ja/ko/yue
- `MinimaxASR`: Cloud ASR via Minimax API
- Factory: `create_asr()` - defaults to Sherpa-ONNX (auto-downloads model on first use)

### LLM (`src/realtalk/cognition/llm.py`)
- `OpenRouterLLM`: Cloud LLM via OpenRouter (default: Gemini)
- `QwenLLM`: Local Qwen via llama.cpp server
- Factory: `create_llm()` - defaults to OpenRouter

### TTS (`src/realtalk/cognition/tts.py`)
- `MinimaxTTS`: Cloud TTS via Minimax API (default, returns hex-encoded MP3)
- `EdgeTTS`: Free Edge TTS (alternative)
- Factory: `create_tts()` - defaults to Minimax

### Gatekeeper (`src/realtalk/orchestration/gatekeeper.py`)
- `RuleBasedGatekeeper`: Regex + heuristic based decisions (default)
- `MLGatekeeper`: BERT-based (requires transformers, falls back to rule-based)
- Detects semantic completeness via punctuation and ending particles
- Detects interruptions via keywords and energy threshold

### FSM (`src/realtalk/orchestration/fsm.py`)
- States: IDLE, LISTENING, SPEAKING, PROCESSING, ACCUMULATING, INTERRUPTED
- Events: USER_START_SPEAKING, USER_STOP_SPEAKING, USER_INTERRUPT, GATEKEEPER_DECISION, LLM_RESPONSE, TTS_COMPLETE
- Factory: `create_default_fsm()` sets up standard transitions

## Error Handling

The project defines a hierarchical exception structure in `src/realtalk/exceptions.py`:
- `RealTalkError` (base)
  - `APIError`
    - `ASRError`
    - `TTSError`
    - `LLMError`
  - `VADError`
  - `TransportError`
  - `OrchestrationError`

## Key Design Patterns

- **Factory functions**: `create_vad()`, `create_asr()`, `create_llm()`, `create_tts()`, `create_gatekeeper()`, `create_orchestrator()`
- **Abstract base classes**: `BaseVAD`, `BaseASR`, `BaseLLM`, `BaseTTS`, `BaseGatekeeper`
- **Async-first**: All I/O operations are async using `asyncio` and `aiohttp`
- **Streaming**: TTS and LLM support streaming responses for low latency

## Testing Patterns

Tests use pytest with asyncio support:
- Mock external API calls (Minimax, OpenRouter)
- Test gatekeeper with different silence durations and text patterns
- Test FSM state transitions
- Test accumulator flush behavior
