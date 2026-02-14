# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealTalk is a **human-like full-duplex voice interaction core module** that enables natural, real-time voice conversations with AI. Unlike traditional request-response systems, it supports "边听边想" (thinking while listening), interruption handling, and emotional resonance.

## Development Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test file
pytest tests/test_gatekeeper.py

# Run tests with verbose output
pytest -v

# Lint code
ruff check src/

# Type check
mypy src/
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
│  - VAD       │    - LLM (Gemini)                │
│  - ASR       │    - TTS                          │
└──────────────┴──────────────────────────────────┘
```

### Key Modules

| Module | Path | Description |
|--------|------|-------------|
| **VAD** | `src/realtalk/perception/vad.py` | Voice Activity Detection (Silero/WebRTC) |
| **ASR** | `src/realtalk/perception/asr.py` | Streaming ASR (Minimax API) |
| **LLM** | `src/realtalk/cognition/llm.py` | Language Model (OpenRouter/Gemini) |
| **TTS** | `src/realtalk/cognition/tts.py` | Text-to-Speech (Minimax API) |
| **Gatekeeper** | `src/realtalk/orchestration/gatekeeper.py` | Intent classifier for turn-taking |
| **FSM** | `src/realtalk/orchestration/fsm.py` | Finite State Machine for orchestration |
| **Accumulator** | `src/realtalk/orchestration/accumulator.py` | Context accumulation for "argument mode" |
| **Orchestrator** | `src/realtalk/core/orchestrator.py` | Main coordinator |
| **Transport** | `src/realtalk/transport/webrtc.py` | WebRTC transport layer |

### Core Concepts

1. **Gatekeeper Decision Logic**: The gatekeeper uses multi-modal fusion to decide when to respond:
   - `WAIT`: Short silence + incomplete semantics → keep listening
   - `REPLY`: Long silence + complete semantics → generate response
   - `INTERRUPT`: User speaks during AI output with high energy/commands
   - `ACCUMULATE`: Long silence but high emotion → keep accumulating context

2. **Stubbornness Level** (0-100): Controls interrupt behavior:
   - 0-30: Polite mode - stop immediately on user speech
   - 70-100: Argument mode - ignore short interruptions, counter-respond

3. **State Machine States**: IDLE → LISTENING → PROCESSING → SPEAKING → (INTERRUPTED → LISTENING)

## Configuration

Configuration is managed via `.env` file with these keys:
- `MINIMAX_API_KEY`: For ASR and TTS
- `MINIMAX_GROUP_ID`: Minimax group identifier
- `OPENROUTER_API_KEY`: For LLM (Gemini)

## Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_gatekeeper.py::TestRuleBasedGatekeeper::test_short_silence_incomplete_sentence -v
```

## Key Design Patterns

- **Factory functions**: `create_vad()`, `create_asr()`, `create_llm()`, `create_tts()`, `create_orchestrator()`
- **Abstract base classes**: `BaseVAD`, `BaseASR`, `BaseLLM`, `BaseTTS`, `BaseTransport`
- **Async-first**: All I/O operations are async using `asyncio` and `aiohttp`
