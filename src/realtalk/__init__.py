"""RealTalk - Human-like full-duplex voice interaction core module.

Architecture:
- perception: VAD, ASR modules
- orchestration: FSM, Gatekeeper, Accumulator
- cognition: LLM, TTS modules
- transport: WebRTC, WebSocket transport
- core: Main orchestrator
"""
from .config import Config, get_config
from .core import OrchestratorConfig, VoiceOrchestrator, create_orchestrator
from .exceptions import (
    APIError,
    ASRError,
    LLMError,
    OrchestrationError,
    RealTalkError,
    TransportError,
    TTSError,
    VADError,
)
from .logging_config import logger, setup_logger

__version__ = "1.0.0"

__all__ = [
    # Config
    "Config",
    "get_config",
    # Logging
    "logger",
    "setup_logger",
    # Core
    "OrchestratorConfig",
    "VoiceOrchestrator",
    "create_orchestrator",
    # Exceptions
    "RealTalkError",
    "APIError",
    "ASRError",
    "TTSError",
    "LLMError",
    "VADError",
    "TransportError",
    "OrchestrationError",
]
