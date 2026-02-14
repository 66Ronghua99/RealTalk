"""Custom exceptions for RealTalk."""


class RealTalkError(Exception):
    """Base exception for RealTalk."""
    pass


class APIError(RealTalkError):
    """API related errors."""
    pass


class ASRError(APIError):
    """ASR related errors."""
    pass


class TTSError(APIError):
    """TTS related errors."""
    pass


class LLMError(APIError):
    """LLM related errors."""
    pass


class VADError(RealTalkError):
    """VAD related errors."""
    pass


class TransportError(RealTalkError):
    """Transport layer errors."""
    pass


class OrchestrationError(RealTalkError):
    """Orchestration layer errors."""
    pass
