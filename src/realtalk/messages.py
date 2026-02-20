"""Message protocol definitions for RealTalk WebSocket API.

This module defines all WebSocket message types using Pydantic v2 models
for type-safe serialization, validation, and JSON Schema generation.

API Version: 1.0.0
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Self, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# API Version
API_VERSION = "1.0.0"


class MessageType(str, Enum):
    """All message types in the RealTalk protocol."""

    # Client -> Server
    TEXT = "text"
    STUBBORNNESS = "stubbornness"
    INTERRUPT = "interrupt"
    CLEAR = "clear"
    MIC_AUDIO_DATA = "mic-audio-data"
    MIC_AUDIO_END = "mic-audio-end"
    AUDIO_START = "audio_start"

    # Server -> Client
    TRANSCRIPT = "transcript"
    LLM_CHUNK = "llm_chunk"
    TTS_AUDIO = "tts_audio"
    STATE = "state"
    GATEKEEPER = "gatekeeper"
    STATUS = "status"
    VAD = "vad"


class State(str, Enum):
    """FSM states sent to client."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ACCUMULATING = "accumulating"


class GatekeeperAction(str, Enum):
    """Gatekeeper decision actions."""

    WAIT = "wait"
    ACCUMULATE = "accumulate"
    INTERRUPT = "interrupt"
    REPLY = "reply"


class BaseMessage(BaseModel, ABC):
    """Base class for all messages.

    Attributes:
        type: Message type discriminator
        api_version: API version for compatibility checking
        timestamp: Unix timestamp in milliseconds (optional)
    """

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",  # Reject unknown fields
        populate_by_name=True,
    )

    type: MessageType
    api_version: str = Field(default=API_VERSION, description="API version")
    timestamp: Optional[int] = Field(
        default=None,
        description="Unix timestamp in milliseconds",
        ge=0,
    )

    @abstractmethod
    def get_type_value(self) -> str:
        """Return the string value of the message type."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return self.model_dump_json(by_alias=True, exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMessage":
        """Deserialize dictionary to appropriate message type."""
        return deserialize_message(data)


# =============================================================================
# Client -> Server Messages
# =============================================================================


class TextInput(BaseMessage):
    """User text input message.

    Sent when user types a message in the text input field.

    Example:
        {"type": "text", "text": "Hello, how are you?"}
    """

    type: Literal[MessageType.TEXT] = MessageType.TEXT
    text: str = Field(..., min_length=1, description="User input text")

    def get_type_value(self) -> str:
        return MessageType.TEXT.value


class StubbornnessUpdate(BaseMessage):
    """Stubbornness level update message.

    Sent when user adjusts the stubbornness slider.
    Level controls how resistant the AI is to interruptions (0-100).

    Example:
        {"type": "stubbornness", "level": 50}
    """

    type: Literal[MessageType.STUBBORNNESS] = MessageType.STUBBORNNESS
    level: int = Field(
        ...,
        ge=0,
        le=100,
        description="Stubbornness level (0=polite, 100=argumentative)",
    )

    def get_type_value(self) -> str:
        return MessageType.STUBBORNNESS.value


class InterruptRequest(BaseMessage):
    """User interruption request message.

    Sent when user clicks the interrupt button or manually triggers
    an interruption while the AI is speaking.

    Example:
        {"type": "interrupt", "text": "Wait, that's not right"}
        {"type": "interrupt"}
    """

    type: Literal[MessageType.INTERRUPT] = MessageType.INTERRUPT
    text: Optional[str] = Field(
        default=None,
        description="Optional interruption text/keywords",
    )

    def get_type_value(self) -> str:
        return MessageType.INTERRUPT.value


class ClearContext(BaseMessage):
    """Clear context accumulator message.

    Sent when user wants to clear accumulated context segments.

    Example:
        {"type": "clear"}
    """

    type: Literal[MessageType.CLEAR] = MessageType.CLEAR

    def get_type_value(self) -> str:
        return MessageType.CLEAR.value


class MicAudioData(BaseMessage):
    """Microphone audio data message.

    Sent continuously while user is speaking. Contains float32 audio samples
    from the browser's audio context (typically 4096 samples per message).

    Example:
        {"type": "mic-audio-data", "audio": [0.0, 0.01, -0.02, ...]}
    """

    type: Literal[MessageType.MIC_AUDIO_DATA] = MessageType.MIC_AUDIO_DATA
    audio: List[float] = Field(
        ...,
        min_length=1,
        description="Audio samples as float32 array (-1.0 to 1.0)",
    )

    @model_validator(mode="after")
    def validate_audio_range(self) -> Self:
        """Validate audio samples are within valid float32 range."""
        # Only check first/last few samples for performance
        sample = self.audio[:10] + self.audio[-10:] if len(self.audio) > 20 else self.audio
        for i, value in enumerate(sample):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"Audio sample at position {i} out of range: {value}")
        return self

    def get_type_value(self) -> str:
        return MessageType.MIC_AUDIO_DATA.value


class MicAudioEnd(BaseMessage):
    """Microphone audio end message.

    Sent when frontend VAD detects end of speech (silence threshold reached).

    Example:
        {"type": "mic-audio-end"}
    """

    type: Literal[MessageType.MIC_AUDIO_END] = MessageType.MIC_AUDIO_END

    def get_type_value(self) -> str:
        return MessageType.MIC_AUDIO_END.value


class AudioStart(BaseMessage):
    """Audio recording start message.

    Sent when frontend VAD detects start of speech.

    Example:
        {"type": "audio_start"}
    """

    type: Literal[MessageType.AUDIO_START] = MessageType.AUDIO_START

    def get_type_value(self) -> str:
        return MessageType.AUDIO_START.value


# =============================================================================
# Server -> Client Messages
# =============================================================================


class Transcript(BaseMessage):
    """ASR transcript message.

    Sent when ASR produces transcription from audio.

    Example:
        {"type": "transcript", "text": "Hello world", "is_final": true}
    """

    type: Literal[MessageType.TRANSCRIPT] = MessageType.TRANSCRIPT
    text: str = Field(..., min_length=0, description="Transcribed text")
    is_final: bool = Field(
        default=True,
        description="Whether this is the final transcript (true for this implementation)",
    )

    def get_type_value(self) -> str:
        return MessageType.TRANSCRIPT.value


class LLMChunk(BaseMessage):
    """LLM streaming response chunk message.

    Sent as streaming chunks from the LLM. The client should concatenate
    or display the latest chunk.

    Example:
        {"type": "llm_chunk", "text": "Hello! I'm doing well", "is_final": false}
        {"type": "llm_chunk", "text": "Hello! I'm doing well, thank you!", "is_final": true}
    """

    type: Literal[MessageType.LLM_CHUNK] = MessageType.LLM_CHUNK
    text: str = Field(..., description="Current accumulated response text")
    is_final: Optional[bool] = Field(
        default=None,
        description="Whether this is the final chunk",
    )

    def get_type_value(self) -> str:
        return MessageType.LLM_CHUNK.value


class TTSAudio(BaseMessage):
    """TTS audio chunk message.

    Sent as audio chunks are generated. Audio is base64-encoded MP3 data.

    Example:
        {
            "type": "tts_audio",
            "audio": "//NEx...",
            "is_final": false,
            "chunk_index": 0
        }
    """

    type: Literal[MessageType.TTS_AUDIO] = MessageType.TTS_AUDIO
    audio: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded MP3 audio data",
    )
    is_final: bool = Field(
        ...,
        description="Whether this is the final audio chunk",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Chunk index for ordering (0-based)",
    )

    def get_type_value(self) -> str:
        return MessageType.TTS_AUDIO.value


class StateUpdate(BaseMessage):
    """FSM state update message.

    Sent when the orchestration state changes.

    Example:
        {"type": "state", "state": "listening"}
        {"type": "state", "state": "speaking"}
    """

    type: Literal[MessageType.STATE] = MessageType.STATE
    state: State = Field(..., description="Current FSM state")

    def get_type_value(self) -> str:
        return MessageType.STATE.value


class GatekeeperDecision(BaseMessage):
    """Gatekeeper decision message.

    Sent when the gatekeeper makes a decision about user intent.

    Example:
        {"type": "gatekeeper", "action": "reply", "confidence": 0.85}
    """

    type: Literal[MessageType.GATEKEEPER] = MessageType.GATEKEEPER
    action: GatekeeperAction = Field(..., description="Gatekeeper decision action")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0)",
    )

    def get_type_value(self) -> str:
        return MessageType.GATEKEEPER.value


class StatusMessage(BaseMessage):
    """Status/notification message.

    Sent for various status updates (accumulation, errors, etc.).

    Example:
        {"type": "status", "message": "Context cleared"}
        {"type": "status", "message": "Accumulated for context"}
    """

    type: Literal[MessageType.STATUS] = MessageType.STATUS
    message: str = Field(..., min_length=1, description="Status message text")

    def get_type_value(self) -> str:
        return MessageType.STATUS.value


class VADResult(BaseMessage):
    """VAD (Voice Activity Detection) result message.

    Sent during audio processing to indicate speech detection status.
    Note: Currently mainly used in legacy mode; frontend VAD uses audio_start/end.

    Example:
        {"type": "vad", "is_speaking": true, "energy": 0.75}
    """

    type: Literal[MessageType.VAD] = MessageType.VAD
    is_speaking: bool = Field(..., description="Whether speech is detected")
    energy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Audio energy level (0.0 to 1.0)",
    )

    def get_type_value(self) -> str:
        return MessageType.VAD.value


# =============================================================================
# Message Union and Deserialization
# =============================================================================

ClientMessage = Union[
    TextInput,
    StubbornnessUpdate,
    InterruptRequest,
    ClearContext,
    MicAudioData,
    MicAudioEnd,
    AudioStart,
]
"""Union of all client-to-server messages."""

ServerMessage = Union[
    Transcript,
    LLMChunk,
    TTSAudio,
    StateUpdate,
    GatekeeperDecision,
    StatusMessage,
    VADResult,
]
"""Union of all server-to-client messages."""

AnyMessage = Union[ClientMessage, ServerMessage]
"""Union of all messages in the protocol."""

# Mapping of message types to their model classes
_MESSAGE_TYPE_MAP: Dict[str, type] = {
    # Client -> Server
    MessageType.TEXT.value: TextInput,
    MessageType.STUBBORNNESS.value: StubbornnessUpdate,
    MessageType.INTERRUPT.value: InterruptRequest,
    MessageType.CLEAR.value: ClearContext,
    MessageType.MIC_AUDIO_DATA.value: MicAudioData,
    MessageType.MIC_AUDIO_END.value: MicAudioEnd,
    MessageType.AUDIO_START.value: AudioStart,
    # Server -> Client
    MessageType.TRANSCRIPT.value: Transcript,
    MessageType.LLM_CHUNK.value: LLMChunk,
    MessageType.TTS_AUDIO.value: TTSAudio,
    MessageType.STATE.value: StateUpdate,
    MessageType.GATEKEEPER.value: GatekeeperDecision,
    MessageType.STATUS.value: StatusMessage,
    MessageType.VAD.value: VADResult,
}


def deserialize_message(data: Dict[str, Any]) -> BaseMessage:
    """Deserialize a dictionary to the appropriate message type.

    Args:
        data: Dictionary containing message data with 'type' field

    Returns:
        Instance of the appropriate message class

    Raises:
        ValueError: If message type is unknown or data is invalid
    """
    msg_type = data.get("type")
    if not msg_type:
        raise ValueError("Message missing 'type' field")

    msg_class = _MESSAGE_TYPE_MAP.get(msg_type)
    if not msg_class:
        raise ValueError(f"Unknown message type: {msg_type}")

    try:
        return msg_class.model_validate(data)
    except Exception as e:
        raise ValueError(f"Failed to validate message of type '{msg_type}': {e}") from e


def get_message_schema(msg_class: type[BaseMessage]) -> Dict[str, Any]:
    """Get JSON Schema for a message class.

    Args:
        msg_class: Message class to generate schema for

    Returns:
        JSON Schema dictionary
    """
    return msg_class.model_json_schema()


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """Get JSON Schemas for all message types.

    Returns:
        Dictionary mapping message type names to their schemas
    """
    return {
        msg_type: msg_class.model_json_schema()
        for msg_type, msg_class in _MESSAGE_TYPE_MAP.items()
    }
