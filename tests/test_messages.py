"""Tests for the RealTalk message protocol."""
import json
import pytest
from pydantic import ValidationError

from realtalk.messages import (
    API_VERSION,
    AudioStart,
    ClearContext,
    GatekeeperAction,
    GatekeeperDecision,
    InterruptRequest,
    LLMChunk,
    MessageType,
    MicAudioData,
    MicAudioEnd,
    State as StateEnum,
    StateUpdate,
    StatusMessage,
    StubbornnessUpdate,
    TextInput,
    Transcript,
    TTSAudio,
    VADResult,
    deserialize_message,
    get_all_schemas,
)


class TestTextInput:
    """Tests for TextInput message."""

    def test_valid_text_input(self):
        msg = TextInput(text="Hello world")
        assert msg.type == MessageType.TEXT
        assert msg.text == "Hello world"
        assert msg.api_version == API_VERSION

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            TextInput(text="")

    def test_serialization(self):
        msg = TextInput(text="Hello", timestamp=1234567890)
        data = msg.to_dict()
        assert data["type"] == "text"
        assert data["text"] == "Hello"
        assert data["timestamp"] == 1234567890

    def test_deserialization(self):
        data = {"type": "text", "text": "Hello world"}
        msg = deserialize_message(data)
        assert isinstance(msg, TextInput)
        assert msg.text == "Hello world"


class TestStubbornnessUpdate:
    """Tests for StubbornnessUpdate message."""

    def test_valid_level(self):
        msg = StubbornnessUpdate(level=50)
        assert msg.type == MessageType.STUBBORNNESS
        assert msg.level == 50

    def test_level_bounds(self):
        with pytest.raises(ValidationError):
            StubbornnessUpdate(level=-1)
        with pytest.raises(ValidationError):
            StubbornnessUpdate(level=101)

    def test_boundary_values(self):
        assert StubbornnessUpdate(level=0).level == 0
        assert StubbornnessUpdate(level=100).level == 100


class TestMicAudioData:
    """Tests for MicAudioData message."""

    def test_valid_audio(self):
        audio = [0.0, 0.1, -0.1, 0.5]
        msg = MicAudioData(audio=audio)
        assert msg.audio == audio

    def test_empty_audio_rejected(self):
        with pytest.raises(ValidationError):
            MicAudioData(audio=[])

    def test_audio_range_validation(self):
        # Valid range -1.0 to 1.0
        MicAudioData(audio=[-1.0, 0.0, 1.0])

        # Out of range should fail validation
        with pytest.raises(ValidationError, match="out of range"):
            MicAudioData(audio=[1.5, 0.0])
        with pytest.raises(ValidationError, match="out of range"):
            MicAudioData(audio=[-1.5, 0.0])


class TestInterruptRequest:
    """Tests for InterruptRequest message."""

    def test_with_text(self):
        msg = InterruptRequest(text="Wait!")
        assert msg.text == "Wait!"

    def test_without_text(self):
        msg = InterruptRequest()
        assert msg.text is None


class TestTranscript:
    """Tests for Transcript message."""

    def test_default_is_final(self):
        msg = Transcript(text="Hello")
        assert msg.is_final is True

    def test_explicit_is_final(self):
        msg = Transcript(text="Hello", is_final=False)
        assert msg.is_final is False


class TestLLMChunk:
    """Tests for LLMChunk message."""

    def test_basic_chunk(self):
        msg = LLMChunk(text="Hello")
        assert msg.text == "Hello"
        assert msg.is_final is None

    def test_final_chunk(self):
        msg = LLMChunk(text="Hello world", is_final=True)
        assert msg.is_final is True


class TestTTSAudio:
    """Tests for TTSAudio message."""

    def test_valid_audio(self):
        msg = TTSAudio(audio="base64data", is_final=False)
        assert msg.audio == "base64data"
        assert msg.is_final is False

    def test_with_chunk_index(self):
        msg = TTSAudio(audio="data", is_final=True, chunk_index=5)
        assert msg.chunk_index == 5

    def test_empty_audio_rejected(self):
        with pytest.raises(ValidationError):
            TTSAudio(audio="", is_final=True)

    def test_negative_chunk_index_rejected(self):
        with pytest.raises(ValidationError):
            TTSAudio(audio="data", is_final=True, chunk_index=-1)


class TestStateUpdate:
    """Tests for StateUpdate message."""

    def test_states(self):
        for state in StateEnum:
            msg = StateUpdate(state=state)
            assert msg.state == state

    def test_serialization(self):
        msg = StateUpdate(state=StateEnum.LISTENING)
        data = msg.to_dict()
        assert data["state"] == "listening"


class TestGatekeeperDecision:
    """Tests for GatekeeperDecision message."""

    def test_all_actions(self):
        for action in GatekeeperAction:
            msg = GatekeeperDecision(action=action, confidence=0.8)
            assert msg.action == action

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            GatekeeperDecision(action=GatekeeperAction.REPLY, confidence=1.5)
        with pytest.raises(ValidationError):
            GatekeeperDecision(action=GatekeeperAction.REPLY, confidence=-0.1)


class TestStatusMessage:
    """Tests for StatusMessage."""

    def test_valid_message(self):
        msg = StatusMessage(message="Context cleared")
        assert msg.message == "Context cleared"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            StatusMessage(message="")


class TestVADResult:
    """Tests for VADResult message."""

    def test_valid_result(self):
        msg = VADResult(is_speaking=True, energy=0.75)
        assert msg.is_speaking is True
        assert msg.energy == 0.75

    def test_energy_bounds(self):
        with pytest.raises(ValidationError):
            VADResult(is_speaking=False, energy=1.5)
        with pytest.raises(ValidationError):
            VADResult(is_speaking=False, energy=-0.1)


class TestClearContext:
    """Tests for ClearContext message."""

    def test_basic(self):
        msg = ClearContext()
        assert msg.type == MessageType.CLEAR


class TestMicAudioEnd:
    """Tests for MicAudioEnd message."""

    def test_basic(self):
        msg = MicAudioEnd()
        assert msg.type == MessageType.MIC_AUDIO_END


class TestAudioStart:
    """Tests for AudioStart message."""

    def test_basic(self):
        msg = AudioStart()
        assert msg.type == MessageType.AUDIO_START


class TestMessageDeserialization:
    """Tests for the deserialize_message function."""

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type' field"):
            deserialize_message({"text": "Hello"})

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            deserialize_message({"type": "unknown"})

    def test_invalid_data(self):
        with pytest.raises(ValueError, match="Failed to validate"):
            deserialize_message({"type": "text", "text": ""})


class TestJSONSchema:
    """Tests for JSON Schema generation."""

    def test_get_all_schemas(self):
        schemas = get_all_schemas()
        expected_types = [
            "text", "stubbornness", "interrupt", "clear",
            "mic-audio-data", "mic-audio-end", "audio_start",
            "transcript", "llm_chunk", "tts_audio", "state",
            "gatekeeper", "status", "vad"
        ]
        for msg_type in expected_types:
            assert msg_type in schemas
            assert "properties" in schemas[msg_type]

    def test_schema_includes_required_fields(self):
        schemas = get_all_schemas()
        # TextInput should require 'text' field
        text_schema = schemas["text"]
        assert "text" in text_schema["properties"]


class TestMessageTimestamps:
    """Tests for message timestamps."""

    def test_default_timestamp_none(self):
        msg = TextInput(text="Hello")
        assert msg.timestamp is None

    def test_explicit_timestamp(self):
        msg = TextInput(text="Hello", timestamp=1234567890)
        assert msg.timestamp == 1234567890


class TestAPIVersion:
    """Tests for API version field."""

    def test_default_api_version(self):
        msg = TextInput(text="Hello")
        assert msg.api_version == API_VERSION

    def test_custom_api_version(self):
        msg = TextInput(text="Hello", api_version="2.0.0")
        assert msg.api_version == "2.0.0"


class TestJSONSerialization:
    """Tests for JSON serialization."""

    def test_to_json(self):
        msg = TextInput(text="Hello", timestamp=1234567890)
        json_str = msg.to_json()
        data = json.loads(json_str)
        assert data["type"] == "text"
        assert data["text"] == "Hello"

    def test_json_roundtrip(self):
        original = TextInput(text="Hello world", timestamp=1234567890)
        json_str = original.to_json()
        data = json.loads(json_str)
        restored = deserialize_message(data)
        assert isinstance(restored, TextInput)
        assert restored.text == original.text
