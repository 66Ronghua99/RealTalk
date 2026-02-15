"""Configuration management for RealTalk."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ApiConfig(BaseModel):
    """API configuration."""
    minimax_api_key: str
    minimax_group_id: str
    openrouter_api_key: str


class VADConfig(BaseModel):
    """VAD configuration."""
    model_name: str = "silero"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    minSilence_duration_ms: int = 300


class ASRConfig(BaseModel):
    """ASR configuration."""
    model_name: str = "minimax"
    language: str = "auto"
    sample_rate: int = 16000


class LLMConfig(BaseModel):
    """LLM configuration."""
    model_name: str = "google/gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 1024


class TTSConfig(BaseModel):
    """TTS configuration."""
    model_name: str = "minimax"
    voice_id: str = "male-qn-qingse"
    sample_rate: int = 32000


class OrchestrationConfig(BaseModel):
    """Orchestration configuration."""
    wait_threshold_ms: int = 300
    reply_threshold_ms: int = 400
    accumulate_threshold_ms: int = 1500
    stubbornness_level: int = 50  # 0-100


class Config(BaseSettings):
    """Main configuration."""
    api: ApiConfig
    vad: VADConfig = VADConfig()
    asr: ASRConfig = ASRConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()

    @classmethod
    def load(cls, env_path: Optional[Path] = None) -> "Config":
        """Load configuration from environment variables."""
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()

        return cls(
            api=ApiConfig(
                minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
                minimax_group_id=os.getenv("MINIMAX_GROUP_ID", ""),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            )
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config
