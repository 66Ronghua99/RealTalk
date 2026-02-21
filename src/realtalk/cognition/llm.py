"""Large Language Model (LLM) module using OpenRouter/Gemini."""
import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from ..config import get_config
from ..exceptions import LLMError
from ..logging_config import setup_logger

logger = setup_logger("realtalk.llm")


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """LLM response."""
    content: str
    model: str
    finish_reason: str
    usage: Optional[Dict[str, int]] = None


class BaseLLM(ABC):
    """Base class for LLM implementations."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Send chat request."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[LLMResponse]:
        """Send chat request with streaming."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the LLM."""
        pass


class OpenRouterLLM(BaseLLM):
    """OpenRouter LLM implementation (supports Gemini, etc.)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "google/gemini-3-flash-preview",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = "https://openrouter.ai/api/v1"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        need_new_session = self._session is None or self._session.closed

        # Check if the session's event loop is still valid
        # by attempting to get the running loop and comparing
        if self._session is not None and not self._session.closed:
            try:
                current_loop = asyncio.get_running_loop()
                session_loop = getattr(self._session, '_loop', None)
                if session_loop is not None and session_loop is not current_loop:
                    logger.debug("Event loop mismatch, creating new aiohttp session")
                    need_new_session = True
                    await self._session.close()
            except RuntimeError:
                # No event loop running, need new session
                need_new_session = True
                await self._session.close()

        if need_new_session:
            self._session = aiohttp.ClientSession()
        return self._session

    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Send chat request."""
        url = f"{self._base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://realtalk.ai",
            "X-Title": "RealTalk",
        }

        # Build messages list
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        payload = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    raise LLMError(f"LLM API error: {response.status}")

                result = await response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    return LLMResponse(
                        content=choice["message"]["content"],
                        model=result.get("model", self.model_name),
                        finish_reason=choice.get("finish_reason", "stop"),
                        usage=result.get("usage")
                    )
                else:
                    raise LLMError("No response in LLM result")

        except LLMError:
            raise
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise LLMError(f"LLM chat failed: {e}")

    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[LLMResponse]:
        """Send chat request with streaming."""
        url = f"{self._base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://realtalk.ai",
            "X-Title": "RealTalk",
        }

        # Build messages list
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        payload = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM stream error: {response.status} - {error_text}")
                    return

                content_buffer = ""

                async for line in response.content:
                    line = line.decode().strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_buffer += delta["content"]
                                    yield LLMResponse(
                                        content=delta["content"],
                                        model=data.get("model", self.model_name),
                                        finish_reason="",
                                    )
                                if data["choices"][0].get("finish_reason"):
                                    yield LLMResponse(
                                        content=delta["content"],
                                        model=data.get("model", self.model_name),
                                        finish_reason=data["choices"][0]["finish_reason"],
                                        usage=data.get("usage")
                                    )
                        except json.JSONDecodeError:
                            continue

        except asyncio.CancelledError:
            logger.info("LLM stream cancelled")
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            raise LLMError(f"LLM stream failed: {e}")

    async def close(self) -> None:
        """Close the LLM."""
        if self._session and not self._session.closed:
            await self._session.close()


class QwenLLM(BaseLLM):
    """Qwen LLM implementation (local)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        base_url: str = "http://localhost:8080/v1",
        api_key: str = "empty",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        need_new_session = self._session is None or self._session.closed

        # Check if the session's event loop is still valid
        # by attempting to get the running loop and comparing
        if self._session is not None and not self._session.closed:
            try:
                current_loop = asyncio.get_running_loop()
                session_loop = getattr(self._session, '_loop', None)
                if session_loop is not None and session_loop is not current_loop:
                    logger.debug("Event loop mismatch, creating new aiohttp session")
                    need_new_session = True
                    await self._session.close()
            except RuntimeError:
                # No event loop running, need new session
                need_new_session = True
                await self._session.close()

        if need_new_session:
            self._session = aiohttp.ClientSession()
        return self._session

    async def chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Send chat request."""
        url = f"{self.base_url}/chat/completions"

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        payload = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(f"Qwen API error: {response.status} - {error_text}")

                result = await response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    return LLMResponse(
                        content=choice["message"]["content"],
                        model=result.get("model", self.model_name),
                        finish_reason=choice.get("finish_reason", "stop"),
                        usage=result.get("usage")
                    )
                else:
                    raise LLMError("No response in Qwen result")

        except Exception as e:
            logger.error(f"Qwen chat error: {e}")
            raise LLMError(f"Qwen chat failed: {e}")

    async def stream_chat(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[LLMResponse]:
        """Send chat request with streaming."""
        url = f"{self.base_url}/chat/completions"

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        payload = {
            "model": self.model_name,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Qwen stream error: {response.status} - {error_text}")
                    return

                content_buffer = ""

                async for line in response.content:
                    line = line.decode().strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_buffer += delta["content"]
                                yield LLMResponse(
                                    content=delta["content"],
                                    model=self.model_name,
                                    finish_reason="",
                                )
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Qwen stream error: {e}")
            raise LLMError(f"Qwen stream failed: {e}")

    async def close(self) -> None:
        """Close the LLM."""
        if self._session and not self._session.closed:
            await self._session.close()


async def create_llm(config: Optional[dict] = None) -> OpenRouterLLM:
    """Factory function to create LLM instance."""
    cfg = get_config()

    if config and config.get("model_name", "").startswith("Qwen"):
        return QwenLLM(
            model_name=config.get("model_name", "Qwen/Qwen2-7B-Instruct"),
            base_url=config.get("base_url", "http://localhost:8080/v1")
        )
    else:
        # Default to OpenRouter
        return OpenRouterLLM(
            api_key=cfg.api.openrouter_api_key,
            model_name=cfg.llm.model_name,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens
        )
