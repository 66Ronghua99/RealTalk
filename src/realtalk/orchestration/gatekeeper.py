"""Gatekeeper (Intent Classifier) for the Orchestration Layer."""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..config import get_config
from ..logging_config import setup_logger

logger = setup_logger("realtalk.gatekeeper")


class Action(Enum):
    """Gatekeeper decision actions."""
    WAIT = "wait"          # Keep listening
    ACCUMULATE = "accumulate"  # Accumulate context
    INTERRUPT = "interrupt"    # User interrupted
    REPLY = "reply"        # Generate response


@dataclass
class GatekeeperInput:
    """Input to the gatekeeper."""
    text: str
    silence_duration_ms: int
    audio_energy: float
    is_speaking: bool
    previous_action: Optional[Action] = None


@dataclass
class GatekeeperOutput:
    """Output from the gatekeeper."""
    action: Action
    confidence: float
    reasoning: str


class BaseGatekeeper(ABC):
    """Base class for gatekeeper implementations."""

    @abstractmethod
    async def decide(self, input: GatekeeperInput) -> GatekeeperOutput:
        """Decide the next action based on input."""
        pass


class RuleBasedGatekeeper(BaseGatekeeper):
    """Rule-based gatekeeper implementation.

    Decision logic based on multi-modal fusion:
    - Silence duration + semantic completeness + tone
    """

    def __init__(
        self,
        wait_threshold_ms: int = 300,
        reply_threshold_ms: int = 500,
        accumulate_threshold_ms: int = 1500
    ):
        self.wait_threshold_ms = wait_threshold_ms
        self.reply_threshold_ms = reply_threshold_ms
        self.accumulate_threshold_ms = accumulate_threshold_ms

        # Patterns for detecting semantic completeness
        self._complete_patterns = [
            r"[。！？\.!?]$",  # Ends with sentence-ending punctuation
            r"[的了吗呢嘛啊]$",  # Chinese sentence endings
            r"\b(?:对|是|好|行|可以|没问题)\b$",  # Confirmation words
        ]

        # Patterns for detecting interruption/intensity
        self._interruption_patterns = [
            r"\b(?:闭嘴|别说了|停下|安静|吵死了|胡说|滚|等等|等一下|stop|wait|)\b",
            r"[\!]{2,}",  # Multiple exclamation marks
            r"[A-Z]{3,}",  # All caps (shouting)
        ]

        # Patterns for low energy/backchannel
        self._backchannel_patterns = [
            r"^[,嗯啊哦呃]$",
            r"^[,嗯啊哦呃的]$",
            r"^[,嗯啊哦]$",
        ]

    def _is_semantically_complete(self, text: str) -> bool:
        """Check if the text is semantically complete."""
        if not text:
            return False

        for pattern in self._complete_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _is_interruption(self, text: str, energy: float) -> bool:
        """Check if the user is trying to interrupt."""
        if not text:
            return False

        # Check for interruption keywords
        for pattern in self._interruption_patterns:
            if re.search(pattern, text):
                return True

        # High energy can indicate interruption
        if energy > 0.7:
            return True

        return False

    def _is_backchannel(self, text: str, energy: float) -> bool:
        """Check if the user is just making backchannel sounds."""
        if not text:
            return False

        for pattern in self._backchannel_patterns:
            if re.match(pattern, text):
                return True

        # Low energy often indicates backchannel
        if energy < 0.2:
            return True

        return False

    async def decide(self, input: GatekeeperInput) -> GatekeeperOutput:
        """Decide the next action."""
        text = input.text
        silence_ms = input.silence_duration_ms
        energy = input.audio_energy

        # If user is currently speaking
        if input.is_speaking:
            if self._is_backchannel(text, energy):
                # Ignore background affirmation sounds
                return GatekeeperOutput(
                    action=Action.WAIT,
                    confidence=0.8,
                    reasoning="Backchannel detected, ignoring"
                )

            if self._is_interruption(text, energy):
                # User is trying to interrupt
                return GatekeeperOutput(
                    action=Action.INTERRUPT,
                    confidence=0.9,
                    reasoning="Interruption detected"
                )

            # User is speaking but not interrupting - continue listening
            return GatekeeperOutput(
                action=Action.WAIT,
                confidence=0.7,
                reasoning="User still speaking"
            )

        # User is not speaking - check silence duration
        if silence_ms < self.wait_threshold_ms:
            # Short silence, user might be thinking
            if self._is_semantically_complete(text):
                return GatekeeperOutput(
                    action=Action.WAIT,
                    confidence=0.6,
                    reasoning="Short silence with complete sentence, waiting"
                )
            return GatekeeperOutput(
                action=Action.WAIT,
                confidence=0.8,
                reasoning="Short silence, user might continue"
            )

        # Medium silence
        if silence_ms < self.reply_threshold_ms:
            if self._is_semantically_complete(text):
                return GatekeeperOutput(
                    action=Action.REPLY,
                    confidence=0.85,
                    reasoning="Medium silence with complete sentence"
                )
            return GatekeeperOutput(
                action=Action.WAIT,
                confidence=0.7,
                reasoning="Medium silence but incomplete"
            )

        # Long silence (potential accumulation)
        if silence_ms < self.accumulate_threshold_ms:
            return GatekeeperOutput(
                action=Action.REPLY,
                confidence=0.9,
                reasoning="Long silence, triggering reply"
            )

        # Very long silence - definitely reply
        return GatekeeperOutput(
            action=Action.REPLY,
            confidence=0.95,
            reasoning="Very long silence"
        )


class MLGatekeeper(BaseGatekeeper):
    """ML-based gatekeeper using BERT/DistilBERT."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        wait_threshold_ms: int = 300,
        reply_threshold_ms: int = 500,
        accumulate_threshold_ms: int = 1500
    ):
        self.model_name = model_name
        self.wait_threshold_ms = wait_threshold_ms
        self.reply_threshold_ms = reply_threshold_ms
        self.accumulate_threshold_ms = accumulate_threshold_ms
        self._model = None

    async def load(self) -> None:
        """Load the model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            logger.info(f"ML Gatekeeper loaded: {self.model_name}")
        except ImportError:
            logger.warning("transformers not installed, using rule-based")
            self._model = None

    async def decide(self, input: GatekeeperInput) -> GatekeeperOutput:
        """Decide using ML model."""
        if self._model is None:
            # Fallback to rule-based
            gatekeeper = RuleBasedGatekeeper(
                wait_threshold_ms=self.wait_threshold_ms,
                reply_threshold_ms=self.reply_threshold_ms,
                accumulate_threshold_ms=self.accumulate_threshold_ms
            )
            return await gatekeeper.decide(input)

        # Use ML model
        import torch

        # Prepare input
        text = input.text or "[SILENCE]"
        features = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self._model(**features)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        # Map class to action (this depends on training)
        action_map = {
            0: Action.WAIT,
            1: Action.ACCUMULATE,
            2: Action.INTERRUPT,
            3: Action.REPLY,
        }
        action = action_map.get(predicted_class, Action.WAIT)
        confidence = probs[0][predicted_class].item()

        return GatekeeperOutput(
            action=action,
            confidence=confidence,
            reasoning=f"ML prediction: {action.value}"
        )


async def create_gatekeeper(config: Optional[dict] = None) -> RuleBasedGatekeeper:
    """Factory function to create gatekeeper."""
    cfg = get_config()

    if config and config.get("use_ml"):
        gatekeeper = MLGatekeeper(
            model_name=config.get("model_name", "bert-base-uncased"),
            wait_threshold_ms=cfg.orchestration.wait_threshold_ms,
            reply_threshold_ms=cfg.orchestration.reply_threshold_ms,
            accumulate_threshold_ms=cfg.orchestration.accumulate_threshold_ms
        )
        await gatekeeper.load()
        return gatekeeper
    else:
        # Default to rule-based
        return RuleBasedGatekeeper(
            wait_threshold_ms=cfg.orchestration.wait_threshold_ms,
            reply_threshold_ms=cfg.orchestration.reply_threshold_ms,
            accumulate_threshold_ms=cfg.orchestration.accumulate_threshold_ms
        )
