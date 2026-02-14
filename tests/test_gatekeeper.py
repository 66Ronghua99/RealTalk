"""Tests for orchestration.gatekeeper."""
import pytest

from realtalk.orchestration.gatekeeper import (
    Action,
    GatekeeperInput,
    RuleBasedGatekeeper,
)


class TestRuleBasedGatekeeper:
    """Test RuleBasedGatekeeper decisions."""

    @pytest.fixture
    def gatekeeper(self):
        """Create a gatekeeper instance."""
        return RuleBasedGatekeeper(
            wait_threshold_ms=300,
            reply_threshold_ms=500,
            accumulate_threshold_ms=1500
        )

    @pytest.mark.asyncio
    async def test_short_silence_incomplete_sentence(self, gatekeeper):
        """Short silence with incomplete sentence -> WAIT."""
        input_data = GatekeeperInput(
            text="我觉得这个",
            silence_duration_ms=200,
            audio_energy=0.3,
            is_speaking=False
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.WAIT

    @pytest.mark.asyncio
    async def test_medium_silence_complete_sentence(self, gatekeeper):
        """Medium silence with complete sentence -> REPLY."""
        input_data = GatekeeperInput(
            text="我觉得这个方案不行。",
            silence_duration_ms=400,
            audio_energy=0.3,
            is_speaking=False
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.REPLY

    @pytest.mark.asyncio
    async def test_long_silence(self, gatekeeper):
        """Long silence -> REPLY."""
        input_data = GatekeeperInput(
            text="好的",
            silence_duration_ms=2000,
            audio_energy=0.3,
            is_speaking=False
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.REPLY

    @pytest.mark.asyncio
    async def test_user_speaking_backchannel(self, gatekeeper):
        """User speaking with backchannel -> WAIT (ignore)."""
        input_data = GatekeeperInput(
            text="嗯",
            silence_duration_ms=0,
            audio_energy=0.1,  # Low energy
            is_speaking=True
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.WAIT

    @pytest.mark.asyncio
    async def test_user_speaking_interruption(self, gatekeeper):
        """User speaking with interruption keywords -> INTERRUPT."""
        input_data = GatekeeperInput(
            text="闭嘴！你在胡说！",
            silence_duration_ms=0,
            audio_energy=0.8,  # High energy
            is_speaking=True
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.INTERRUPT

    @pytest.mark.asyncio
    async def test_confirmation_word(self, gatekeeper):
        """Confirmation word triggers REPLY."""
        input_data = GatekeeperInput(
            text="可以",
            silence_duration_ms=600,
            audio_energy=0.3,
            is_speaking=False
        )

        result = await gatekeeper.decide(input_data)
        assert result.action == Action.REPLY
