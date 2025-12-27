"""
Tests for reward functions.
"""

import pytest

from judicaita.training.rewards import (
    CompositeReward,
    FormatReward,
    OutcomeReward,
    VerbosityReward,
)


class TestFormatReward:
    """Tests for FormatReward class."""

    def test_format_reward_with_steps_and_conclusion(self) -> None:
        """Test format reward with proper structure."""
        reward = FormatReward(min_steps=2, require_conclusion=True)

        response = """Step 1: Identify the rule
Step 2: Apply to facts
Conclusion: The answer is X"""

        result = reward.compute("test prompt", response, "")

        assert result.score == 1.0
        assert result.details["num_steps"] == 2
        assert result.details["has_conclusion"] is True

    def test_format_reward_missing_conclusion(self) -> None:
        """Test format reward when conclusion is missing."""
        reward = FormatReward(min_steps=2, require_conclusion=True)

        response = """Step 1: Identify the rule
Step 2: Apply to facts"""

        result = reward.compute("test prompt", response, "")

        assert result.score == 0.6  # Only step score
        assert result.details["has_conclusion"] is False

    def test_format_reward_insufficient_steps(self) -> None:
        """Test format reward with insufficient steps."""
        reward = FormatReward(min_steps=3, require_conclusion=True)

        response = """Step 1: Identify the rule
Conclusion: The answer is X"""

        result = reward.compute("test prompt", response, "")

        assert result.score < 1.0
        assert result.details["num_steps"] == 1


class TestOutcomeReward:
    """Tests for OutcomeReward class."""

    def test_outcome_reward_with_match(self) -> None:
        """Test outcome reward with matching conclusion."""
        reward = OutcomeReward(use_semantic_similarity=True, threshold=0.5)

        response = """Step 1: Analysis
Conclusion: The contract is valid"""

        reference = "The contract is valid"

        result = reward.compute("test prompt", response, reference)

        assert result.score > 0.0
        assert "conclusion" in result.details

    def test_outcome_reward_no_reference(self) -> None:
        """Test outcome reward without reference."""
        reward = OutcomeReward()

        result = reward.compute("test prompt", "any response", "")

        assert result.score == 0.5
        assert result.details["no_reference"] is True


class TestVerbosityReward:
    """Tests for VerbosityReward class."""

    def test_verbosity_reward_optimal_length(self) -> None:
        """Test verbosity reward with optimal length."""
        reward = VerbosityReward(target_length=10, tolerance=0.5)

        response = "This is exactly ten words in total for testing purposes here"

        result = reward.compute("test prompt", response, "")

        assert result.score == 1.0

    def test_verbosity_reward_too_short(self) -> None:
        """Test verbosity reward when response is too short."""
        reward = VerbosityReward(min_length=20)

        response = "Too short"

        result = reward.compute("test prompt", response, "")

        assert result.score < 1.0
        assert result.details["reason"] == "too_short"


class TestCompositeReward:
    """Tests for CompositeReward class."""

    def test_composite_reward_combination(self) -> None:
        """Test composite reward combines multiple signals."""
        reward = CompositeReward(
            format_weight=0.3, outcome_weight=0.5, verbosity_weight=0.2
        )

        response = """Step 1: Identify rule
Step 2: Apply facts
Conclusion: Result is X"""

        result = reward.compute("test prompt", response, "Result is X")

        assert 0.0 <= result.score <= 1.0
        assert "format" in result.details
        assert "outcome" in result.details
        assert "verbosity" in result.details

    def test_composite_reward_batch(self) -> None:
        """Test batch reward computation."""
        reward = CompositeReward()

        prompts = ["prompt1", "prompt2"]
        responses = ["response1", "response2"]
        references = ["ref1", "ref2"]

        rewards_tensor = reward.compute_batch(prompts, responses, references)

        assert len(rewards_tensor) == 2
        assert rewards_tensor.dtype.is_floating_point
