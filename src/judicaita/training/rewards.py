"""
Reward functions for GRPO training pipeline.

This module provides various reward functions for evaluating model outputs
during GRPO training, including format rewards, outcome rewards, and
verbosity/conciseness balance.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger


@dataclass
class RewardResult:
    """Result of a reward computation."""

    score: float  # Reward score in range [0, 1]
    details: dict[str, Any]  # Additional details about the reward


class BaseReward(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """
        Compute reward for a model response.

        Args:
            prompt: Input prompt
            response: Model response
            reference: Optional reference response for comparison

        Returns:
            RewardResult with score and details
        """
        pass


class FormatReward(BaseReward):
    """
    Reward function for step-by-step reasoning format validation.

    Checks if the response follows the expected format:
    - Step 1: ...
    - Step 2: ...
    - Conclusion: ...
    """

    def __init__(
        self,
        min_steps: int = 2,
        require_conclusion: bool = True,
        step_pattern: str = r"Step\s+\d+:",
        conclusion_pattern: str = r"Conclusion:",
    ) -> None:
        """
        Initialize format reward.

        Args:
            min_steps: Minimum number of steps required
            require_conclusion: Whether a conclusion is required
            step_pattern: Regex pattern for step markers
            conclusion_pattern: Regex pattern for conclusion marker
        """
        self.min_steps = min_steps
        self.require_conclusion = require_conclusion
        self.step_pattern = step_pattern
        self.conclusion_pattern = conclusion_pattern

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute format reward based on step-by-step structure."""
        # Find all step markers
        steps = re.findall(self.step_pattern, response, re.IGNORECASE)
        num_steps = len(steps)

        # Check for conclusion
        has_conclusion = bool(re.search(self.conclusion_pattern, response, re.IGNORECASE))

        # Calculate score
        score = 0.0

        # Step count reward (0.6 weight)
        if num_steps >= self.min_steps:
            score += 0.6
        else:
            score += 0.6 * (num_steps / self.min_steps)

        # Conclusion reward (0.4 weight)
        if self.require_conclusion:
            if has_conclusion:
                score += 0.4
        else:
            score += 0.4  # Full credit if not required

        details = {
            "num_steps": num_steps,
            "has_conclusion": has_conclusion,
            "meets_min_steps": num_steps >= self.min_steps,
        }

        return RewardResult(score=score, details=details)


class OutcomeReward(BaseReward):
    """
    Reward function based on outcome correctness.

    Evaluates whether the model's conclusion matches the expected answer
    for LegalBench tasks.
    """

    def __init__(self, use_semantic_similarity: bool = True, threshold: float = 0.8) -> None:
        """
        Initialize outcome reward.

        Args:
            use_semantic_similarity: Whether to use semantic similarity vs exact match
            threshold: Similarity threshold for positive reward
        """
        self.use_semantic_similarity = use_semantic_similarity
        self.threshold = threshold

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute outcome reward based on correctness."""
        if not reference:
            # No reference available, return neutral score
            return RewardResult(score=0.5, details={"no_reference": True})

        # Extract conclusion from response
        conclusion = self._extract_conclusion(response)

        if self.use_semantic_similarity:
            # Use simple token overlap as proxy for semantic similarity
            # In production, use a proper semantic similarity model
            similarity = self._compute_token_overlap(conclusion, reference)
        else:
            # Exact match (case-insensitive)
            similarity = 1.0 if conclusion.lower() == reference.lower() else 0.0

        # Binary reward based on threshold
        score = 1.0 if similarity >= self.threshold else 0.0

        details = {
            "conclusion": conclusion,
            "reference": reference,
            "similarity": similarity,
            "threshold": self.threshold,
        }

        return RewardResult(score=score, details=details)

    def _extract_conclusion(self, response: str) -> str:
        """Extract conclusion from response."""
        # Look for conclusion section
        match = re.search(
            r"Conclusion:\s*(.+?)(?:\n\n|$)", response, re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()

        # Fallback: use last sentence
        sentences = response.strip().split(".")
        return sentences[-1].strip() if sentences else ""

    def _compute_token_overlap(self, text1: str, text2: str) -> float:
        """Compute token overlap similarity between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0


class VerbosityReward(BaseReward):
    """
    Reward function for balancing verbosity and conciseness.

    Penalizes responses that are too short or too long, encouraging
    optimal length for legal reasoning.
    """

    def __init__(
        self,
        target_length: int = 500,
        tolerance: float = 0.5,
        min_length: int = 100,
        max_length: int = 2000,
    ) -> None:
        """
        Initialize verbosity reward.

        Args:
            target_length: Target response length in tokens
            tolerance: Tolerance factor for length deviation (0-1)
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
        """
        self.target_length = target_length
        self.tolerance = tolerance
        self.min_length = min_length
        self.max_length = max_length

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute verbosity reward based on response length."""
        # Approximate token count by word count
        response_length = len(response.split())

        # Check hard limits
        if response_length < self.min_length:
            score = max(0.0, response_length / self.min_length)
            details = {"length": response_length, "reason": "too_short"}
            return RewardResult(score=score, details=details)

        if response_length > self.max_length:
            score = max(0.0, 1.0 - (response_length - self.max_length) / self.max_length)
            details = {"length": response_length, "reason": "too_long"}
            return RewardResult(score=score, details=details)

        # Compute score based on distance from target
        deviation = abs(response_length - self.target_length) / self.target_length

        if deviation <= self.tolerance:
            # Within tolerance, full score
            score = 1.0
        else:
            # Outside tolerance, linear decay
            score = max(0.0, 1.0 - (deviation - self.tolerance) / (1.0 - self.tolerance))

        details = {
            "length": response_length,
            "target": self.target_length,
            "deviation": deviation,
            "reason": "within_range",
        }

        return RewardResult(score=score, details=details)


class CompositeReward(BaseReward):
    """
    Composite reward function that combines multiple reward signals.

    Aggregates format, outcome, and verbosity rewards with configurable weights.
    """

    def __init__(
        self,
        format_weight: float = 0.3,
        outcome_weight: float = 0.5,
        verbosity_weight: float = 0.2,
        format_reward: FormatReward | None = None,
        outcome_reward: OutcomeReward | None = None,
        verbosity_reward: VerbosityReward | None = None,
    ) -> None:
        """
        Initialize composite reward.

        Args:
            format_weight: Weight for format reward
            outcome_weight: Weight for outcome reward
            verbosity_weight: Weight for verbosity reward
            format_reward: Format reward instance (creates default if None)
            outcome_reward: Outcome reward instance (creates default if None)
            verbosity_reward: Verbosity reward instance (creates default if None)
        """
        # Normalize weights
        total_weight = format_weight + outcome_weight + verbosity_weight
        self.format_weight = format_weight / total_weight
        self.outcome_weight = outcome_weight / total_weight
        self.verbosity_weight = verbosity_weight / total_weight

        # Initialize individual reward functions
        self.format_reward = format_reward or FormatReward()
        self.outcome_reward = outcome_reward or OutcomeReward()
        self.verbosity_reward = verbosity_reward or VerbosityReward()

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute composite reward by combining individual rewards."""
        # Compute individual rewards
        format_result = self.format_reward.compute(prompt, response, reference)
        outcome_result = self.outcome_reward.compute(prompt, response, reference)
        verbosity_result = self.verbosity_reward.compute(prompt, response, reference)

        # Compute weighted sum
        composite_score = (
            self.format_weight * format_result.score
            + self.outcome_weight * outcome_result.score
            + self.verbosity_weight * verbosity_result.score
        )

        details = {
            "format": {
                "score": format_result.score,
                "weight": self.format_weight,
                "details": format_result.details,
            },
            "outcome": {
                "score": outcome_result.score,
                "weight": self.outcome_weight,
                "details": outcome_result.details,
            },
            "verbosity": {
                "score": verbosity_result.score,
                "weight": self.verbosity_weight,
                "details": verbosity_result.details,
            },
            "composite_score": composite_score,
        }

        return RewardResult(score=composite_score, details=details)

    def compute_batch(
        self, prompts: list[str], responses: list[str], references: list[str] | None = None
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of responses.

        Args:
            prompts: List of input prompts
            responses: List of model responses
            references: Optional list of reference responses

        Returns:
            Tensor of reward scores
        """
        if references is None:
            references = [""] * len(prompts)

        rewards = []

        for prompt, response, reference in zip(prompts, responses, references):
            result = self.compute(prompt, response, reference)
            rewards.append(result.score)

        return torch.tensor(rewards, dtype=torch.float32)
