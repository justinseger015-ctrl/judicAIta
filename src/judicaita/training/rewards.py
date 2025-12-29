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
        match = re.search(r"Conclusion:\s*(.+?)(?:\n\n|$)", response, re.IGNORECASE | re.DOTALL)

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


class CitationAccuracyReward(BaseReward):
    """
    Reward function for evaluating citation accuracy in legal responses.

    Checks for presence and proper formatting of legal citations
    (U.S.C., case names, section symbols, etc.).
    """

    # Common legal citation patterns
    CITATION_PATTERNS = [
        r"\d+\s+U\.S\.C\.\s+§?\s*\d+",  # Federal code: 42 U.S.C. § 1983
        r"\d+\s+U\.S\.\s+\d+",  # Supreme Court: 347 U.S. 483
        r"\d+\s+F\.\d+d\s+\d+",  # Federal Reporter: 123 F.2d 456
        r"\d+\s+F\.\s*Supp\.\s*\d*d?\s+\d+",  # F. Supp. citations
        r"[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+",  # Case names: Brown v. Board
        r"§\s*\d+[\.\d]*",  # Section symbols: § 1234
        r"\d+\s+S\.\s*Ct\.\s+\d+",  # Supreme Court Reporter
        r"\d+\s+L\.\s*Ed\.\s*\d*d?\s+\d+",  # Lawyers' Edition
    ]

    def __init__(
        self,
        min_citations: int = 1,
        bonus_per_citation: float = 0.1,
        max_bonus: float = 0.3,
    ) -> None:
        """
        Initialize citation accuracy reward.

        Args:
            min_citations: Minimum citations for full base score
            bonus_per_citation: Bonus score per additional citation
            max_bonus: Maximum bonus for multiple citations
        """
        self.min_citations = min_citations
        self.bonus_per_citation = bonus_per_citation
        self.max_bonus = max_bonus
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.CITATION_PATTERNS]

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute citation accuracy reward based on legal citations present."""
        citations_found = []

        for pattern in self._compiled_patterns:
            matches = pattern.findall(response)
            citations_found.extend(matches)

        num_citations = len(citations_found)

        # Base score: 0.7 if minimum citations present, scaled otherwise
        if num_citations >= self.min_citations:
            base_score = 0.7
        else:
            base_score = 0.7 * (num_citations / self.min_citations) if self.min_citations > 0 else 0.0

        # Bonus for additional citations
        extra_citations = max(0, num_citations - self.min_citations)
        bonus = min(extra_citations * self.bonus_per_citation, self.max_bonus)

        score = min(1.0, base_score + bonus)

        details = {
            "num_citations": num_citations,
            "citations_found": citations_found[:5],  # First 5 for debugging
            "meets_minimum": num_citations >= self.min_citations,
            "base_score": base_score,
            "bonus": bonus,
        }

        return RewardResult(score=score, details=details)


class ClarityReward(BaseReward):
    """
    Reward function for evaluating clarity of legal responses.

    Uses readability metrics and sentence structure analysis to
    evaluate plain-English quality.
    """

    def __init__(
        self,
        target_avg_sentence_length: int = 20,
        max_sentence_length: int = 50,
        target_avg_word_length: float = 5.0,
    ) -> None:
        """
        Initialize clarity reward.

        Args:
            target_avg_sentence_length: Target average sentence length in words
            max_sentence_length: Maximum acceptable sentence length
            target_avg_word_length: Target average word length in characters
        """
        self.target_avg_sentence_length = target_avg_sentence_length
        self.max_sentence_length = max_sentence_length
        self.target_avg_word_length = target_avg_word_length

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute clarity reward based on readability metrics."""
        if not response.strip():
            return RewardResult(score=0.0, details={"reason": "empty_response"})

        # Split into sentences
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return RewardResult(score=0.0, details={"reason": "no_sentences"})

        # Calculate metrics
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

        # Count long sentences
        long_sentences = sum(1 for length in sentence_lengths if length > self.max_sentence_length)
        long_sentence_ratio = long_sentences / len(sentences)

        # Calculate average word length
        words = response.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
        else:
            avg_word_length = 0.0

        # Compute component scores
        # Sentence length score (prefer target, penalize extremes)
        sentence_length_deviation = abs(avg_sentence_length - self.target_avg_sentence_length)
        sentence_length_score = max(0.0, 1.0 - sentence_length_deviation / self.target_avg_sentence_length)

        # Penalize long sentences
        long_sentence_penalty = 1.0 - (long_sentence_ratio * 0.5)

        # Word complexity score (prefer moderate word length)
        word_length_deviation = abs(avg_word_length - self.target_avg_word_length)
        word_length_score = max(0.0, 1.0 - word_length_deviation / self.target_avg_word_length)

        # Combine scores (weights: sentence length 40%, long sentence penalty 30%, word length 30%)
        score = (
            0.4 * sentence_length_score
            + 0.3 * long_sentence_penalty
            + 0.3 * word_length_score
        )

        details = {
            "avg_sentence_length": round(avg_sentence_length, 1),
            "num_sentences": len(sentences),
            "long_sentences": long_sentences,
            "avg_word_length": round(avg_word_length, 1),
            "sentence_length_score": round(sentence_length_score, 2),
            "word_length_score": round(word_length_score, 2),
        }

        return RewardResult(score=score, details=details)


class CompositeReward(BaseReward):
    """
    Composite reward function that combines multiple reward signals.

    Aggregates correctness, reasoning quality, citation accuracy, and clarity rewards
    with configurable weights. Default weights align with the Phase 2 specification:
    - Correctness: 40%
    - Reasoning Quality: 30%
    - Citation Accuracy: 20%
    - Clarity: 10%
    """

    def __init__(
        self,
        correctness_weight: float = 0.4,
        reasoning_quality_weight: float = 0.3,
        citation_accuracy_weight: float = 0.2,
        clarity_weight: float = 0.1,
        correctness_reward: OutcomeReward | None = None,
        reasoning_quality_reward: FormatReward | None = None,
        citation_accuracy_reward: CitationAccuracyReward | None = None,
        clarity_reward: ClarityReward | None = None,
        # Legacy support for old 3-component interface
        format_weight: float | None = None,
        outcome_weight: float | None = None,
        verbosity_weight: float | None = None,
        format_reward: FormatReward | None = None,
        outcome_reward: OutcomeReward | None = None,
        verbosity_reward: VerbosityReward | None = None,
    ) -> None:
        """
        Initialize composite reward.

        Args:
            correctness_weight: Weight for correctness reward (40% default)
            reasoning_quality_weight: Weight for reasoning quality reward (30% default)
            citation_accuracy_weight: Weight for citation accuracy reward (20% default)
            clarity_weight: Weight for clarity reward (10% default)
            correctness_reward: Correctness reward instance (creates default if None)
            reasoning_quality_reward: Reasoning quality reward instance (creates default if None)
            citation_accuracy_reward: Citation accuracy reward instance (creates default if None)
            clarity_reward: Clarity reward instance (creates default if None)

        Legacy Args (for backward compatibility):
            format_weight: Maps to reasoning_quality_weight
            outcome_weight: Maps to correctness_weight
            verbosity_weight: Ignored in 4-component mode
            format_reward: Maps to reasoning_quality_reward
            outcome_reward: Maps to correctness_reward
            verbosity_reward: Ignored in 4-component mode
        """
        # Handle legacy 3-component interface for backward compatibility
        if format_weight is not None or outcome_weight is not None or verbosity_weight is not None:
            # Legacy mode: use old 3-component weights but map to new 4-component structure
            fw = format_weight if format_weight is not None else 0.3
            ow = outcome_weight if outcome_weight is not None else 0.5
            vw = verbosity_weight if verbosity_weight is not None else 0.2

            # Redistribute weights to 4-component structure
            total_legacy = fw + ow + vw
            correctness_weight = ow / total_legacy * 0.4 / 0.5  # Scale to new system
            reasoning_quality_weight = fw / total_legacy * 0.3 / 0.3
            citation_accuracy_weight = 0.2
            clarity_weight = 0.1

            # Map legacy reward instances
            if outcome_reward is not None:
                correctness_reward = outcome_reward
            if format_reward is not None:
                reasoning_quality_reward = format_reward

        # Normalize weights
        total_weight = (
            correctness_weight
            + reasoning_quality_weight
            + citation_accuracy_weight
            + clarity_weight
        )
        self.correctness_weight = correctness_weight / total_weight
        self.reasoning_quality_weight = reasoning_quality_weight / total_weight
        self.citation_accuracy_weight = citation_accuracy_weight / total_weight
        self.clarity_weight = clarity_weight / total_weight

        # Initialize individual reward functions
        self.correctness_reward = correctness_reward or OutcomeReward()
        self.reasoning_quality_reward = reasoning_quality_reward or FormatReward()
        self.citation_accuracy_reward = citation_accuracy_reward or CitationAccuracyReward()
        self.clarity_reward = clarity_reward or ClarityReward()

        # Legacy attribute aliases for backward compatibility
        self.format_weight = self.reasoning_quality_weight
        self.outcome_weight = self.correctness_weight
        self.verbosity_weight = self.clarity_weight
        self.format_reward = self.reasoning_quality_reward
        self.outcome_reward = self.correctness_reward
        self.verbosity_reward = VerbosityReward()  # Keep for legacy tests

    def compute(self, prompt: str, response: str, reference: str = "") -> RewardResult:
        """Compute composite reward by combining all 4 component rewards."""
        # Compute individual rewards
        correctness_result = self.correctness_reward.compute(prompt, response, reference)
        reasoning_quality_result = self.reasoning_quality_reward.compute(prompt, response, reference)
        citation_accuracy_result = self.citation_accuracy_reward.compute(prompt, response, reference)
        clarity_result = self.clarity_reward.compute(prompt, response, reference)

        # Compute weighted sum
        composite_score = (
            self.correctness_weight * correctness_result.score
            + self.reasoning_quality_weight * reasoning_quality_result.score
            + self.citation_accuracy_weight * citation_accuracy_result.score
            + self.clarity_weight * clarity_result.score
        )

        details = {
            "correctness": {
                "score": correctness_result.score,
                "weight": self.correctness_weight,
                "details": correctness_result.details,
            },
            "reasoning_quality": {
                "score": reasoning_quality_result.score,
                "weight": self.reasoning_quality_weight,
                "details": reasoning_quality_result.details,
            },
            "citation_accuracy": {
                "score": citation_accuracy_result.score,
                "weight": self.citation_accuracy_weight,
                "details": citation_accuracy_result.details,
            },
            "clarity": {
                "score": clarity_result.score,
                "weight": self.clarity_weight,
                "details": clarity_result.details,
            },
            # Legacy keys for backward compatibility
            "format": {
                "score": reasoning_quality_result.score,
                "weight": self.reasoning_quality_weight,
                "details": reasoning_quality_result.details,
            },
            "outcome": {
                "score": correctness_result.score,
                "weight": self.correctness_weight,
                "details": correctness_result.details,
            },
            "verbosity": {
                "score": clarity_result.score,
                "weight": self.clarity_weight,
                "details": clarity_result.details,
            },
            "composite_score": composite_score,
        }

        return RewardResult(score=composite_score, details=details)

    def validate_reward_output(self, reward_result: RewardResult) -> bool:
        """
        Verify reward function returns expected 4-component structure.

        Args:
            reward_result: The result to validate

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        required_components = ["correctness", "reasoning_quality", "citation_accuracy", "clarity"]

        for component in required_components:
            if component not in reward_result.details:
                raise ValueError(f"Missing reward component: {component}")
            if "score" not in reward_result.details[component]:
                raise ValueError(f"Missing score for component: {component}")

        # Verify total weight sums to ~1.0
        total_weight = sum(
            reward_result.details[c]["weight"] for c in required_components
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Component weights do not sum to 1.0: {total_weight}")

        # Verify composite score is normalized
        if reward_result.score < 0.0 or reward_result.score > 1.0:
            raise ValueError(f"Composite score out of range [0, 1]: {reward_result.score}")

        return True

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

        for prompt, response, reference in zip(prompts, responses, references, strict=False):
            result = self.compute(prompt, response, reference)
            rewards.append(result.score)

        return torch.tensor(rewards, dtype=torch.float32)
