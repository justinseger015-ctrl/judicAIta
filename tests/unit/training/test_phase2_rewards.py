"""
Tests for new Phase 2 reward components.
"""

import pytest

from judicaita.training.rewards import (
    CitationAccuracyReward,
    ClarityReward,
    CompositeReward,
)


class TestCitationAccuracyReward:
    """Tests for CitationAccuracyReward class."""

    def test_citation_reward_with_legal_citations(self) -> None:
        """Test citation reward with valid legal citations."""
        reward = CitationAccuracyReward(min_citations=1)

        response = """The statute 42 U.S.C. § 1983 provides a cause of action.
        As established in Brown v. Board of Education, segregation is unconstitutional."""

        result = reward.compute("test prompt", response, "")

        assert result.score > 0.0
        assert result.details["num_citations"] >= 2
        assert result.details["meets_minimum"] is True

    def test_citation_reward_with_federal_reporter(self) -> None:
        """Test citation reward with F.2d citations."""
        reward = CitationAccuracyReward(min_citations=1)

        response = "The court in 123 F.2d 456 held that..."

        result = reward.compute("test prompt", response, "")

        assert result.score > 0.0
        assert result.details["num_citations"] >= 1

    def test_citation_reward_no_citations(self) -> None:
        """Test citation reward when no citations present."""
        reward = CitationAccuracyReward(min_citations=2)

        response = "This is a legal argument without any proper citations."

        result = reward.compute("test prompt", response, "")

        assert result.score < 0.7
        assert result.details["num_citations"] == 0
        assert result.details["meets_minimum"] is False

    def test_citation_reward_section_symbol(self) -> None:
        """Test citation reward with section symbols."""
        reward = CitationAccuracyReward(min_citations=1)

        response = "Under § 1234.5, the requirement applies."

        result = reward.compute("test prompt", response, "")

        assert result.details["num_citations"] >= 1


class TestClarityReward:
    """Tests for ClarityReward class."""

    def test_clarity_reward_clear_writing(self) -> None:
        """Test clarity reward with clear, simple writing."""
        reward = ClarityReward(
            target_avg_sentence_length=20,
            max_sentence_length=50,
        )

        response = """The court found the defendant liable. 
        This decision was based on clear evidence. 
        The ruling affects similar future cases."""

        result = reward.compute("test prompt", response, "")

        assert result.score > 0.0
        assert "avg_sentence_length" in result.details
        assert "num_sentences" in result.details

    def test_clarity_reward_complex_writing(self) -> None:
        """Test clarity reward with overly complex sentences."""
        reward = ClarityReward(max_sentence_length=20)

        # Very long sentence
        response = (
            "The defendant's motion for summary judgment was denied by the court "
            "after considering the extensive evidence presented by the plaintiff "
            "which included numerous documents and witness testimonies that "
            "clearly demonstrated the defendant's liability in this matter "
            "as established by the applicable legal standards and precedents."
        )

        result = reward.compute("test prompt", response, "")

        # Should have lower score due to long sentences
        assert result.details["long_sentences"] >= 1

    def test_clarity_reward_empty_response(self) -> None:
        """Test clarity reward with empty response."""
        reward = ClarityReward()

        result = reward.compute("test prompt", "", "")

        assert result.score == 0.0
        assert result.details["reason"] == "empty_response"


class TestCompositeRewardFourComponents:
    """Tests for 4-component CompositeReward."""

    def test_composite_reward_has_four_components(self) -> None:
        """Test that composite reward returns all 4 components."""
        reward = CompositeReward()

        response = """Step 1: Identify the issue.
        Step 2: Apply 42 U.S.C. § 1983.
        Conclusion: The claim is valid."""

        result = reward.compute("test prompt", response, "valid claim")

        # Check all 4 new components are present
        assert "correctness" in result.details
        assert "reasoning_quality" in result.details
        assert "citation_accuracy" in result.details
        assert "clarity" in result.details

        # Check each has a score
        for component in ["correctness", "reasoning_quality", "citation_accuracy", "clarity"]:
            assert "score" in result.details[component]
            assert "weight" in result.details[component]

    def test_composite_reward_weights_sum_to_one(self) -> None:
        """Test that component weights sum to 1.0."""
        reward = CompositeReward(
            correctness_weight=0.4,
            reasoning_quality_weight=0.3,
            citation_accuracy_weight=0.2,
            clarity_weight=0.1,
        )

        total_weight = (
            reward.correctness_weight
            + reward.reasoning_quality_weight
            + reward.citation_accuracy_weight
            + reward.clarity_weight
        )

        assert abs(total_weight - 1.0) < 0.01

    def test_composite_reward_validate_output(self) -> None:
        """Test validate_reward_output method."""
        reward = CompositeReward()

        response = """Step 1: Analysis.
        Step 2: Apply 42 U.S.C. § 1983.
        Conclusion: Valid."""

        result = reward.compute("test prompt", response, "valid")

        # Validation should pass
        assert reward.validate_reward_output(result) is True

    def test_composite_reward_backward_compatibility(self) -> None:
        """Test backward compatibility with old 3-component interface."""
        reward = CompositeReward(
            format_weight=0.3,
            outcome_weight=0.5,
            verbosity_weight=0.2,
        )

        response = """Step 1: Analysis.
        Conclusion: Valid."""

        result = reward.compute("test prompt", response, "valid")

        # Legacy keys should be present
        assert "format" in result.details
        assert "outcome" in result.details
        assert "verbosity" in result.details

    def test_composite_reward_score_normalized(self) -> None:
        """Test that composite score is normalized between 0 and 1."""
        reward = CompositeReward()

        response = """Step 1: Identify the applicable statute 42 U.S.C. § 1983.
        Step 2: Apply the legal test to facts.
        Conclusion: The defendant is liable."""

        result = reward.compute("test prompt", response, "defendant is liable")

        assert 0.0 <= result.score <= 1.0
