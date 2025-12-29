"""
Tests for ground truth validation in data curation.
"""

import pytest
from datasets import Dataset

from judicaita.training.data_curation import (
    GroundTruthValidationError,
    LegalBenchDataset,
)


class TestGroundTruthValidation:
    """Tests for ground_truth field validation."""

    def test_validate_ground_truth_success(self) -> None:
        """Test validation passes when ground_truth present."""
        loader = LegalBenchDataset()

        # Create a mock dataset with ground_truth
        loader.dataset = Dataset.from_dict({
            "prompt": ["What is the law?", "Is this valid?"],
            "response": ["The law is...", "Yes, it is valid."],
            "ground_truth": ["The law states...", "Valid under § 1234."],
            "task": ["rule_qa", "contract_qa"],
        })

        # Should pass without raising
        result = loader.validate_ground_truth_presence(require_xml_tags=False)
        assert result is True

    def test_validate_ground_truth_missing_field(self) -> None:
        """Test validation fails when ground_truth field missing."""
        loader = LegalBenchDataset()

        # Create a mock dataset WITHOUT ground_truth
        loader.dataset = Dataset.from_dict({
            "prompt": ["What is the law?"],
            "response": ["The law is..."],
            "task": ["rule_qa"],
        })

        # Should raise GroundTruthValidationError
        with pytest.raises(GroundTruthValidationError) as exc_info:
            loader.validate_ground_truth_presence()

        assert "PR #7 NOT MERGED" in str(exc_info.value)
        assert "ground_truth metadata missing" in str(exc_info.value)

    def test_validate_ground_truth_empty_dataset(self) -> None:
        """Test validation passes with empty dataset."""
        loader = LegalBenchDataset()

        # Create an empty dataset
        loader.dataset = Dataset.from_dict({
            "prompt": [],
            "response": [],
            "ground_truth": [],
        })

        # Should pass (empty dataset doesn't fail)
        result = loader.validate_ground_truth_presence()
        assert result is True

    def test_validate_ground_truth_no_dataset(self) -> None:
        """Test validation fails when dataset not loaded."""
        loader = LegalBenchDataset()

        # Don't load any dataset
        with pytest.raises(GroundTruthValidationError) as exc_info:
            loader.validate_ground_truth_presence()

        assert "Dataset not loaded" in str(exc_info.value)

    def test_has_xml_tags_reasoning(self) -> None:
        """Test XML tag detection for reasoning tags."""
        loader = LegalBenchDataset()

        text_with_tags = "<reasoning>Step 1: Analyze the issue.</reasoning>"
        assert loader._has_xml_tags(text_with_tags) is True

        text_without_tags = "Step 1: Analyze the issue."
        assert loader._has_xml_tags(text_without_tags) is False

    def test_has_xml_tags_answer(self) -> None:
        """Test XML tag detection for answer tags."""
        loader = LegalBenchDataset()

        text_with_tags = "<answer>The defendant is liable.</answer>"
        assert loader._has_xml_tags(text_with_tags) is True

    def test_has_xml_tags_empty(self) -> None:
        """Test XML tag detection for empty text."""
        loader = LegalBenchDataset()

        assert loader._has_xml_tags("") is False
        assert loader._has_xml_tags(None) is False


class TestTrainingConfigPhase2:
    """Tests for Phase 2 training configuration."""

    def test_config_default_max_steps(self) -> None:
        """Test default max_steps is None."""
        from judicaita.training import TrainingConfig

        config = TrainingConfig()

        assert config.max_steps is None
        assert config.validation_mode is False
        assert config.memory_log_steps == 10
        assert config.memory_threshold_gb == 12.0
        assert config.time_limit_hours == 8.5

    def test_config_validation_mode(self) -> None:
        """Test validation mode configuration."""
        from judicaita.training import TrainingConfig

        config = TrainingConfig(
            max_steps=50,
            validation_mode=True,
            memory_threshold_gb=10.0,
            time_limit_hours=8.0,
        )

        assert config.max_steps == 50
        assert config.validation_mode is True
        assert config.memory_threshold_gb == 10.0
        assert config.time_limit_hours == 8.0
