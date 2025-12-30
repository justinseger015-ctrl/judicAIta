"""
Tests for training configuration.
"""



from judicaita.training.grpo_trainer import TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.base_model == "google/gemma-2-2b-it"
        assert config.use_lora is True
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TrainingConfig(
            base_model="google/gemma-2-9b-it",
            num_epochs=5,
            batch_size=8,
            learning_rate=5e-6,
            use_lora=False,
        )

        assert config.base_model == "google/gemma-2-9b-it"
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 5e-6
        assert config.use_lora is False

    def test_lora_config(self) -> None:
        """Test LoRA-specific configuration."""
        config = TrainingConfig(use_lora=True, lora_r=32, lora_alpha=64)

        assert config.use_lora is True
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05

    def test_grpo_config(self) -> None:
        """Test GRPO-specific configuration."""
        config = TrainingConfig(grpo_tau=0.2, grpo_gamma=0.95, num_rollouts=8)

        assert config.grpo_tau == 0.2
        assert config.grpo_gamma == 0.95
        assert config.num_rollouts == 8
