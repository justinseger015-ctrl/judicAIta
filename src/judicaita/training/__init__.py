"""
Training module for Judicaita GRPO pipeline.

This module provides training infrastructure for legal domain adaptation using
GRPO (Group Relative Policy Optimization) with LegalBench and Pile-of-Law datasets.
"""

from judicaita.training.data_curation import (
    LegalBenchDataset,
    PileOfLawDataset,
    SyntheticCoTGenerator,
    create_training_dataset,
)
from judicaita.training.grpo_trainer import GRPOTrainer, TrainingConfig
from judicaita.training.rewards import (
    CompositeReward,
    FormatReward,
    OutcomeReward,
    VerbosityReward,
)

__all__ = [
    "LegalBenchDataset",
    "PileOfLawDataset",
    "SyntheticCoTGenerator",
    "create_training_dataset",
    "GRPOTrainer",
    "TrainingConfig",
    "CompositeReward",
    "FormatReward",
    "OutcomeReward",
    "VerbosityReward",
]
