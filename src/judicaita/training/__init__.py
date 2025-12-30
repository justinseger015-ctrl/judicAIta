"""
Training module for Judicaita GRPO pipeline.

This module provides training infrastructure for legal domain adaptation using
GRPO (Group Relative Policy Optimization) with LegalBench and Pile-of-Law datasets.
"""

from judicaita.training.data_curation import (
    GroundTruthValidationError,
    LegalBenchDataset,
    PileOfLawDataset,
    SyntheticCoTGenerator,
    create_training_dataset,
)
from judicaita.training.grpo_trainer import GRPOTrainer, TrainingConfig
from judicaita.training.profiler import (
    GradientMonitor,
    MemoryProfiler,
    MemorySnapshot,
    MemoryStats,
    TrainingTimeEstimator,
)
from judicaita.training.rewards import (
    CitationAccuracyReward,
    ClarityReward,
    CompositeReward,
    FormatReward,
    OutcomeReward,
    VerbosityReward,
)
from judicaita.training.validation import (
    ValidationChecker,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    # Data curation
    "GroundTruthValidationError",
    "LegalBenchDataset",
    "PileOfLawDataset",
    "SyntheticCoTGenerator",
    "create_training_dataset",
    # Trainer
    "GRPOTrainer",
    "TrainingConfig",
    # Reward functions
    "CitationAccuracyReward",
    "ClarityReward",
    "CompositeReward",
    "FormatReward",
    "OutcomeReward",
    "VerbosityReward",
    # Profiling
    "GradientMonitor",
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryStats",
    "TrainingTimeEstimator",
    # Validation
    "ValidationChecker",
    "ValidationReport",
    "ValidationResult",
]
