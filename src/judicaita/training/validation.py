"""
Validation checkers for GRPO training Phase 2.

This module provides validation utilities for verifying training
completion criteria and generating validation reports.
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class ValidationReport:
    """Complete validation report for Phase 2 training."""

    completion: ValidationResult
    reward_components: ValidationResult
    memory_stable: ValidationResult
    time_feasible: ValidationResult
    gradients_stable: ValidationResult
    ready_for_phase_3: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "completion": {
                "passed": self.completion.passed,
                "message": self.completion.message,
                "details": self.completion.details,
            },
            "reward_components": {
                "passed": self.reward_components.passed,
                "message": self.reward_components.message,
                "details": self.reward_components.details,
            },
            "memory_stable": {
                "passed": self.memory_stable.passed,
                "message": self.memory_stable.message,
                "details": self.memory_stable.details,
            },
            "time_feasible": {
                "passed": self.time_feasible.passed,
                "message": self.time_feasible.message,
                "details": self.time_feasible.details,
            },
            "gradients_stable": {
                "passed": self.gradients_stable.passed,
                "message": self.gradients_stable.message,
                "details": self.gradients_stable.details,
            },
            "ready_for_phase_3": self.ready_for_phase_3,
        }

    def print_report(self) -> None:
        """Print validation report to console."""
        print("\n" + "=" * 60)
        print("PHASE 2 VALIDATION REPORT")
        print("=" * 60)

        checks = [
            self.completion,
            self.reward_components,
            self.memory_stable,
            self.time_feasible,
            self.gradients_stable,
        ]

        for check in checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            print(f"\n{status} - {check.name}")
            print(f"    {check.message}")

        print("\n" + "-" * 60)
        if self.ready_for_phase_3:
            print("🎉 ALL CHECKS PASSED - Ready for Phase 3!")
        else:
            print("⚠️  VALIDATION FAILED - Address issues before proceeding")
        print("=" * 60 + "\n")


class ValidationChecker:
    """
    Validator for Phase 2 GRPO training completion criteria.

    Checks all success criteria from the ticket:
    - 50-step completion without OOM/errors
    - Reward function returns all 4 components
    - Memory stable under threshold (default 12 GB)
    - Estimated training time < limit (default 8.5 hours)
    - Gradient stability (no NaN/Inf)
    """

    def __init__(
        self,
        target_steps: int = 50,
        memory_threshold_gb: float = 12.0,
        time_limit_hours: float = 8.5,
    ) -> None:
        """
        Initialize validation checker.

        Args:
            target_steps: Target number of steps for validation run
            memory_threshold_gb: Maximum allowed peak memory in GB
            time_limit_hours: Maximum allowed total training time in hours
        """
        self.target_steps = target_steps
        self.memory_threshold_gb = memory_threshold_gb
        self.time_limit_hours = time_limit_hours

    def check_completion(
        self,
        completed_steps: int,
        had_error: bool = False,
        error_message: str | None = None,
    ) -> ValidationResult:
        """
        Verify training completed target steps without OOM/errors.

        Args:
            completed_steps: Number of steps actually completed
            had_error: Whether an error occurred during training
            error_message: Optional error message if error occurred

        Returns:
            ValidationResult with pass/fail status
        """
        if had_error:
            return ValidationResult(
                name="50-Step Completion",
                passed=False,
                message=f"Training failed with error: {error_message or 'Unknown error'}",
                details={"completed_steps": completed_steps, "error": error_message},
            )

        if completed_steps >= self.target_steps:
            return ValidationResult(
                name="50-Step Completion",
                passed=True,
                message=f"Successfully completed {completed_steps} training steps",
                details={"completed_steps": completed_steps, "target_steps": self.target_steps},
            )

        return ValidationResult(
            name="50-Step Completion",
            passed=False,
            message=f"Only completed {completed_steps}/{self.target_steps} steps",
            details={"completed_steps": completed_steps, "target_steps": self.target_steps},
        )

    def check_reward_components(
        self,
        reward_details: dict[str, Any] | None,
    ) -> ValidationResult:
        """
        Ensure all 4 reward components are returning values.

        Args:
            reward_details: Details dict from CompositeReward.compute()

        Returns:
            ValidationResult with pass/fail status
        """
        required_components = ["correctness", "reasoning_quality", "citation_accuracy", "clarity"]

        if reward_details is None:
            return ValidationResult(
                name="Reward Components",
                passed=False,
                message="No reward details available",
                details=None,
            )

        missing = []
        component_scores = {}

        for component in required_components:
            if component not in reward_details:
                missing.append(component)
            else:
                score = reward_details[component].get("score", 0.0)
                component_scores[component] = score

        if missing:
            return ValidationResult(
                name="Reward Components",
                passed=False,
                message=f"Missing reward components: {', '.join(missing)}",
                details={"missing": missing, "found": list(component_scores.keys())},
            )

        # Verify scores are valid numbers.
        # Note: All reward components must be normalized to [0, 1] range.
        # This is enforced in the reward implementations (e.g., CitationAccuracyReward,
        # ClarityReward) which clamp their outputs. If future reward components operate
        # on different scales, they must also normalize to [0, 1] before returning.
        invalid_scores = []
        for component, score in component_scores.items():
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                invalid_scores.append(f"{component}={score}")

        if invalid_scores:
            return ValidationResult(
                name="Reward Components",
                passed=False,
                message=f"Invalid reward scores: {', '.join(invalid_scores)}",
                details={"component_scores": component_scores},
            )

        return ValidationResult(
            name="Reward Components",
            passed=True,
            message="All 4 reward components returning valid values",
            details={"component_scores": component_scores},
        )

    def check_memory_limit(
        self,
        memory_report: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate peak memory usage stays under threshold.

        Args:
            memory_report: Report from MemoryProfiler.get_memory_report()

        Returns:
            ValidationResult with pass/fail status
        """
        peak_memory = memory_report.get("peak_allocated_gb", 0.0)

        if peak_memory <= self.memory_threshold_gb:
            return ValidationResult(
                name="Memory Stability",
                passed=True,
                message=f"Peak memory {peak_memory:.2f} GB within {self.memory_threshold_gb} GB limit",
                details=memory_report,
            )

        return ValidationResult(
            name="Memory Stability",
            passed=False,
            message=f"Peak memory {peak_memory:.2f} GB exceeds {self.memory_threshold_gb} GB limit",
            details=memory_report,
        )

    def check_time_estimate(
        self,
        time_estimate: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate estimated total training time is feasible.

        Args:
            time_estimate: Estimate from TrainingTimeEstimator.estimate_total_time()

        Returns:
            ValidationResult with pass/fail status
        """
        estimated_hours = time_estimate.get("estimated_total_hours", 0.0)

        if estimated_hours <= self.time_limit_hours:
            return ValidationResult(
                name="Time Estimate",
                passed=True,
                message=f"Estimated training time {estimated_hours:.1f}h within {self.time_limit_hours}h limit",
                details=time_estimate,
            )

        return ValidationResult(
            name="Time Estimate",
            passed=False,
            message=f"Estimated training time {estimated_hours:.1f}h exceeds {self.time_limit_hours}h limit",
            details=time_estimate,
        )

    def check_gradient_stability(
        self,
        gradient_report: dict[str, Any],
    ) -> ValidationResult:
        """
        Verify no NaN/Inf gradients occurred during training.

        Args:
            gradient_report: Report from GradientMonitor.get_stability_report()

        Returns:
            ValidationResult with pass/fail status
        """
        nan_count = gradient_report.get("nan_count", 0)
        inf_count = gradient_report.get("inf_count", 0)
        is_stable = gradient_report.get("is_stable", True)

        if is_stable and nan_count == 0 and inf_count == 0:
            avg_norm = gradient_report.get("avg_gradient_norm", 0.0)
            return ValidationResult(
                name="Gradient Stability",
                passed=True,
                message=f"Gradients stable, avg norm: {avg_norm:.4f}",
                details=gradient_report,
            )

        return ValidationResult(
            name="Gradient Stability",
            passed=False,
            message=f"Gradient instability detected: {nan_count} NaN, {inf_count} Inf",
            details=gradient_report,
        )

    def generate_report(
        self,
        completed_steps: int,
        had_error: bool,
        error_message: str | None,
        reward_details: dict[str, Any] | None,
        memory_report: dict[str, Any],
        time_estimate: dict[str, Any],
        gradient_report: dict[str, Any],
    ) -> ValidationReport:
        """
        Generate complete validation report.

        Args:
            completed_steps: Number of steps completed
            had_error: Whether an error occurred
            error_message: Error message if applicable
            reward_details: Reward function output details
            memory_report: Memory profiler report
            time_estimate: Time estimator report
            gradient_report: Gradient monitor report

        Returns:
            ValidationReport with all checks
        """
        completion = self.check_completion(completed_steps, had_error, error_message)
        reward_components = self.check_reward_components(reward_details)
        memory_stable = self.check_memory_limit(memory_report)
        time_feasible = self.check_time_estimate(time_estimate)
        gradients_stable = self.check_gradient_stability(gradient_report)

        ready_for_phase_3 = all(
            [
                completion.passed,
                reward_components.passed,
                memory_stable.passed,
                time_feasible.passed,
                gradients_stable.passed,
            ]
        )

        report = ValidationReport(
            completion=completion,
            reward_components=reward_components,
            memory_stable=memory_stable,
            time_feasible=time_feasible,
            gradients_stable=gradients_stable,
            ready_for_phase_3=ready_for_phase_3,
        )

        # Log summary
        if ready_for_phase_3:
            logger.info("✅ All Phase 2 validation checks passed!")
        else:
            all_checks = [
                completion,
                reward_components,
                memory_stable,
                time_feasible,
                gradients_stable,
            ]
            failed = [check.name for check in all_checks if not check.passed]
            logger.warning(f"⚠️ Phase 2 validation failed: {', '.join(failed)}")

        return report
