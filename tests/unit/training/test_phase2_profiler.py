"""
Tests for memory profiling and validation infrastructure.
"""

import pytest

from judicaita.training.profiler import (
    GradientMonitor,
    MemoryProfiler,
    MemorySnapshot,
    MemoryStats,
    TrainingTimeEstimator,
)
from judicaita.training.validation import (
    ValidationChecker,
    ValidationReport,
)


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_memory_profiler_initialization(self) -> None:
        """Test memory profiler initializes correctly."""
        profiler = MemoryProfiler(threshold_gb=12.0, log_interval=10)

        assert profiler.threshold_gb == 12.0
        assert profiler.log_interval == 10

    def test_memory_profiler_get_stats(self) -> None:
        """Test getting memory stats (works even without GPU)."""
        profiler = MemoryProfiler()

        stats = profiler.get_gpu_memory_stats()

        assert "allocated_gb" in stats
        assert "reserved_gb" in stats
        assert "max_allocated_gb" in stats

    def test_memory_profiler_log_usage(self) -> None:
        """Test logging memory usage."""
        profiler = MemoryProfiler(log_interval=5)

        snapshot = profiler.log_memory_usage(step=10)

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.step == 10
        assert len(profiler.stats.snapshots) == 1

    def test_memory_profiler_report(self) -> None:
        """Test memory report generation."""
        profiler = MemoryProfiler(threshold_gb=12.0)

        # Log a few snapshots
        profiler.log_memory_usage(step=1)
        profiler.log_memory_usage(step=2)

        report = profiler.get_memory_report()

        assert "peak_allocated_gb" in report
        assert "avg_allocated_gb" in report
        assert "within_threshold" in report
        assert "num_snapshots" in report
        assert report["num_snapshots"] == 2


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_memory_stats_update(self) -> None:
        """Test updating memory stats."""
        stats = MemoryStats()

        snapshot1 = MemorySnapshot(
            step=1,
            timestamp=0.0,
            allocated_gb=5.0,
            reserved_gb=6.0,
            max_allocated_gb=5.0,
            device="cuda:0",
        )
        stats.update(snapshot1)

        assert stats.peak_allocated_gb == 5.0
        assert stats.min_allocated_gb == 5.0
        assert stats.avg_allocated_gb == 5.0

        snapshot2 = MemorySnapshot(
            step=2,
            timestamp=1.0,
            allocated_gb=8.0,
            reserved_gb=10.0,
            max_allocated_gb=8.0,
            device="cuda:0",
        )
        stats.update(snapshot2)

        assert stats.peak_allocated_gb == 8.0
        assert stats.min_allocated_gb == 5.0
        assert stats.avg_allocated_gb == 6.5


class TestGradientMonitor:
    """Tests for GradientMonitor class."""

    def test_gradient_monitor_initialization(self) -> None:
        """Test gradient monitor initializes correctly."""
        monitor = GradientMonitor(log_interval=10)

        assert monitor.log_interval == 10
        assert len(monitor.gradient_norms) == 0
        assert monitor.nan_count == 0
        assert monitor.inf_count == 0

    def test_gradient_monitor_stability_report(self) -> None:
        """Test gradient stability report generation."""
        monitor = GradientMonitor()

        # Simulate some gradient norms
        monitor.gradient_norms = [1.0, 1.5, 1.2]

        report = monitor.get_stability_report()

        assert "avg_gradient_norm" in report
        assert "max_gradient_norm" in report
        assert "nan_count" in report
        assert "inf_count" in report
        assert "is_stable" in report
        assert report["is_stable"] is True
        assert report["max_gradient_norm"] == 1.5

    def test_gradient_monitor_reset(self) -> None:
        """Test gradient monitor reset."""
        monitor = GradientMonitor()
        monitor.gradient_norms = [1.0, 2.0]
        monitor.nan_count = 1

        monitor.reset()

        assert len(monitor.gradient_norms) == 0
        assert monitor.nan_count == 0


class TestTrainingTimeEstimator:
    """Tests for TrainingTimeEstimator class."""

    def test_time_estimator_initialization(self) -> None:
        """Test time estimator initializes correctly."""
        estimator = TrainingTimeEstimator(time_limit_hours=8.5)

        assert estimator.time_limit_hours == 8.5
        assert len(estimator.step_times) == 0

    def test_time_estimator_step_timing(self) -> None:
        """Test step timing."""
        import time

        estimator = TrainingTimeEstimator()

        estimator.start_step()
        time.sleep(0.01)  # 10ms
        duration = estimator.end_step()

        assert duration > 0
        assert len(estimator.step_times) == 1

    def test_time_estimator_average(self) -> None:
        """Test average step time calculation."""
        estimator = TrainingTimeEstimator()

        # Simulate step times
        estimator.step_times = [1.0, 2.0, 3.0]

        avg = estimator.get_average_step_time()

        assert avg == 2.0

    def test_time_estimator_estimate(self) -> None:
        """Test total time estimation."""
        estimator = TrainingTimeEstimator(time_limit_hours=8.5)

        # Simulate 10 steps at 1 second each
        estimator.step_times = [1.0] * 10

        estimate = estimator.estimate_total_time(total_steps=100)

        assert estimate["avg_step_time_seconds"] == 1.0
        assert estimate["completed_steps"] == 10
        assert estimate["remaining_steps"] == 90
        assert estimate["estimated_total_hours"] == pytest.approx(100 / 3600, abs=0.01)
        assert estimate["exceeds_limit"] is False

    def test_time_estimator_exceeds_limit(self) -> None:
        """Test detection of exceeding time limit."""
        estimator = TrainingTimeEstimator(time_limit_hours=1.0)

        # Simulate 10 steps at 1 minute each
        estimator.step_times = [60.0] * 10

        estimate = estimator.estimate_total_time(total_steps=100)

        # 100 steps * 60 seconds = 100 minutes = 1.67 hours > 1 hour limit
        assert estimate["exceeds_limit"] is True


class TestValidationChecker:
    """Tests for ValidationChecker class."""

    def test_validation_checker_initialization(self) -> None:
        """Test validation checker initializes correctly."""
        checker = ValidationChecker(
            target_steps=50,
            memory_threshold_gb=12.0,
            time_limit_hours=8.5,
        )

        assert checker.target_steps == 50
        assert checker.memory_threshold_gb == 12.0
        assert checker.time_limit_hours == 8.5

    def test_check_completion_success(self) -> None:
        """Test completion check passes."""
        checker = ValidationChecker(target_steps=50)

        result = checker.check_completion(completed_steps=50)

        assert result.passed is True
        assert "50" in result.message

    def test_check_completion_failure(self) -> None:
        """Test completion check fails when steps incomplete."""
        checker = ValidationChecker(target_steps=50)

        result = checker.check_completion(completed_steps=30)

        assert result.passed is False
        assert "30" in result.message

    def test_check_completion_error(self) -> None:
        """Test completion check fails on error."""
        checker = ValidationChecker(target_steps=50)

        result = checker.check_completion(
            completed_steps=25,
            had_error=True,
            error_message="OOM error",
        )

        assert result.passed is False
        assert "OOM error" in result.message

    def test_check_reward_components_success(self) -> None:
        """Test reward components check passes."""
        checker = ValidationChecker()

        reward_details = {
            "correctness": {"score": 0.8, "weight": 0.4},
            "reasoning_quality": {"score": 0.7, "weight": 0.3},
            "citation_accuracy": {"score": 0.5, "weight": 0.2},
            "clarity": {"score": 0.6, "weight": 0.1},
        }

        result = checker.check_reward_components(reward_details)

        assert result.passed is True

    def test_check_reward_components_missing(self) -> None:
        """Test reward components check fails when components missing."""
        checker = ValidationChecker()

        reward_details = {
            "correctness": {"score": 0.8, "weight": 0.4},
            # Missing other components
        }

        result = checker.check_reward_components(reward_details)

        assert result.passed is False

    def test_check_memory_limit_success(self) -> None:
        """Test memory limit check passes."""
        checker = ValidationChecker(memory_threshold_gb=12.0)

        memory_report = {
            "peak_allocated_gb": 8.5,
            "avg_allocated_gb": 6.0,
        }

        result = checker.check_memory_limit(memory_report)

        assert result.passed is True

    def test_check_memory_limit_failure(self) -> None:
        """Test memory limit check fails when exceeded."""
        checker = ValidationChecker(memory_threshold_gb=12.0)

        memory_report = {
            "peak_allocated_gb": 14.5,
            "avg_allocated_gb": 12.0,
        }

        result = checker.check_memory_limit(memory_report)

        assert result.passed is False

    def test_check_time_estimate_success(self) -> None:
        """Test time estimate check passes."""
        checker = ValidationChecker(time_limit_hours=8.5)

        time_estimate = {
            "estimated_total_hours": 6.0,
        }

        result = checker.check_time_estimate(time_estimate)

        assert result.passed is True

    def test_check_time_estimate_failure(self) -> None:
        """Test time estimate check fails when exceeded."""
        checker = ValidationChecker(time_limit_hours=8.5)

        time_estimate = {
            "estimated_total_hours": 10.0,
        }

        result = checker.check_time_estimate(time_estimate)

        assert result.passed is False

    def test_check_gradient_stability_success(self) -> None:
        """Test gradient stability check passes."""
        checker = ValidationChecker()

        gradient_report = {
            "nan_count": 0,
            "inf_count": 0,
            "is_stable": True,
            "avg_gradient_norm": 1.5,
        }

        result = checker.check_gradient_stability(gradient_report)

        assert result.passed is True

    def test_check_gradient_stability_failure(self) -> None:
        """Test gradient stability check fails on NaN."""
        checker = ValidationChecker()

        gradient_report = {
            "nan_count": 5,
            "inf_count": 2,
            "is_stable": False,
        }

        result = checker.check_gradient_stability(gradient_report)

        assert result.passed is False

    def test_generate_full_report(self) -> None:
        """Test full validation report generation."""
        checker = ValidationChecker(
            target_steps=50,
            memory_threshold_gb=12.0,
            time_limit_hours=8.5,
        )

        report = checker.generate_report(
            completed_steps=50,
            had_error=False,
            error_message=None,
            reward_details={
                "correctness": {"score": 0.8, "weight": 0.4},
                "reasoning_quality": {"score": 0.7, "weight": 0.3},
                "citation_accuracy": {"score": 0.5, "weight": 0.2},
                "clarity": {"score": 0.6, "weight": 0.1},
            },
            memory_report={"peak_allocated_gb": 8.0},
            time_estimate={"estimated_total_hours": 5.0},
            gradient_report={"nan_count": 0, "inf_count": 0, "is_stable": True},
        )

        assert isinstance(report, ValidationReport)
        assert report.completion.passed is True
        assert report.reward_components.passed is True
        assert report.memory_stable.passed is True
        assert report.time_feasible.passed is True
        assert report.gradients_stable.passed is True
        assert report.ready_for_phase_3 is True

    def test_report_to_dict(self) -> None:
        """Test validation report to_dict method."""
        checker = ValidationChecker(target_steps=50)

        report = checker.generate_report(
            completed_steps=50,
            had_error=False,
            error_message=None,
            reward_details={
                "correctness": {"score": 0.8, "weight": 0.4},
                "reasoning_quality": {"score": 0.7, "weight": 0.3},
                "citation_accuracy": {"score": 0.5, "weight": 0.2},
                "clarity": {"score": 0.6, "weight": 0.1},
            },
            memory_report={"peak_allocated_gb": 8.0},
            time_estimate={"estimated_total_hours": 5.0},
            gradient_report={"nan_count": 0, "inf_count": 0, "is_stable": True},
        )

        report_dict = report.to_dict()

        assert "completion" in report_dict
        assert "reward_components" in report_dict
        assert "memory_stable" in report_dict
        assert "time_feasible" in report_dict
        assert "gradients_stable" in report_dict
        assert "ready_for_phase_3" in report_dict
