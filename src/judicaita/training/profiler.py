"""
Memory profiling infrastructure for GRPO training.

This module provides GPU memory tracking and profiling utilities
for monitoring training resource usage and ensuring stability.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger


@dataclass
class MemorySnapshot:
    """A snapshot of GPU memory usage at a specific training step."""

    step: int
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    device: str


@dataclass
class MemoryStats:
    """Aggregated memory statistics over training."""

    peak_allocated_gb: float = 0.0
    avg_allocated_gb: float = 0.0
    min_allocated_gb: float = float("inf")
    snapshots: list[MemorySnapshot] = field(default_factory=list)

    def update(self, snapshot: MemorySnapshot) -> None:
        """Update statistics with a new snapshot."""
        self.snapshots.append(snapshot)
        self.peak_allocated_gb = max(self.peak_allocated_gb, snapshot.allocated_gb)
        self.min_allocated_gb = min(self.min_allocated_gb, snapshot.allocated_gb)

        # Update running average
        total = sum(s.allocated_gb for s in self.snapshots)
        self.avg_allocated_gb = total / len(self.snapshots)


class MemoryProfiler:
    """
    GPU memory profiler for tracking training resource usage.

    Tracks CUDA memory allocation, provides threshold warnings,
    and generates memory usage reports.
    """

    def __init__(
        self,
        threshold_gb: float = 12.0,
        log_interval: int = 10,
        device: str | None = None,
    ) -> None:
        """
        Initialize memory profiler.

        Args:
            threshold_gb: Memory threshold in GB for warnings (default 12 GB for TPU/GPU)
            log_interval: Steps between memory logs
            device: CUDA device to track (None for default)
        """
        self.threshold_gb = threshold_gb
        self.log_interval = log_interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = MemoryStats()
        self._start_time = time.time()

    def get_gpu_memory_stats(self) -> dict[str, float]:
        """
        Get current GPU memory statistics.

        Returns:
            Dictionary with memory usage in GB:
            - allocated_gb: Currently allocated memory
            - reserved_gb: Total reserved memory
            - max_allocated_gb: Peak allocated memory since start
        """
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "available": False,
            }

        try:
            device_idx = self._parse_device_index()

            allocated_bytes = torch.cuda.memory_allocated(device_idx)
            reserved_bytes = torch.cuda.memory_reserved(device_idx)
            max_allocated_bytes = torch.cuda.max_memory_allocated(device_idx)

            return {
                "allocated_gb": allocated_bytes / (1024**3),
                "reserved_gb": reserved_bytes / (1024**3),
                "max_allocated_gb": max_allocated_bytes / (1024**3),
                "available": True,
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory stats: {e}")
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "available": False,
            }

    def _parse_device_index(self) -> int:
        """
        Parse the CUDA device index from the device string.

        Returns:
            Integer device index (0 if unable to parse)
        """
        if self.device == "cuda":
            return torch.cuda.current_device()

        if ":" in self.device:
            try:
                return int(self.device.split(":")[-1])
            except ValueError:
                return 0

        return 0

    def log_memory_usage(self, step: int) -> MemorySnapshot:
        """
        Log memory usage at a specific training step.

        Args:
            step: Current training step

        Returns:
            MemorySnapshot with current usage
        """
        stats = self.get_gpu_memory_stats()

        snapshot = MemorySnapshot(
            step=step,
            timestamp=time.time() - self._start_time,
            allocated_gb=stats["allocated_gb"],
            reserved_gb=stats["reserved_gb"],
            max_allocated_gb=stats["max_allocated_gb"],
            device=self.device,
        )

        self.stats.update(snapshot)

        if step % self.log_interval == 0:
            logger.info(
                f"Step {step}: GPU Memory = {snapshot.allocated_gb:.2f} GB "
                f"(peak: {self.stats.peak_allocated_gb:.2f} GB)"
            )

        return snapshot

    def check_memory_threshold(self, step: int | None = None) -> bool:
        """
        Check if memory usage exceeds threshold and log warning.

        Args:
            step: Optional step number for logging

        Returns:
            True if memory is within threshold, False if exceeded
        """
        stats = self.get_gpu_memory_stats()
        allocated_gb = stats["allocated_gb"]

        if allocated_gb > self.threshold_gb:
            step_info = f" at step {step}" if step is not None else ""
            logger.warning(
                f"⚠️ Memory threshold exceeded{step_info}: "
                f"{allocated_gb:.2f} GB > {self.threshold_gb:.2f} GB threshold"
            )
            return False

        return True

    def get_memory_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive memory usage report.

        Returns:
            Dictionary with memory statistics and analysis
        """
        # Use math.isinf for robust infinity checking
        min_alloc = (
            0.0
            if math.isinf(self.stats.min_allocated_gb)
            else round(self.stats.min_allocated_gb, 2)
        )

        return {
            "peak_allocated_gb": round(self.stats.peak_allocated_gb, 2),
            "avg_allocated_gb": round(self.stats.avg_allocated_gb, 2),
            "min_allocated_gb": min_alloc,
            "threshold_gb": self.threshold_gb,
            "within_threshold": self.stats.peak_allocated_gb <= self.threshold_gb,
            "num_snapshots": len(self.stats.snapshots),
            "device": self.device,
        }

    def reset(self) -> None:
        """Reset memory statistics and CUDA peak memory tracking."""
        self.stats = MemoryStats()
        self._start_time = time.time()

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                # Some CUDA backends or driver/toolkit combinations may not support
                # resetting peak memory stats. This failure is non-fatal and should
                # not prevent the training state from being reset.
                logger.debug(
                    "Failed to reset CUDA peak memory stats in MemoryProfiler.reset()",
                    exc_info=True,
                )


class GradientMonitor:
    """
    Monitor gradient statistics during training for stability detection.

    Tracks gradient norms, detects NaN/Inf values, and provides
    stability analysis.
    """

    def __init__(self, log_interval: int = 10) -> None:
        """
        Initialize gradient monitor.

        Args:
            log_interval: Steps between gradient logs
        """
        self.log_interval = log_interval
        self.gradient_norms: list[float] = []
        self.nan_count = 0
        self.inf_count = 0

    def check_gradients(
        self,
        model: torch.nn.Module,
        step: int,
    ) -> tuple[float, bool]:
        """
        Check gradients for stability and compute norm.

        Args:
            model: The model to check gradients for
            step: Current training step

        Returns:
            Tuple of (gradient_norm, is_stable)
        """
        total_norm = 0.0
        has_nan = False
        has_inf = False

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)

                if torch.isnan(param_norm):
                    has_nan = True
                    self.nan_count += 1
                elif torch.isinf(param_norm):
                    has_inf = True
                    self.inf_count += 1
                else:
                    total_norm += param_norm.item() ** 2

        # Set total_norm to NaN/Inf when gradients are unstable to make instability visible
        if has_nan:
            total_norm = float("nan")
        elif has_inf:
            total_norm = float("inf")
        else:
            total_norm = total_norm**0.5
        self.gradient_norms.append(total_norm)

        is_stable = not (has_nan or has_inf)

        if step % self.log_interval == 0:
            logger.debug(f"Step {step}: Gradient norm = {total_norm:.4f}")

        if not is_stable:
            logger.error(
                f"Gradient instability at step {step}: "
                f"NaN={has_nan}, Inf={has_inf}, norm={total_norm}"
            )

        return total_norm, is_stable

    def get_stability_report(self) -> dict[str, Any]:
        """
        Generate gradient stability report.

        Returns:
            Dictionary with gradient statistics
        """
        if not self.gradient_norms:
            return {
                "avg_gradient_norm": 0.0,
                "max_gradient_norm": 0.0,
                "nan_count": self.nan_count,
                "inf_count": self.inf_count,
                "is_stable": True,
                "num_steps": 0,
            }

        return {
            "avg_gradient_norm": sum(self.gradient_norms) / len(self.gradient_norms),
            "max_gradient_norm": max(self.gradient_norms),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "is_stable": self.nan_count == 0 and self.inf_count == 0,
            "num_steps": len(self.gradient_norms),
        }

    def reset(self) -> None:
        """Reset gradient statistics."""
        self.gradient_norms = []
        self.nan_count = 0
        self.inf_count = 0


class TrainingTimeEstimator:
    """
    Estimate total training time based on step timing.

    Tracks time per step and extrapolates to full training duration.
    """

    def __init__(self, time_limit_hours: float = 8.5) -> None:
        """
        Initialize training time estimator.

        Args:
            time_limit_hours: Maximum allowed training time in hours (Kaggle limit)
        """
        self.time_limit_hours = time_limit_hours
        self.step_times: list[float] = []
        self._last_step_time: float | None = None

    def start_step(self) -> None:
        """Mark the start of a training step."""
        self._last_step_time = time.time()

    def end_step(self) -> float:
        """
        Mark the end of a training step and record duration.

        Returns:
            Time taken for the step in seconds
        """
        if self._last_step_time is None:
            return 0.0

        step_duration = time.time() - self._last_step_time
        self.step_times.append(step_duration)
        self._last_step_time = None

        return step_duration

    def get_average_step_time(self) -> float:
        """Get average time per step in seconds."""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def estimate_total_time(
        self,
        total_steps: int,
        completed_steps: int | None = None,
    ) -> dict[str, float]:
        """
        Estimate total training time.

        Args:
            total_steps: Total number of training steps
            completed_steps: Number of steps already completed (default: len(step_times))

        Returns:
            Dictionary with time estimates
        """
        completed = completed_steps if completed_steps is not None else len(self.step_times)
        avg_step_time = self.get_average_step_time()

        remaining_steps = max(0, total_steps - completed)
        estimated_remaining_seconds = remaining_steps * avg_step_time
        estimated_total_seconds = total_steps * avg_step_time

        estimated_total_hours = estimated_total_seconds / 3600
        estimated_remaining_hours = estimated_remaining_seconds / 3600

        exceeds_limit = estimated_total_hours > self.time_limit_hours

        if exceeds_limit:
            logger.warning(
                f"⚠️ Estimated training time ({estimated_total_hours:.1f}h) "
                f"exceeds {self.time_limit_hours}h limit. "
                f"Consider reducing epochs or increasing batch size."
            )

        return {
            "avg_step_time_seconds": round(avg_step_time, 3),
            "completed_steps": completed,
            "remaining_steps": remaining_steps,
            "estimated_remaining_hours": round(estimated_remaining_hours, 2),
            "estimated_total_hours": round(estimated_total_hours, 2),
            "time_limit_hours": self.time_limit_hours,
            "exceeds_limit": exceeds_limit,
        }

    def log_progress(self, step: int, total_steps: int, log_interval: int = 10) -> None:
        """
        Log training progress with time estimates.

        Args:
            step: Current step
            total_steps: Total training steps
            log_interval: Steps between logs
        """
        if step % log_interval != 0:
            return

        estimates = self.estimate_total_time(total_steps, step)

        logger.info(
            f"⏱️ Step {step}/{total_steps}: "
            f"{estimates['avg_step_time_seconds']:.2f}s/step, "
            f"ETA: {estimates['estimated_remaining_hours']:.1f}h "
            f"(total: {estimates['estimated_total_hours']:.1f}h)"
        )

    def reset(self) -> None:
        """Reset timing statistics."""
        self.step_times = []
        self._last_step_time = None
