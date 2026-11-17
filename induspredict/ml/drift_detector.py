"""
IndusPredict AI - Concept Drift and Distribution Shift Monitoring Engine.
Implements ADWIN (Adaptive Windowing) and Page-Hinkley sequential change detection
for continuous streaming sensor telemetry and prediction residuals.
"""

from collections import deque
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from induspredict.core.logger import get_logger

logger = get_logger("ml.drift")


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) algorithm for detecting distribution drift in data streams.
    Dynamically adjusts window size: expands during stationary regimes and shrinks when
    a statistically significant difference between sub-windows is detected.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window: deque = deque(maxlen=max_window)
        self.total: float = 0.0
        self.drift_detected: bool = False

    def add_element(self, value: float) -> bool:
        """
        Add a new real-valued observation to the window.
        Returns True if a concept drift was detected, False otherwise.
        """
        self.drift_detected = False
        self.window.append(value)
        self.total += value

        n = len(self.window)
        if n < 30:
            return False

        # Evaluate cut points across sub-windows W0 and W1
        # Check candidate splits
        arr = np.array(self.window)
        step = max(5, n // 10)

        for split in range(step, n - step, step):
            n0 = split
            n1 = n - split
            w0 = arr[:split]
            w1 = arr[split:]

            u0 = np.mean(w0)
            u1 = np.mean(w1)
            diff = abs(u0 - u1)

            # Harmonic mean
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            # ADWIN threshold
            epsilon_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta))

            if diff >= epsilon_cut:
                # Drift detected! Drop older sub-window W0
                for _ in range(n0):
                    removed = self.window.popleft()
                    self.total -= removed

                self.drift_detected = True
                logger.info(f"ADWIN Drift detected! Δμ = {diff:.4f} >= ε_cut = {epsilon_cut:.4f}. Dropped {n0} samples.")
                return True

        return False

    @property
    def current_mean(self) -> float:
        return (self.total / len(self.window)) if self.window else 0.0

    @property
    def window_size(self) -> int:
        return len(self.window)


class PageHinkleyDetector:
    """
    Page-Hinkley sequential change detection test.
    Detects abrupt changes in the average of a continuous sensor signal.
    """

    def __init__(self, delta: float = 0.05, threshold: float = 30.0, alpha: float = 0.99):
        self.delta = delta              # Minimum magnitude of change to detect
        self.threshold = threshold      # Detection threshold lambda
        self.alpha = alpha              # Forgetting factor for running mean
        self.mean: float = 0.0
        self.sum_stat: float = 0.0      # Cumulative sum U_t
        self.min_stat: float = 0.0      # Running minimum m_t
        self.n: int = 0
        self.drift_detected: bool = False

    def add_element(self, value: float) -> bool:
        """Add observation and check if Page-Hinkley statistic exceeds threshold."""
        self.drift_detected = False
        self.n += 1

        if self.n == 1:
            self.mean = value
            self.sum_stat = 0.0
            self.min_stat = 0.0
            return False

        # Update running mean with exponential weighting
        self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value

        # Update cumulative sum U_t = sum(x_i - mean - delta)
        self.sum_stat += (value - self.mean - self.delta)
        if self.sum_stat < self.min_stat:
            self.min_stat = self.sum_stat

        # Page-Hinkley statistic: PH_t = U_t - m_t
        ph_stat = self.sum_stat - self.min_stat

        if ph_stat > self.threshold:
            self.drift_detected = True
            logger.info(f"Page-Hinkley drift detected! PH_t = {ph_stat:.2f} > threshold {self.threshold:.2f}")
            # Reset after trigger
            self.sum_stat = 0.0
            self.min_stat = 0.0
            return True

        return False


def compute_kolmogorov_smirnov(baseline: np.ndarray, current_window: np.ndarray, alpha: float = 0.01) -> Tuple[float, float, bool]:
    """
    Perform two-sample Kolmogorov-Smirnov test to detect non-parametric distribution shift.
    Returns (statistic, p_value, is_drift).
    """
    if len(baseline) < 10 or len(current_window) < 10:
        return 0.0, 1.0, False

    stat, p_val = stats.ks_2samp(baseline, current_window)
    is_drift = bool(p_val < alpha)
    return float(stat), float(p_val), is_drift
