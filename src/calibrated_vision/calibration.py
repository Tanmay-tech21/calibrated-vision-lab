"""Post-hoc calibration methods for classifier logits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import negative_log_likelihood


def _as_logits(logits: ArrayLike) -> NDArray[np.float64]:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes), with n_classes >= 2")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    return scores


@dataclass
class TemperatureScaler:
    """Fit a single positive temperature by minimising held-out NLL.

    Optimisation takes place in log-temperature space. This guarantees a
    positive temperature while allowing the bounded search to cover both
    sharpening (T < 1) and softening (T > 1) regimes.
    """

    bounds: tuple[float, float] = (0.05, 10.0)
    max_iterations: int = 96
    temperature: float = 1.0

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not (0.0 < lower < upper):
            raise ValueError("bounds must satisfy 0 < lower < upper")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def fit(self, logits: ArrayLike, labels: ArrayLike) -> "TemperatureScaler":
        """Fit on a calibration split and return this scaler."""
        scores = _as_logits(logits)
        targets = np.asarray(labels)

        lower, upper = np.log(self.bounds)
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        left = upper - (upper - lower) / golden_ratio
        right = lower + (upper - lower) / golden_ratio

        def objective(log_temperature: float) -> float:
            return negative_log_likelihood(scores / np.exp(log_temperature), targets)

        left_value = objective(left)
        right_value = objective(right)
        for _ in range(self.max_iterations):
            if left_value <= right_value:
                upper, right, right_value = right, left, left_value
                left = upper - (upper - lower) / golden_ratio
                left_value = objective(left)
            else:
                lower, left, left_value = left, right, right_value
                right = lower + (upper - lower) / golden_ratio
                right_value = objective(right)

        self.temperature = float(np.exp((lower + upper) / 2.0))
        return self

    def transform(self, logits: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted temperature while preserving class ordering."""
        return _as_logits(logits) / self.temperature

    def fit_transform(self, logits: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
        """Fit and transform the calibration logits."""
        return self.fit(logits, labels).transform(logits)
