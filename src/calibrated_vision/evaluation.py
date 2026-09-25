"""Consistent prediction-quality summaries for calibration experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from .metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)
from .selective import risk_coverage_curve


@dataclass(frozen=True)
class PredictionMetrics:
    """Metrics spanning classification, calibration, and selective prediction."""

    accuracy: float
    nll: float
    brier: float
    ece: float
    aurc: float
    risk_at_50_percent_coverage: float
    risk_at_90_percent_coverage: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def evaluate_predictions(logits: ArrayLike, labels: ArrayLike) -> PredictionMetrics:
    """Evaluate one logits array using the project's declared conventions."""
    probabilities = softmax(logits)
    targets = np.asarray(labels)
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("labels must have shape (n_samples,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must be integer class indices")
    targets = targets.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("labels contain an out-of-range class index")

    curve = risk_coverage_curve(logits, targets)
    return PredictionMetrics(
        accuracy=float((probabilities.argmax(axis=1) == targets).mean()),
        nll=negative_log_likelihood(logits, targets),
        brier=brier_score(logits, targets),
        ece=expected_calibration_error(logits, targets),
        aurc=curve.aurc,
        risk_at_50_percent_coverage=curve.risk_at_coverage(0.50),
        risk_at_90_percent_coverage=curve.risk_at_coverage(0.90),
    )
