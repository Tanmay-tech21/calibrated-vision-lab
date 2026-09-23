"""Selective-prediction diagnostics for confidence-based abstention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import softmax

ConfidenceScore = Literal["max_probability", "negative_entropy"]


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Empirical error risk as progressively less-confident samples are kept."""

    coverage: NDArray[np.float64]
    risk: NDArray[np.float64]
    threshold: NDArray[np.float64]
    accepted_indices: NDArray[np.int64]

    @property
    def aurc(self) -> float:
        """Return the discrete area under the empirical risk-coverage curve."""
        return float(self.risk.mean())

    def risk_at_coverage(self, target_coverage: float) -> float:
        """Return risk after retaining at least the requested sample fraction."""
        if not np.isfinite(target_coverage) or not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1]")
        retained = int(np.ceil(target_coverage * self.coverage.size))
        return float(self.risk[retained - 1])


def confidence_scores(logits: ArrayLike, *, score: ConfidenceScore) -> NDArray[np.float64]:
    """Convert logits to a confidence score where larger means more certain."""
    probabilities = softmax(logits)
    if score == "max_probability":
        return probabilities.max(axis=1)
    if score == "negative_entropy":
        entropy = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=1
        )
        return 1.0 - entropy / np.log(probabilities.shape[1])
    raise ValueError("score must be 'max_probability' or 'negative_entropy'")


def risk_coverage_curve(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    score: ConfidenceScore = "max_probability",
) -> RiskCoverageCurve:
    """Rank predictions by confidence and compute cumulative selective risk.

    Confidence ties preserve input order. This produces a deterministic rank
    curve; a deployed scalar threshold may retain a different number of tied
    observations and should report its realised coverage.
    """
    probabilities = softmax(logits)
    targets = np.asarray(labels)
    if targets.ndim != 1 or targets.shape[0] != probabilities.shape[0]:
        raise ValueError("labels must have shape (n_samples,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must be integer class indices")
    targets = targets.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("labels contain an out-of-range class index")

    confidence = confidence_scores(logits, score=score)
    accepted_indices = np.argsort(-confidence, kind="stable")
    errors = probabilities.argmax(axis=1) != targets
    ranked_errors = errors[accepted_indices].astype(np.float64)
    retained = np.arange(1, targets.size + 1, dtype=np.float64)

    return RiskCoverageCurve(
        coverage=retained / targets.size,
        risk=np.cumsum(ranked_errors) / retained,
        threshold=confidence[accepted_indices],
        accepted_indices=accepted_indices.astype(np.int64),
    )
