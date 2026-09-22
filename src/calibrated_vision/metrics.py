"""Framework-independent metrics for multiclass confidence calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CalibrationBins:
    """Sufficient statistics for an equal-width reliability diagram."""

    edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    accuracy: NDArray[np.float64]
    mean_confidence: NDArray[np.float64]

    @property
    def expected_calibration_error(self) -> float:
        """Return sample-weighted absolute accuracy-confidence gaps."""
        total = int(self.counts.sum())
        gaps = np.nan_to_num(np.abs(self.accuracy - self.mean_confidence))
        return float(np.dot(self.counts / total, gaps))


@dataclass(frozen=True)
class GroupwiseCalibration:
    """Top-label calibration diagnostics partitioned by true class."""

    support: NDArray[np.int64]
    accuracy: NDArray[np.float64]
    mean_confidence: NDArray[np.float64]
    expected_calibration_error: NDArray[np.float64]

    @property
    def macro_ece(self) -> float:
        """Average ECE across represented classes, irrespective of support."""
        return float(np.nanmean(self.expected_calibration_error))

    @property
    def support_weighted_ece(self) -> float:
        """Average ECE weighted by empirical class prevalence."""
        weights = self.support / self.support.sum()
        return float(np.dot(weights, np.nan_to_num(self.expected_calibration_error)))


def _validate_logits_and_labels(
    logits: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels)

    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes), with n_classes >= 2")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("labels must have shape (n_samples,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must be integer class indices")

    targets = targets.astype(np.int64, copy=False)
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("labels contain an out-of-range class index")
    return scores, targets


def softmax(logits: ArrayLike) -> NDArray[np.float64]:
    """Convert a two-dimensional logits array into class probabilities."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes), with n_classes >= 2")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: ArrayLike, labels: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Return mean multiclass negative log-likelihood from unnormalised logits."""
    scores, targets = _validate_logits_and_labels(logits, labels)
    probabilities = softmax(scores)
    true_class_probability = probabilities[np.arange(targets.size), targets]
    losses = -np.log(np.clip(true_class_probability, 1e-15, 1.0))
    if sample_weight is None:
        return float(losses.mean())

    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (targets.size,):
        raise ValueError("sample_weight must have shape (n_samples,)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
    return float(np.average(losses, weights=weights))


def brier_score(logits: ArrayLike, labels: ArrayLike) -> float:
    """Return the mean multiclass Brier score from unnormalised logits."""
    scores, targets = _validate_logits_and_labels(logits, labels)
    probabilities = softmax(scores)
    one_hot_targets = np.eye(scores.shape[1], dtype=np.float64)[targets]
    return float(np.square(probabilities - one_hot_targets).sum(axis=1).mean())


def expected_calibration_error(
    logits: ArrayLike, labels: ArrayLike, *, n_bins: int = 15
) -> float:
    """Compute equal-width, top-label Expected Calibration Error.

    Confidence exactly equal to zero belongs to the first bin; confidence equal
    to one belongs to the final bin. Empty bins contribute zero.
    """
    return calibration_bins(logits, labels, n_bins=n_bins).expected_calibration_error


def calibration_bins(
    logits: ArrayLike, labels: ArrayLike, *, n_bins: int = 15
) -> CalibrationBins:
    """Aggregate top-label accuracy and confidence in equal-width bins.

    Empty bins retain ``NaN`` accuracy and confidence so that plots do not
    imply measurements where no observations exist.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    scores, targets = _validate_logits_and_labels(logits, labels)
    probabilities = softmax(scores)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, edges[1:-1], right=True)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.int64)
    accuracy = np.full(n_bins, np.nan, dtype=np.float64)
    mean_confidence = np.full(n_bins, np.nan, dtype=np.float64)

    for bin_id in np.flatnonzero(counts):
        members = bin_ids == bin_id
        accuracy[bin_id] = float(correct[members].mean())
        mean_confidence[bin_id] = float(confidence[members].mean())

    return CalibrationBins(
        edges=edges,
        counts=counts,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
    )


def groupwise_calibration(
    logits: ArrayLike, labels: ArrayLike, *, n_bins: int = 15
) -> GroupwiseCalibration:
    """Report top-label calibration separately for each represented true class.

    Conditioning on the true class is a diagnostic for disparities, not a
    replacement for population-level calibration. Unrepresented classes are
    returned as ``NaN`` and excluded from the macro average.
    """
    scores, targets = _validate_logits_and_labels(logits, labels)
    n_classes = scores.shape[1]
    support = np.bincount(targets, minlength=n_classes).astype(np.int64)
    accuracy = np.full(n_classes, np.nan, dtype=np.float64)
    mean_confidence = np.full(n_classes, np.nan, dtype=np.float64)
    ece = np.full(n_classes, np.nan, dtype=np.float64)

    probabilities = softmax(scores)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    for class_id in np.flatnonzero(support):
        members = targets == class_id
        accuracy[class_id] = float((predictions[members] == targets[members]).mean())
        mean_confidence[class_id] = float(confidence[members].mean())
        ece[class_id] = calibration_bins(
            scores[members], targets[members], n_bins=n_bins
        ).expected_calibration_error

    return GroupwiseCalibration(
        support=support,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        expected_calibration_error=ece,
    )
