"""Framework-independent metrics for multiclass confidence calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


def negative_log_likelihood(logits: ArrayLike, labels: ArrayLike) -> float:
    """Return mean multiclass negative log-likelihood from unnormalised logits."""
    scores, targets = _validate_logits_and_labels(logits, labels)
    probabilities = softmax(scores)
    true_class_probability = probabilities[np.arange(targets.size), targets]
    return float(-np.log(np.clip(true_class_probability, 1e-15, 1.0)).mean())


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
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    scores, targets = _validate_logits_and_labels(logits, labels)
    probabilities = softmax(scores)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets

    # digitize against interior edges gives stable handling at both endpoints.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=True)

    ece = 0.0
    for bin_id in range(n_bins):
        members = bin_ids == bin_id
        if np.any(members):
            weight = float(members.mean())
            accuracy = float(correct[members].mean())
            mean_confidence = float(confidence[members].mean())
            ece += weight * abs(accuracy - mean_confidence)
    return ece
