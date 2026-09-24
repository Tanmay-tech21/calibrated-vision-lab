"""Controlled logit-space stressors for shift-sensitivity experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ShiftKind = Literal["confidence_softening", "gaussian_noise", "class_bias"]


def apply_logit_shift(
    logits: ArrayLike,
    *,
    kind: ShiftKind,
    severity: float,
    seed: int = 0,
    target_class: int = 0,
) -> NDArray[np.float64]:
    """Apply a deterministic, parameterised stressor to classifier logits.

    The transformations are diagnostic proxies, not substitutes for evaluating
    genuine image corruptions. Severity zero is an identity operation for every
    stressor, which makes clean and shifted measurements directly comparable.
    """
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes), with n_classes >= 2")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    if not np.isfinite(severity) or severity < 0.0:
        raise ValueError("severity must be finite and non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(target_class, int) or isinstance(target_class, bool):
        raise ValueError("target_class must be an integer")
    if not 0 <= target_class < scores.shape[1]:
        raise ValueError("target_class is outside the class range")

    shifted = scores.copy()
    if kind == "confidence_softening":
        return shifted / (1.0 + severity)
    if kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        return shifted + severity * rng.normal(size=shifted.shape)
    if kind == "class_bias":
        shifted[:, target_class] += severity
        return shifted
    raise ValueError(
        "kind must be 'confidence_softening', 'gaussian_noise', or 'class_bias'"
    )
