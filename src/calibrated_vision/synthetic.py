"""Deterministic synthetic inputs for exercising the calibration pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_synthetic_logits(
    *, n_samples: int = 600, n_classes: int = 5, seed: int = 7
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate noisy labels and deliberately sharpened classifier logits.

    The returned sample is a software fixture, not a scientific benchmark.
    Sharpening the logits creates a useful input for later temperature-scaling
    exercises without predetermining that every calibration metric must worsen.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")

    rng = np.random.default_rng(seed)
    latent_scores = rng.normal(size=(n_samples, n_classes))
    clean_labels = latent_scores.argmax(axis=1)

    labels = clean_labels.copy()
    noisy = rng.random(n_samples) < 0.18
    labels[noisy] = rng.integers(0, n_classes, size=int(noisy.sum()))

    overconfident_logits = 2.4 * latent_scores + rng.normal(
        scale=0.35, size=latent_scores.shape
    )
    return overconfident_logits, labels.astype(np.int64)
