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


def make_imbalanced_synthetic_logits(
    *, n_samples: int = 5_000, seed: int = 23
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a five-class fixture with heterogeneous support and difficulty.

    This deliberately simplified sample exists to exercise imbalance-aware
    diagnostics. It is not evidence about a trained classifier.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")

    rng = np.random.default_rng(seed)
    class_prior = np.array([0.70, 0.18, 0.08, 0.03, 0.01])
    labels = rng.choice(class_prior.size, size=n_samples, p=class_prior)
    logits = rng.normal(scale=0.9, size=(n_samples, class_prior.size))

    # Scarcer classes receive weaker signal to mimic a long-tail failure mode.
    signal = np.array([3.4, 2.8, 2.2, 1.6, 1.0])
    logits[np.arange(n_samples), labels] += signal[labels]
    logits += np.log(class_prior + 0.03)
    return logits, labels.astype(np.int64)
