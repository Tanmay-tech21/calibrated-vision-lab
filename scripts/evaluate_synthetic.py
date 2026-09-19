"""Run the Day 1 calibration metrics on deterministic synthetic logits."""

from __future__ import annotations

import json

from calibrated_vision.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)
from calibrated_vision.synthetic import make_synthetic_logits


def main() -> None:
    logits, labels = make_synthetic_logits()
    probabilities = softmax(logits)
    predictions = probabilities.argmax(axis=1)

    report = {
        "accuracy": float((predictions == labels).mean()),
        "brier_score": brier_score(logits, labels),
        "ece_15_bins": expected_calibration_error(logits, labels, n_bins=15),
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
        "n_classes": int(logits.shape[1]),
        "n_samples": int(logits.shape[0]),
        "source": "deterministic synthetic fixture; not a model benchmark",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
