"""Compare raw and temperature-scaled logits on a held-out synthetic split."""

from __future__ import annotations

import json

import numpy as np

from calibrated_vision import TemperatureScaler
from calibrated_vision.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)
from calibrated_vision.synthetic import make_synthetic_logits


def metric_report(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = softmax(logits)
    return {
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "brier_score": brier_score(logits, labels),
        "ece_15_bins": expected_calibration_error(logits, labels, n_bins=15),
        "negative_log_likelihood": negative_log_likelihood(logits, labels),
    }


def main() -> None:
    logits, labels = make_synthetic_logits(n_samples=1_000, seed=17)
    permutation = np.random.default_rng(17).permutation(len(labels))
    split = int(0.4 * len(labels))
    calibration_ids, evaluation_ids = permutation[:split], permutation[split:]

    scaler = TemperatureScaler().fit(logits[calibration_ids], labels[calibration_ids])
    evaluation_logits = logits[evaluation_ids]
    calibrated_logits = scaler.transform(evaluation_logits)

    report = {
        "calibrated": metric_report(calibrated_logits, labels[evaluation_ids]),
        "calibration_samples": int(split),
        "evaluation_samples": int(len(evaluation_ids)),
        "raw": metric_report(evaluation_logits, labels[evaluation_ids]),
        "source": "deterministic synthetic fixture; not a model benchmark",
        "temperature": scaler.temperature,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
