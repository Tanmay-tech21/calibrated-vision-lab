"""Compare prevalence-weighted and class-balanced temperature scaling."""

from __future__ import annotations

import json

import numpy as np

from calibrated_vision import (
    TemperatureScaler,
    expected_calibration_error,
    groupwise_calibration,
    negative_log_likelihood,
)
from calibrated_vision.synthetic import make_imbalanced_synthetic_logits


def _stratified_split(labels: np.ndarray, seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calibration: list[np.ndarray] = []
    evaluation: list[np.ndarray] = []
    for class_id in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        split = max(1, int(0.4 * indices.size))
        calibration.append(indices[:split])
        evaluation.append(indices[split:])
    return np.concatenate(calibration), np.concatenate(evaluation)


def _class_balanced_weights(labels: np.ndarray) -> np.ndarray:
    support = np.bincount(labels)
    return 1.0 / support[labels]


def _summarise(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    report = groupwise_calibration(logits, labels, n_bins=15)
    return {
        "nll": negative_log_likelihood(logits, labels),
        "aggregate_ece": expected_calibration_error(logits, labels, n_bins=15),
        "macro_group_ece": report.macro_ece,
        "support_weighted_group_ece": report.support_weighted_ece,
        "per_class": [
            {
                "class_id": int(class_id),
                "support": int(report.support[class_id]),
                "accuracy": float(report.accuracy[class_id]),
                "mean_confidence": float(report.mean_confidence[class_id]),
                "ece": float(report.expected_calibration_error[class_id]),
            }
            for class_id in np.flatnonzero(report.support)
        ],
    }


def main() -> None:
    logits, labels = make_imbalanced_synthetic_logits()
    calibration_indices, evaluation_indices = _stratified_split(labels)
    calibration_logits = logits[calibration_indices]
    calibration_labels = labels[calibration_indices]
    evaluation_logits = logits[evaluation_indices]
    evaluation_labels = labels[evaluation_indices]

    standard = TemperatureScaler().fit(calibration_logits, calibration_labels)
    balanced = TemperatureScaler().fit(
        calibration_logits,
        calibration_labels,
        sample_weight=_class_balanced_weights(calibration_labels),
    )

    result = {
        "fixture": "deterministic imbalanced synthetic pipeline validation",
        "class_support": np.bincount(evaluation_labels, minlength=5).tolist(),
        "temperatures": {
            "prevalence_weighted": standard.temperature,
            "class_balanced": balanced.temperature,
        },
        "raw": _summarise(evaluation_logits, evaluation_labels),
        "prevalence_weighted": _summarise(standard.transform(evaluation_logits), evaluation_labels),
        "class_balanced": _summarise(balanced.transform(evaluation_logits), evaluation_labels),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
