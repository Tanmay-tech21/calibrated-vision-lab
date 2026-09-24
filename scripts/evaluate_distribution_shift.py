"""Stress-test calibration and abstention under controlled logit shifts."""

from __future__ import annotations

import json

import numpy as np

from calibrated_vision import (
    TemperatureScaler,
    apply_logit_shift,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    risk_coverage_curve,
)
from calibrated_vision.synthetic import make_synthetic_logits


STRESSORS = ("confidence_softening", "gaussian_noise", "class_bias")
SEVERITIES = (0.0, 0.5, 1.0, 1.5)


def _metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    curve = risk_coverage_curve(logits, labels)
    return {
        "accuracy": float((logits.argmax(axis=1) == labels).mean()),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(logits, labels),
        "ece": expected_calibration_error(logits, labels),
        "aurc": curve.aurc,
        "risk_at_50_percent_coverage": curve.risk_at_coverage(0.50),
        "risk_at_90_percent_coverage": curve.risk_at_coverage(0.90),
    }


def main() -> None:
    logits, labels = make_synthetic_logits(n_samples=3_000, seed=61)
    calibration_logits, evaluation_logits = logits[:1_000], logits[1_000:]
    calibration_labels, evaluation_labels = labels[:1_000], labels[1_000:]
    scaler = TemperatureScaler().fit(calibration_logits, calibration_labels)

    results: dict[str, list[dict[str, object]]] = {}
    for stressor in STRESSORS:
        measurements = []
        for severity in SEVERITIES:
            shifted = apply_logit_shift(
                evaluation_logits,
                kind=stressor,
                severity=severity,
                seed=113,
                target_class=0,
            )
            measurements.append(
                {
                    "severity": severity,
                    "raw": _metrics(shifted, evaluation_labels),
                    "clean_fitted_temperature": _metrics(
                        scaler.transform(shifted), evaluation_labels
                    ),
                }
            )
        results[stressor] = measurements

    output = {
        "fixture": "deterministic synthetic logit-shift pipeline validation",
        "limitations": (
            "Logit stressors isolate failure modes but do not represent a measured "
            "image-corruption benchmark."
        ),
        "calibration_samples": int(calibration_labels.size),
        "evaluation_samples": int(evaluation_labels.size),
        "temperature_fitted_on_clean_data": scaler.temperature,
        "stressors": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
