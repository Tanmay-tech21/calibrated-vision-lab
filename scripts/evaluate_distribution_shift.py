"""Stress-test calibration and abstention under controlled logit shifts."""

from __future__ import annotations

import json

from calibrated_vision import (
    TemperatureScaler,
    apply_logit_shift,
    evaluate_predictions,
)
from calibrated_vision.synthetic import make_synthetic_logits


STRESSORS = ("confidence_softening", "gaussian_noise", "class_bias")
SEVERITIES = (0.0, 0.5, 1.0, 1.5)


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
                    "raw": evaluate_predictions(shifted, evaluation_labels).as_dict(),
                    "clean_fitted_temperature": evaluate_predictions(
                        scaler.transform(shifted), evaluation_labels
                    ).as_dict(),
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
