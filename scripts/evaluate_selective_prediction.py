"""Evaluate confidence-based abstention on a deterministic held-out fixture."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibrated_vision import (
    TemperatureScaler,
    plot_risk_coverage_curve,
    risk_coverage_curve,
)
from calibrated_vision.synthetic import make_synthetic_logits


COVERAGE_LEVELS = (0.50, 0.70, 0.90, 1.00)


def _summarise(curve) -> dict[str, object]:
    return {
        "aurc": curve.aurc,
        "risk_at_coverage": {
            f"{coverage:.0%}": curve.risk_at_coverage(coverage)
            for coverage in COVERAGE_LEVELS
        },
    }


def main() -> None:
    logits, labels = make_synthetic_logits(n_samples=2_000, seed=47)
    calibration_logits, evaluation_logits = logits[:800], logits[800:]
    calibration_labels, evaluation_labels = labels[:800], labels[800:]
    scaler = TemperatureScaler().fit(calibration_logits, calibration_labels)
    calibrated_logits = scaler.transform(evaluation_logits)

    curves = {
        "raw_max_probability": risk_coverage_curve(evaluation_logits, evaluation_labels),
        "calibrated_max_probability": risk_coverage_curve(
            calibrated_logits, evaluation_labels
        ),
        "raw_negative_entropy": risk_coverage_curve(
            evaluation_logits, evaluation_labels, score="negative_entropy"
        ),
        "calibrated_negative_entropy": risk_coverage_curve(
            calibrated_logits, evaluation_labels, score="negative_entropy"
        ),
    }

    output_path = Path("artifacts/risk_coverage.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for name, curve in curves.items():
        plot_risk_coverage_curve(curve, ax=ax, label=name.replace("_", " "))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    result = {
        "fixture": "deterministic synthetic pipeline validation",
        "evaluation_samples": int(evaluation_labels.size),
        "temperature": scaler.temperature,
        "methods": {name: _summarise(curve) for name, curve in curves.items()},
        "diagram": str(output_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
