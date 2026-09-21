"""Measure ECE sensitivity and save raw/calibrated reliability diagrams."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrated_vision import (
    TemperatureScaler,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    plot_reliability_diagram,
)
from calibrated_vision.synthetic import make_synthetic_logits


SEEDS = tuple(range(10))
BIN_COUNTS = (10, 15, 30, 50)


def _mean_and_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def run_sensitivity_study() -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    records: dict[str, dict[str, list[float]]] = {
        "raw": {"nll": [], "brier": [], **{f"ece_{n}": [] for n in BIN_COUNTS}},
        "calibrated": {"nll": [], "brier": [], **{f"ece_{n}": [] for n in BIN_COUNTS}},
    }
    representative: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for seed in SEEDS:
        logits, labels = make_synthetic_logits(seed=seed, n_samples=1_000)
        order = np.random.default_rng(seed + 10_000).permutation(labels.size)
        split = int(0.4 * labels.size)
        calibration_indices, evaluation_indices = order[:split], order[split:]
        scaler = TemperatureScaler().fit(logits[calibration_indices], labels[calibration_indices])
        evaluation_logits = logits[evaluation_indices]
        evaluation_labels = labels[evaluation_indices]
        calibrated_logits = scaler.transform(evaluation_logits)

        for name, candidate_logits in (
            ("raw", evaluation_logits),
            ("calibrated", calibrated_logits),
        ):
            records[name]["nll"].append(negative_log_likelihood(candidate_logits, evaluation_labels))
            records[name]["brier"].append(brier_score(candidate_logits, evaluation_labels))
            for n_bins in BIN_COUNTS:
                records[name][f"ece_{n_bins}"].append(
                    expected_calibration_error(candidate_logits, evaluation_labels, n_bins=n_bins)
                )

        if seed == SEEDS[0]:
            representative = evaluation_logits, calibrated_logits, evaluation_labels

    summary = {
        "fixture": "deterministic synthetic pipeline validation",
        "seeds": list(SEEDS),
        "evaluation_samples_per_seed": 600,
        "metrics": {
            method: {metric: _mean_and_std(values) for metric, values in method_records.items()}
            for method, method_records in records.items()
        },
    }
    assert representative is not None
    return summary, representative


def save_diagram(
    raw_logits: np.ndarray,
    calibrated_logits: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    plot_reliability_diagram(raw_logits, labels, n_bins=15, ax=axes[0], title="Raw logits")
    plot_reliability_diagram(
        calibrated_logits, labels, n_bins=15, ax=axes[1], title="Temperature-scaled logits"
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    summary, (raw_logits, calibrated_logits, labels) = run_sensitivity_study()
    output_path = Path("artifacts/reliability_diagram.png")
    save_diagram(raw_logits, calibrated_logits, labels, output_path)
    summary["representative_diagram"] = str(output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
