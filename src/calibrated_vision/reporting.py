"""Deterministic experiment-card assembly for the seven-day laboratory."""

from __future__ import annotations

from typing import Any

from .calibration import TemperatureScaler
from .evaluation import evaluate_predictions
from .shifts import apply_logit_shift
from .synthetic import make_synthetic_logits


def build_experiment_snapshot() -> dict[str, Any]:
    """Recompute the final headline results from a fixed, documented protocol."""
    fixture_seed = 61
    shift_seed = 113
    logits, labels = make_synthetic_logits(n_samples=3_000, seed=fixture_seed)
    calibration_logits, evaluation_logits = logits[:1_000], logits[1_000:]
    calibration_labels, evaluation_labels = labels[:1_000], labels[1_000:]
    scaler = TemperatureScaler().fit(calibration_logits, calibration_labels)

    clean_raw = evaluate_predictions(evaluation_logits, evaluation_labels)
    clean_calibrated = evaluate_predictions(
        scaler.transform(evaluation_logits), evaluation_labels
    )
    shifted: dict[str, dict[str, object]] = {}
    for kind in ("confidence_softening", "gaussian_noise", "class_bias"):
        shifted_logits = apply_logit_shift(
            evaluation_logits,
            kind=kind,
            severity=1.5,
            seed=shift_seed,
            target_class=0,
        )
        shifted[kind] = {
            "severity": 1.5,
            "raw": evaluate_predictions(shifted_logits, evaluation_labels).as_dict(),
            "clean_fitted_temperature": evaluate_predictions(
                scaler.transform(shifted_logits), evaluation_labels
            ).as_dict(),
        }

    return {
        "status": "synthetic pipeline validation",
        "protocol": {
            "fixture_seed": fixture_seed,
            "shift_seed": shift_seed,
            "calibration_samples": int(calibration_labels.size),
            "evaluation_samples": int(evaluation_labels.size),
            "ece_bins": 15,
            "temperature_fitted_on": "clean calibration split only",
        },
        "temperature": scaler.temperature,
        "clean": {
            "raw": clean_raw.as_dict(),
            "calibrated": clean_calibrated.as_dict(),
        },
        "shifted": shifted,
        "limitations": [
            "The logits and labels are synthetic software fixtures, not model benchmarks.",
            "Logit stressors isolate failure modes but do not reproduce image corruptions.",
            "ECE is bin-dependent and must be interpreted with proper scoring rules.",
            "No deployment threshold is validated on real coin images.",
        ],
    }


def render_experiment_card(snapshot: dict[str, Any]) -> str:
    """Render the machine-readable snapshot as a portfolio-facing Markdown card."""
    protocol = snapshot["protocol"]
    clean = snapshot["clean"]
    rows = []
    for name, result in snapshot["shifted"].items():
        raw = result["raw"]
        calibrated = result["clean_fitted_temperature"]
        rows.append(
            f"| {name.replace('_', ' ')} | raw | {raw['accuracy']:.4f} | "
            f"{raw['nll']:.4f} | {raw['ece']:.4f} | {raw['aurc']:.4f} |"
        )
        rows.append(
            f"| {name.replace('_', ' ')} | calibrated | "
            f"{calibrated['accuracy']:.4f} | {calibrated['nll']:.4f} | "
            f"{calibrated['ece']:.4f} | {calibrated['aurc']:.4f} |"
        )

    limitations = "\n".join(f"- {item}" for item in snapshot["limitations"])
    raw = clean["raw"]
    calibrated = clean["calibrated"]
    return f"""# Experiment Card: Calibrated Vision Lab

## Claim boundary

This report validates a deterministic calibration-analysis pipeline. It does
not claim performance for a trained vision model or a real deployment.

## Protocol

- Fixture seed: `{protocol['fixture_seed']}`
- Shift seed: `{protocol['shift_seed']}`
- Calibration split: `{protocol['calibration_samples']}` samples
- Evaluation split: `{protocol['evaluation_samples']}` untouched samples
- Temperature fitting: {protocol['temperature_fitted_on']}
- ECE convention: equal-width top-label ECE with `{protocol['ece_bins']}` bins

## Clean held-out results

| Method | Accuracy | NLL | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|
| Raw | {raw['accuracy']:.4f} | {raw['nll']:.4f} | {raw['brier']:.4f} | {raw['ece']:.4f} | {raw['aurc']:.4f} |
| Temperature scaled | {calibrated['accuracy']:.4f} | {calibrated['nll']:.4f} | {calibrated['brier']:.4f} | {calibrated['ece']:.4f} | {calibrated['aurc']:.4f} |

## Controlled shift results at severity 1.5

| Stressor | Method | Accuracy | NLL | ECE | AURC |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Severity is internal to each stressor; values are not comparable between
stressor families.

## Interpretation

Temperature scaling improves its clean held-out NLL objective without
guaranteeing improvement in ECE, Brier score, or selective ranking. Under
shift, changes in aggregate ECE can also disagree with changes in accuracy.
Accordingly, no single metric is treated as sufficient evidence of reliability.

## Limitations

{limitations}

## Intended next validation

Evaluate genuine held-out model logits under measured lighting, blur, glare,
compression, and background changes. Select abstention thresholds on calibration
data and report realised risk and coverage on a separately preserved test set.
"""
