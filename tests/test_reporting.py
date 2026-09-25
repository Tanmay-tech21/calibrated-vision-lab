import numpy as np
import pytest

from calibrated_vision import (
    build_experiment_snapshot,
    evaluate_predictions,
    render_experiment_card,
)


def test_prediction_summary_matches_perfect_fixture() -> None:
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [3.0, -1.0], [-1.0, 3.0]])
    labels = np.array([0, 1, 0, 1])
    metrics = evaluate_predictions(logits, labels)

    assert metrics.accuracy == 1.0
    assert metrics.aurc == 0.0
    assert metrics.risk_at_50_percent_coverage == 0.0
    assert set(metrics.as_dict()) == {
        "accuracy",
        "nll",
        "brier",
        "ece",
        "aurc",
        "risk_at_50_percent_coverage",
        "risk_at_90_percent_coverage",
    }


def test_prediction_summary_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        evaluate_predictions([[1.0, 0.0], [0.0, 1.0]], [0])


def test_experiment_snapshot_is_deterministic_and_held_out() -> None:
    first = build_experiment_snapshot()
    second = build_experiment_snapshot()

    assert first == second
    assert first["protocol"]["calibration_samples"] == 1_000
    assert first["protocol"]["evaluation_samples"] == 2_000
    assert first["clean"]["raw"]["accuracy"] == first["clean"]["calibrated"]["accuracy"]
    assert set(first["shifted"]) == {
        "confidence_softening",
        "gaussian_noise",
        "class_bias",
    }


def test_experiment_card_preserves_claim_boundary() -> None:
    card = render_experiment_card(build_experiment_snapshot())

    assert card.startswith("# Experiment Card")
    assert "not claim performance" in card
    assert "for a trained vision model" in card
    assert "No deployment threshold is validated on real coin images." in card
    assert "Calibration split: `1000` samples" in card
