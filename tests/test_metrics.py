import numpy as np
import pytest

from calibrated_vision.metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    softmax,
)
from calibrated_vision.synthetic import make_synthetic_logits


def test_softmax_is_stable_and_normalised() -> None:
    logits = np.array([[1_000.0, 1_001.0], [-1_000.0, -999.0]])
    probabilities = softmax(logits)

    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_perfect_high_confidence_predictions_have_small_losses() -> None:
    logits = np.array([[12.0, -12.0], [-12.0, 12.0]])
    labels = np.array([0, 1])

    assert negative_log_likelihood(logits, labels) < 1e-8
    assert brier_score(logits, labels) < 1e-15
    assert expected_calibration_error(logits, labels, n_bins=5) < 1e-8


def test_ece_matches_a_hand_computed_single_bin_case() -> None:
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4]])
    logits = np.log(probabilities)
    labels = np.array([0, 1])

    # Accuracy is 0.5 and mean confidence is 0.7.
    assert expected_calibration_error(logits, labels, n_bins=1) == pytest.approx(0.2)


def test_synthetic_fixture_is_reproducible() -> None:
    first_logits, first_labels = make_synthetic_logits(seed=19)
    second_logits, second_labels = make_synthetic_logits(seed=19)

    np.testing.assert_array_equal(first_logits, second_logits)
    np.testing.assert_array_equal(first_labels, second_labels)


@pytest.mark.parametrize(
    ("logits", "labels"),
    [
        (np.array([1.0, 2.0]), np.array([0])),
        (np.array([[1.0, 2.0]]), np.array([2])),
        (np.array([[1.0, np.nan]]), np.array([0])),
        (np.array([[1.0, 2.0]]), np.array([0.0])),
    ],
)
def test_invalid_inputs_are_rejected(logits: np.ndarray, labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        negative_log_likelihood(logits, labels)
