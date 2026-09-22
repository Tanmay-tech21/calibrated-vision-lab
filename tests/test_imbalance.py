import numpy as np
import pytest

from calibrated_vision import TemperatureScaler, groupwise_calibration, negative_log_likelihood
from calibrated_vision.synthetic import make_imbalanced_synthetic_logits


def test_groupwise_report_exposes_support_and_macro_weighting() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.9, 0.1],
        ]
    )
    labels = np.array([0, 0, 0, 0, 1])

    report = groupwise_calibration(np.log(probabilities), labels, n_bins=2)

    np.testing.assert_array_equal(report.support, [4, 1])
    assert report.macro_ece > report.support_weighted_ece
    assert report.expected_calibration_error[1] > report.expected_calibration_error[0]


def test_unrepresented_classes_are_excluded_from_macro_average() -> None:
    logits = np.array([[3.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
    report = groupwise_calibration(logits, np.array([0, 0]), n_bins=3)

    assert np.isnan(report.expected_calibration_error[1:]).all()
    assert report.macro_ece == pytest.approx(report.expected_calibration_error[0])


def test_weighted_temperature_fit_optimises_weighted_nll() -> None:
    logits, labels = make_imbalanced_synthetic_logits(n_samples=1_000)
    support = np.bincount(labels)
    weights = 1.0 / support[labels]
    scaler = TemperatureScaler().fit(logits, labels, sample_weight=weights)

    before = negative_log_likelihood(logits, labels, sample_weight=weights)
    after = negative_log_likelihood(scaler.transform(logits), labels, sample_weight=weights)

    assert after <= before


def test_invalid_sample_weights_are_rejected() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="sample_weight"):
        TemperatureScaler().fit(logits, labels, sample_weight=np.array([1.0, -1.0]))
