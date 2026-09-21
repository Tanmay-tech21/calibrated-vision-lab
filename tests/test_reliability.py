import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from calibrated_vision import calibration_bins, expected_calibration_error, plot_reliability_diagram


def test_calibration_bins_preserve_all_observations_and_ece() -> None:
    logits = np.array([[4.0, 0.0], [1.2, 0.0], [0.0, 1.2], [0.0, 4.0]])
    labels = np.array([0, 1, 1, 0])

    statistics = calibration_bins(logits, labels, n_bins=5)

    assert statistics.counts.sum() == labels.size
    assert statistics.expected_calibration_error == pytest.approx(
        expected_calibration_error(logits, labels, n_bins=5)
    )


def test_empty_bins_are_explicitly_missing() -> None:
    statistics = calibration_bins(np.array([[10.0, 0.0]]), np.array([0]), n_bins=10)

    empty = statistics.counts == 0
    assert np.isnan(statistics.accuracy[empty]).all()
    assert np.isnan(statistics.mean_confidence[empty]).all()
    assert np.isfinite(statistics.expected_calibration_error)


def test_reliability_diagram_uses_requested_axes() -> None:
    figure, ax = plt.subplots()
    returned = plot_reliability_diagram(
        np.array([[2.0, 0.0], [0.0, 2.0]]), np.array([0, 1]), n_bins=4, ax=ax
    )

    assert returned is ax
    assert ax.get_xlabel() == "Confidence"
    assert ax.get_ylabel() == "Accuracy"
    assert len(ax.patches) == 1
    plt.close(figure)
