import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from calibrated_vision import plot_risk_coverage_curve, risk_coverage_curve


def test_risk_coverage_curve_matches_hand_computed_ranking() -> None:
    probabilities = np.array(
        [[0.95, 0.05], [0.85, 0.15], [0.70, 0.30], [0.55, 0.45]]
    )
    labels = np.array([0, 1, 0, 1])

    curve = risk_coverage_curve(np.log(probabilities), labels)

    np.testing.assert_allclose(curve.coverage, [0.25, 0.50, 0.75, 1.00])
    np.testing.assert_allclose(curve.risk, [0.0, 0.5, 1.0 / 3.0, 0.5])
    assert curve.risk_at_coverage(0.70) == pytest.approx(1.0 / 3.0)
    assert curve.aurc == pytest.approx((0.0 + 0.5 + 1.0 / 3.0 + 0.5) / 4.0)


def test_perfect_predictions_have_zero_selective_risk() -> None:
    logits = np.array([[5.0, 0.0], [0.0, 4.0], [3.0, 0.0]])
    curve = risk_coverage_curve(logits, np.array([0, 1, 0]), score="negative_entropy")

    np.testing.assert_array_equal(curve.risk, np.zeros(3))
    assert curve.aurc == 0.0


def test_confidence_ties_preserve_input_order() -> None:
    logits = np.zeros((3, 2))
    curve = risk_coverage_curve(logits, np.array([0, 1, 0]))

    np.testing.assert_array_equal(curve.accepted_indices, [0, 1, 2])


@pytest.mark.parametrize("target", [0.0, -0.1, 1.1, np.nan])
def test_invalid_target_coverage_is_rejected(target: float) -> None:
    curve = risk_coverage_curve(np.array([[1.0, 0.0]]), np.array([0]))
    with pytest.raises(ValueError, match="target_coverage"):
        curve.risk_at_coverage(target)


def test_risk_coverage_plot_uses_requested_axes() -> None:
    curve = risk_coverage_curve(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([0, 1]))
    figure, ax = plt.subplots()

    returned = plot_risk_coverage_curve(curve, ax=ax, label="fixture")

    assert returned is ax
    assert ax.get_xlabel() == "Coverage"
    assert ax.get_ylabel() == "Selective risk"
    plt.close(figure)
