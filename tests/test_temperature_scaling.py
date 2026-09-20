import numpy as np
import pytest

from calibrated_vision import TemperatureScaler
from calibrated_vision.metrics import negative_log_likelihood
from calibrated_vision.synthetic import make_synthetic_logits


def test_temperature_scaling_improves_held_out_nll_and_preserves_predictions() -> None:
    logits, labels = make_synthetic_logits(n_samples=1_200, seed=31)
    calibration_logits, evaluation_logits = logits[:500], logits[500:]
    calibration_labels, evaluation_labels = labels[:500], labels[500:]

    scaler = TemperatureScaler().fit(calibration_logits, calibration_labels)
    calibrated_logits = scaler.transform(evaluation_logits)

    assert scaler.temperature > 1.0
    assert negative_log_likelihood(
        calibrated_logits, evaluation_labels
    ) < negative_log_likelihood(evaluation_logits, evaluation_labels)
    np.testing.assert_array_equal(
        calibrated_logits.argmax(axis=1), evaluation_logits.argmax(axis=1)
    )


def test_transform_divides_logits_by_temperature() -> None:
    scaler = TemperatureScaler(temperature=2.0)
    logits = np.array([[4.0, 2.0, -2.0]])

    np.testing.assert_allclose(scaler.transform(logits), [[2.0, 1.0, -1.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (0.0, 1.0)},
        {"bounds": (2.0, 1.0)},
        {"max_iterations": 0},
        {"temperature": -1.0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TemperatureScaler(**kwargs)
