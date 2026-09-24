import numpy as np
import pytest

from calibrated_vision import TemperatureScaler, apply_logit_shift


@pytest.fixture
def logits() -> np.ndarray:
    return np.array([[3.0, 1.0, -1.0], [0.2, 1.5, -0.4], [-1.0, 0.0, 2.0]])


@pytest.mark.parametrize(
    "kind", ["confidence_softening", "gaussian_noise", "class_bias"]
)
def test_zero_severity_is_identity(logits: np.ndarray, kind: str) -> None:
    shifted = apply_logit_shift(logits, kind=kind, severity=0.0, seed=9)
    np.testing.assert_allclose(shifted, logits)
    assert not np.shares_memory(shifted, logits)


def test_confidence_softening_preserves_predictions(logits: np.ndarray) -> None:
    shifted = apply_logit_shift(logits, kind="confidence_softening", severity=2.0)
    np.testing.assert_array_equal(shifted.argmax(axis=1), logits.argmax(axis=1))
    np.testing.assert_allclose(shifted, logits / 3.0)


def test_gaussian_noise_is_seeded_and_severity_scaled(logits: np.ndarray) -> None:
    mild = apply_logit_shift(logits, kind="gaussian_noise", severity=0.25, seed=17)
    severe = apply_logit_shift(logits, kind="gaussian_noise", severity=1.0, seed=17)
    np.testing.assert_allclose(severe - logits, 4.0 * (mild - logits))


def test_class_bias_only_changes_target_column(logits: np.ndarray) -> None:
    shifted = apply_logit_shift(logits, kind="class_bias", severity=0.75, target_class=1)
    expected = logits.copy()
    expected[:, 1] += 0.75
    np.testing.assert_allclose(shifted, expected)


def test_clean_fitted_temperature_preserves_shifted_predictions(logits: np.ndarray) -> None:
    labels = np.array([0, 1, 2])
    scaler = TemperatureScaler().fit(logits, labels)
    shifted = apply_logit_shift(logits, kind="class_bias", severity=2.0)
    np.testing.assert_array_equal(
        scaler.transform(shifted).argmax(axis=1), shifted.argmax(axis=1)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "unknown", "severity": 1.0}, "kind"),
        ({"kind": "gaussian_noise", "severity": -0.1}, "severity"),
        ({"kind": "class_bias", "severity": 1.0, "target_class": 4}, "target_class"),
    ],
)
def test_invalid_shift_configuration_is_rejected(
    logits: np.ndarray, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_logit_shift(logits, **kwargs)


def test_non_finite_logits_are_rejected(logits: np.ndarray) -> None:
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        apply_logit_shift(logits, kind="gaussian_noise", severity=1.0)
