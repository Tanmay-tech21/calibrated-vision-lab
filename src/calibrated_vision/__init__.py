"""Confidence-calibration utilities for vision experiments."""

from .calibration import TemperatureScaler
from .metrics import (
    CalibrationBins,
    GroupwiseCalibration,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    groupwise_calibration,
    negative_log_likelihood,
    softmax,
)
from .visualization import plot_reliability_diagram

__all__ = [
    "TemperatureScaler",
    "CalibrationBins",
    "GroupwiseCalibration",
    "brier_score",
    "calibration_bins",
    "expected_calibration_error",
    "groupwise_calibration",
    "negative_log_likelihood",
    "plot_reliability_diagram",
    "softmax",
]
