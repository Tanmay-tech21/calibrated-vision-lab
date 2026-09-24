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
from .selective import RiskCoverageCurve, confidence_scores, risk_coverage_curve
from .shifts import ShiftKind, apply_logit_shift
from .visualization import plot_reliability_diagram, plot_risk_coverage_curve

__all__ = [
    "TemperatureScaler",
    "CalibrationBins",
    "GroupwiseCalibration",
    "RiskCoverageCurve",
    "ShiftKind",
    "apply_logit_shift",
    "brier_score",
    "calibration_bins",
    "confidence_scores",
    "expected_calibration_error",
    "groupwise_calibration",
    "negative_log_likelihood",
    "plot_reliability_diagram",
    "plot_risk_coverage_curve",
    "risk_coverage_curve",
    "softmax",
]
