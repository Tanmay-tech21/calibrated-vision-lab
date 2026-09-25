"""Confidence-calibration utilities for vision experiments."""

from .calibration import TemperatureScaler
from .evaluation import PredictionMetrics, evaluate_predictions
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
from .reporting import build_experiment_snapshot, render_experiment_card
from .shifts import ShiftKind, apply_logit_shift
from .visualization import plot_reliability_diagram, plot_risk_coverage_curve

__all__ = [
    "TemperatureScaler",
    "PredictionMetrics",
    "CalibrationBins",
    "GroupwiseCalibration",
    "RiskCoverageCurve",
    "ShiftKind",
    "apply_logit_shift",
    "build_experiment_snapshot",
    "brier_score",
    "calibration_bins",
    "confidence_scores",
    "expected_calibration_error",
    "evaluate_predictions",
    "groupwise_calibration",
    "negative_log_likelihood",
    "plot_reliability_diagram",
    "plot_risk_coverage_curve",
    "risk_coverage_curve",
    "render_experiment_card",
    "softmax",
]
