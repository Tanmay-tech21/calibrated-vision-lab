"""Plotting utilities for confidence-calibration diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import ArrayLike

from .metrics import calibration_bins

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_reliability_diagram(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = 15,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot observed accuracy and mean confidence for populated bins."""
    import matplotlib.pyplot as plt

    statistics = calibration_bins(logits, labels, n_bins=n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 5.0))

    widths = statistics.edges[1:] - statistics.edges[:-1]
    centres = statistics.edges[:-1] + widths / 2.0
    populated = statistics.counts > 0

    ax.bar(
        centres[populated],
        statistics.accuracy[populated],
        width=widths[populated] * 0.9,
        alpha=0.55,
        color="#3B82F6",
        label="Observed accuracy",
    )
    ax.scatter(
        statistics.mean_confidence[populated],
        statistics.accuracy[populated],
        color="#B91C1C",
        s=28,
        zorder=3,
        label="Bin mean",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="#374151", label="Perfect calibration")
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Confidence", ylabel="Accuracy")
    ax.set_title(title or f"Reliability diagram (ECE={statistics.expected_calibration_error:.3f})")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    return ax
