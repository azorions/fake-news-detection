"""Reliability diagrams for the evaluation report (needs the ``classic`` extra: matplotlib).

Drawn with matplotlib's object interface, so nothing here selects a global backend or opens a
window. Colors are the first two slots of the report palette (blue, orange), checked for
color-blind separation and contrast on the light surface; the two curves also differ in marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from fnd_ml.calibration.metrics import expected_calibration_error

N_BINS = 15
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
CALIBRATED = ("calibrated", "#2a78d6", "o")
UNCALIBRATED = ("uncalibrated", "#eb6834", "s")


def _curve(
    y: NDArray[np.float64], p: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean predicted probability and observed rate of each of the 15 bins that hold rows."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    index = np.digitize(p, edges[1:-1])
    counts = np.bincount(index, minlength=N_BINS)
    filled = counts > 0
    mean_p = np.bincount(index, weights=p, minlength=N_BINS)[filled] / counts[filled]
    rate = np.bincount(index, weights=y, minlength=N_BINS)[filled] / counts[filled]
    return mean_p, rate


def reliability_diagram(
    y: ArrayLike, p_before: ArrayLike, p_after: ArrayLike, path: Path, title: str
) -> None:
    """Write a 15-bin reliability diagram: ``p_before`` uncalibrated, ``p_after`` as served.

    A point is one bin: its mean predicted probability against the share of its rows that are
    unreliable. A calibrated model lies on the diagonal. Each curve's ECE is in the legend.
    """
    labels = np.asarray(y, dtype=np.float64)
    figure = Figure(figsize=(5.4, 5.6), dpi=120, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    axes.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linewidth=1,
        linestyle=(0, (4, 4)),
        label="perfect calibration",
        zorder=2,
    )
    series = ((UNCALIBRATED, p_before, 3), (CALIBRATED, p_after, 4))
    for (name, color, marker), scores, layer in series:
        p = np.asarray(scores, dtype=np.float64)
        mean_p, rate = _curve(labels, p)
        ece = expected_calibration_error(labels, p, N_BINS)
        axes.plot(
            mean_p,
            rate,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=f"{name} (ECE {ece:.3f})",
            clip_on=False,  # a marker on the frame (p near 0 or 1) is drawn whole
            zorder=layer,
        )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.set_xlabel("Predicted probability of unreliable", color=MUTED)
    axes.set_ylabel("Observed share unreliable", color=MUTED)
    axes.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, length=0)
    handles, names = axes.get_legend_handles_labels()
    legend = axes.legend(handles[::-1], names[::-1], loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
