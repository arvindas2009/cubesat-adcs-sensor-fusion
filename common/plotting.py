"""Small shared matplotlib helpers (CONVENTIONS section 7: PNG, 150 dpi, labelled axes with units).

No global state and no backend selection here; scripts that run headless call
``matplotlib.use("Agg")`` themselves before importing ``pyplot``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150


def save_fig(fig, path, dpi: int = DPI, close: bool = True) -> Path:
    """Save ``fig`` as PNG at ``dpi`` (default 150) to ``path`` (parent dirs are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path


def label_axes(ax, xlabel: str, ylabel: str, title: str | None = None, grid: bool = True):
    """Set axis labels (include units, e.g. ``"time [s]"``), optional title and grid."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(grid, alpha=0.3)
    return ax


def time_axes(ax, ylabel: str, title: str | None = None):
    """``label_axes`` with the x label fixed to ``"time since window start [s]"``."""
    return label_axes(ax, "time since window start [s]", ylabel, title)


def deg(x) -> np.ndarray:
    """Radians -> degrees for plotting only (internal quantities stay in radians)."""
    return np.degrees(np.asarray(x, dtype=float))


def subplots_stacked(n: int, sharex: bool = True, figsize=(9, None)):
    """``n`` vertically stacked axes; height defaults to 2.3 in per row."""
    import matplotlib.pyplot as plt

    w, h = figsize
    fig, axes = plt.subplots(n, 1, sharex=sharex, figsize=(w, h or 2.3 * n), squeeze=False)
    return fig, axes[:, 0]
