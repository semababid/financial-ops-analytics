"""Shared matplotlib/seaborn styling and a figure-saving helper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = sns.color_palette("crest", as_cmap=False)


def save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to reports/figures/<name>.png and close it."""
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def brl(x, _pos=None) -> str:
    """Format a number as Brazilian Reais for axis ticks."""
    if abs(x) >= 1_000_000:
        return f"R${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"R${x / 1_000:.0f}K"
    return f"R${x:.0f}"
