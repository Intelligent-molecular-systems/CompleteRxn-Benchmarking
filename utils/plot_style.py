"""Shared colors and formatting for the paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_neurips_style(
    font_size: float = 8.5,
    tick_size: float = 8.5,
    line_width: float = 0.8,
    save_dpi: int = 300,
    figure_dpi: int = 300,
) -> None:
    """Apply compact NeurIPS-style matplotlib defaults.

    Call this after seaborn theme setup (if seaborn is used), because seaborn can
    override matplotlib rcParams.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': tick_size,
        'axes.linewidth': line_width,
        'lines.linewidth': line_width,
        'savefig.dpi': save_dpi,
        'figure.dpi': figure_dpi,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
