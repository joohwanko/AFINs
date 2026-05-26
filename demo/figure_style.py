"""Shared visual style for README/demo figures."""
from __future__ import annotations


def apply_figure_style():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = (
        "Inter",
        "Source Sans 3",
        "IBM Plex Sans",
        "Aptos",
        "Nimbus Sans",
        "DejaVu Sans",
    )
    font = next((name for name in preferred if name in available), "DejaVu Sans")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "font.weight": "regular",
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.titleweight": "regular",
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 18,
        "figure.titleweight": "regular",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.82",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return font
