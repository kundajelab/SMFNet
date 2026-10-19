"""Shared plot style: the SMFNet paper's palette and matplotlib settings.

The style matches the paper's figures: ggplot's ``theme_classic`` with a full
black panel border, no gridlines, small uniform text in a Helvetica-metric
font, and a grey and purple palette.

Plotting functions in this package draw inside :func:`plot_style`, which
applies the settings to their own figures only. Scripts that make every
figure in this style can call :func:`apply_style` once instead.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Liberation Sans is metric-compatible with Helvetica/Arial. Install it
# system-wide or point SMFNET_FONT_DIR at a directory of its .ttf files (e.g.
# extracted from conda-forge's r-fontliberation package); otherwise Arial, then
# DejaVu Sans, is used.
FONT_DIR = os.environ.get("SMFNET_FONT_DIR")

# Print size, for figures placed at their final size in a paper.
FONT_SIZE = 7
# Multi-panel diagnostic reports, read on screen, use slightly larger text.
REPORT_FONT_SIZE = 8

SMFNET_ALL_READS = "#636bb2"  # model scored with every observed read
SMFNET = "#9FA4D3"  # model scored at the matched read count
PSEUDO_REP = "#DEDEDE"  # observed pseudo-replicate
BASELINE = "#BCBCBC"  # positional baseline and shuffles
ACCENT = "#C65D65"
POINT = "#454545"
GREYS = ("#676767", "#BFBFBF", "#EFEFEF")
EXPECTED_FILL = "#EFF0F8"  # light tint of SMFNET_ALL_READS

ZERO_LINE = {"color": "black", "ls": "--", "lw": 0.5, "zorder": 0}


def import_pyplot() -> Any:
    """Import matplotlib's pyplot, which only the 'plotting' extra installs."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Plotting requires the 'plotting' extra: pip install 'SMFNet[plotting]'"
        ) from error
    return plt


def _register_fonts() -> None:
    if FONT_DIR:
        from matplotlib import font_manager

        for path in sorted(Path(FONT_DIR).glob("*.ttf")):
            font_manager.fontManager.addfont(str(path))


def style_settings(font_size: float = FONT_SIZE) -> dict[str, Any]:
    """matplotlib rcParams for the style, with every text at ``font_size``."""

    return {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "figure.labelsize": font_size,
        "legend.frameon": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.5,
        "axes.labelcolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
    }


@contextmanager
def plot_style(font_size: float = FONT_SIZE) -> Iterator[None]:
    """Apply the style to figures made inside the block, then restore."""

    plt = import_pyplot()
    _register_fonts()
    with plt.style.context("default"), plt.rc_context(style_settings(font_size)):
        yield


def apply_style(font_size: float = FONT_SIZE) -> None:
    """Apply the style to every figure from now on."""

    plt = import_pyplot()
    _register_fonts()
    plt.style.use("default")
    plt.rcParams.update(style_settings(font_size))
