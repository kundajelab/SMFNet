"""Optional plotting helpers for PRISM results.

Every figure uses the shared style in :mod:`smf_net.evaluation.plot_style`.
"""

from typing import Any

import torch

from smf_net.evaluation.ism_plotting import (
    _mutation_axis_label,
    _plot_logo,
    _wildtype_logo_values,
)
from smf_net.evaluation.plot_style import (
    ACCENT,
    REPORT_FONT_SIZE,
    SMFNET_ALL_READS,
    import_pyplot,
    plot_style,
)
from smf_net.evaluation.prism import PRISMLogRatioResult, PRISMResult


def _plot_read(axis: Any, read: torch.Tensor, color: str, label: str | None) -> None:
    positions = torch.arange(len(read))
    observed = read != -1
    axis.scatter(positions[observed], read[observed], color=color, label=label, s=12)
    axis.plot(positions[observed], read[observed], color=color, alpha=0.7)


def plot_prism(
    result: PRISMResult,
    region_id: str | None = None,
) -> Any:
    """Plot an observed read and its wild-type and substitution effects."""

    plt, plot_logo = import_pyplot(), _plot_logo()
    deltas = result.deltas.float().cpu()
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(nrows=4, figsize=(7, 7), sharex=True)
        read_axis, positive_axis, negative_axis, delta_axis = axes

        _plot_read(read_axis, result.read.cpu(), SMFNET_ALL_READS, None)
        read_axis.set_ylabel("SMF signal")
        if region_id is not None:
            read_axis.set_title(region_id)

        positive = -deltas.min(dim=1).values
        negative = -deltas.max(dim=1).values
        plot_logo(_wildtype_logo_values(result.dna, positive), ax=positive_axis)
        plot_logo(_wildtype_logo_values(result.dna, negative), ax=negative_axis)
        plot_logo(deltas.T, ax=delta_axis)

        positive_axis.set_ylabel("Positive\ninfluence")
        negative_axis.set_ylabel("Negative\ninfluence")
        delta_axis.set_ylabel(r"$\Delta \log p$")
        delta_axis.set_xlabel(_mutation_axis_label(result))
    return figure


def plot_prism_log_ratio(
    result: PRISMLogRatioResult,
    region_id: str | None = None,
) -> Any:
    """Plot two reads and DNA positions favoring either read's likelihood."""

    plt, plot_logo = import_pyplot(), _plot_logo()
    deltas = result.deltas.float().cpu()
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(nrows=3, figsize=(7, 6), sharex=True)
        read_axis, read_a_axis, read_b_axis = axes

        _plot_read(read_axis, result.read_a.cpu(), SMFNET_ALL_READS, "Read A")
        _plot_read(read_axis, result.read_b.cpu(), ACCENT, "Read B")
        read_axis.set_ylabel("SMF signal")
        read_axis.legend(loc="upper right")
        if region_id is not None:
            read_axis.set_title(region_id)

        favors_a = -deltas.min(dim=1).values
        favors_b = deltas.max(dim=1).values
        plot_logo(_wildtype_logo_values(result.dna, favors_a), ax=read_a_axis)
        plot_logo(_wildtype_logo_values(result.dna, favors_b), ax=read_b_axis)

        read_a_axis.set_ylabel("Favors read A")
        read_b_axis.set_ylabel("Favors read B")
        read_b_axis.set_xlabel(_mutation_axis_label(result))
    return figure
