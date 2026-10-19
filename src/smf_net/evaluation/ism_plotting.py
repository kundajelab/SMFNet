"""Optional plotting helpers for profile ISM results.

Every figure uses the shared style in :mod:`smf_net.evaluation.plot_style`.
"""

from typing import Any

import torch

from smf_net.data import dna_tensor_to_one_hot
from smf_net.evaluation.ism import ProfileISM
from smf_net.evaluation.plot_style import REPORT_FONT_SIZE, import_pyplot, plot_style


def _plot_logo() -> Any:
    try:
        from tangermeme.plot import plot_logo
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ISM plotting requires the 'plotting' extra: pip install 'SMFNet[plotting]'"
        ) from error
    return plot_logo


def _wildtype_logo_values(dna: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    one_hot = dna_tensor_to_one_hot(dna.unsqueeze(0))[0].cpu()
    return one_hot * values.cpu().unsqueeze(0)


def _mutation_axis_label(result: Any) -> str:
    return f"Mutation position (input offset {result.mutation_start})"


def plot_profile_ism(
    result: ProfileISM,
    region_id: str | None = None,
    *,
    scaled: bool = False,
) -> Any:
    """Plot a profile influence heatmap and wild-type attribution logo."""

    plt, plot_logo = import_pyplot(), _plot_logo()
    from matplotlib.colors import TwoSlopeNorm

    influence = (result.scaled_influence if scaled else result.influence).float().cpu()
    max_abs = float(influence.abs().max()) or 1.0
    with plot_style(REPORT_FONT_SIZE):
        figure = plt.figure(figsize=(13, 9))
        grid = figure.add_gridspec(2, 2, width_ratios=[30, 1], wspace=0.02, hspace=0.1)
        heatmap_axis = figure.add_subplot(grid[0, 0])
        logo_axis = figure.add_subplot(grid[1, 0], sharex=heatmap_axis)
        colorbar_axis = figure.add_subplot(grid[0, 1])

        # Effects are signed, so they get a diverging map centered on zero.
        image = heatmap_axis.imshow(
            influence.numpy(),
            aspect="auto",
            origin="lower",
            cmap="bwr",
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
        )
        heatmap_axis.set_ylabel("Mutation position")
        if region_id is not None:
            heatmap_axis.set_title(region_id)
        figure.colorbar(image, cax=colorbar_axis, orientation="vertical")

        plot_logo(_wildtype_logo_values(result.dna, influence.sum(dim=1)), ax=logo_axis)
        logo_axis.set_xlabel(_mutation_axis_label(result))
        logo_axis.set_ylabel("Attribution")
    return figure
