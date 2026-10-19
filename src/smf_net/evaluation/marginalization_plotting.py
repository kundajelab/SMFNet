"""Plots of motif marginalizations: mean SMF around an inserted motif.

Each panel shows one motif's profile, the model's reads averaged per synthetic
background and then summarized over backgrounds, against its matched control.
Both orientations of a motif are pooled by default, as in the SMFNet paper.
Every figure uses the shared style in :mod:`smf_net.evaluation.plot_style`.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal

import h5py
import torch
import torch.nn.functional as F
from tqdm import tqdm

from smf_net.data import site_means
from smf_net.evaluation.marginalization import (
    MarginalizationDesign,
    MarginalizationSampleBatch,
)
from smf_net.evaluation.plot_style import (
    EXPECTED_FILL,
    GREYS,
    REPORT_FONT_SIZE,
    SMFNET_ALL_READS,
    import_pyplot,
    plot_style,
)

Statistic = Literal["median", "mean"]

MOTIF_COLOR = SMFNET_ALL_READS
CONTROL_COLOR = GREYS[0]


@dataclass(frozen=True)
class MotifProfiles:
    """Per-background profiles of one motif and its control, pooled over the
    orientations being combined.

    ``motif`` and ``control`` are ``[backgrounds, positions]`` mean SMF, NaN
    where no read has a call; ``x`` gives each position relative to the motif
    center in bp.
    """

    name: str
    length: int
    x: torch.Tensor
    motif: torch.Tensor
    control: torch.Tensor


def _read_mean(reads: torch.Tensor) -> torch.Tensor:
    """Mean over reads at each position, NaN where no read has a call."""

    means, counts = site_means(reads)
    return means.masked_fill(counts == 0, torch.nan)


def _smooth(profiles: torch.Tensor, window: int) -> torch.Tensor:
    """Moving average over positions, ignoring NaN."""

    if window == 1:
        return profiles
    original_shape = profiles.shape
    profiles = profiles.reshape(-1, original_shape[-1])
    valid = torch.isfinite(profiles)
    values = profiles.nan_to_num().unsqueeze(1)
    weights = valid.float().unsqueeze(1)
    kernel = torch.ones(1, 1, window, dtype=values.dtype)
    sums = F.conv1d(values, kernel, padding=window // 2)
    counts = F.conv1d(weights, kernel, padding=window // 2)
    smoothed = (
        (sums / counts.clamp_min(1))
        .squeeze(1)
        .masked_fill(
            counts.squeeze(1) == 0,
            torch.nan,
        )
    )
    return smoothed.reshape(original_shape)


def _summary(
    profiles: torch.Tensor,
    statistic: Statistic,
    interval: tuple[float, float] | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if statistic == "median":
        center = torch.nanmedian(profiles, dim=0).values
    else:
        center = torch.nanmean(profiles, dim=0)
    if interval is None:
        return center, None, None
    lower, upper = interval
    return (
        center,
        torch.nanquantile(profiles, lower, dim=0),
        torch.nanquantile(profiles, upper, dim=0),
    )


def effect(profiles: MotifProfiles, window_bp: int = 100) -> float:
    """Motif minus control mean SMF within ``window_bp`` of the motif center.

    Unsmoothed, averaged over positions and backgrounds.
    """

    within = profiles.x.abs() <= window_bp
    return float(
        torch.nanmean(profiles.motif[:, within]) - torch.nanmean(profiles.control[:, within])
    )


def _base_name(name: str, orientation: str) -> str:
    suffix = f"({orientation})"
    return name.removesuffix(suffix)


def _pool(
    names: list[str],
    orientations: list[str],
    lengths: list[int],
    x: list[torch.Tensor],
    motif: list[torch.Tensor],
    control: list[torch.Tensor],
    combine_orientations: bool,
) -> list[MotifProfiles]:
    """Group per-motif profiles by motif, pooling the backgrounds of each group."""

    groups: dict[str, list[int]] = {}
    for index, (name, orientation) in enumerate(zip(names, orientations)):
        key = _base_name(name, orientation) if combine_orientations else name
        groups.setdefault(key, []).append(index)
    return [
        MotifProfiles(
            name=key,
            length=lengths[indices[0]],
            # A motif and its reverse complement are inserted at the same
            # center, so their positions line up without flipping.
            x=x[indices[0]],
            motif=torch.cat([motif[index] for index in indices]),
            control=torch.cat([control[index] for index in indices]),
        )
        for key, indices in groups.items()
    ]


def marginalization_profiles_h5(
    path: str | PathLike[str],
    *,
    combine_orientations: bool = True,
) -> list[MotifProfiles]:
    """Read per-background profiles of every motif from a result HDF5 file.

    Each motif's reads are loaded and reduced to one profile per background in
    turn, rather than loading the complete sample arrays.
    """

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "smfnet.marginalizations":
            raise ValueError("Expected an smfnet.marginalizations HDF5 file")
        names = handle["motif_names"].asstr()[:].tolist()
        orientations = handle["motif_orientations"].asstr()[:].tolist()
        sequences = handle["motif_sequences"].asstr()[:].tolist()
        offset = int(handle.attrs["output_offset"])
        lengths = [len(sequence) for sequence in sequences]
        motif, control, x = [], [], []
        for index, length in enumerate(lengths):
            motif.append(_read_mean(torch.from_numpy(handle["samples/motif"][index])))
            control.append(_read_mean(torch.from_numpy(handle["samples/control"][index])))
            center = int(handle["insertion_starts"][index]) + length / 2
            x.append(offset + torch.arange(motif[-1].shape[-1]) - center)
    return _pool(names, orientations, lengths, x, motif, control, combine_orientations)


def marginalization_profiles(
    design: MarginalizationDesign,
    batches: Iterable[MarginalizationSampleBatch],
    *,
    combine_orientations: bool = True,
    show_progress: bool = True,
) -> list[MotifProfiles]:
    """Reduce streaming sample batches to per-background motif profiles.

    Reads are reduced to one profile per background as they arrive, so the
    full raw sample tensor is not retained in memory.
    """

    motif = control = None
    total = 2 * design.n_motifs * design.n_backgrounds
    with tqdm(total=total, desc="Marginalization", disable=not show_progress) as progress:
        for batch in batches:
            if batch.condition == "background":
                continue
            profiles = _read_mean(batch.samples.reads)
            if motif is None:
                shape = (design.n_motifs, design.n_backgrounds, profiles.shape[-1])
                motif = torch.full(shape, torch.nan)
                control = torch.full(shape, torch.nan)
                offset = batch.samples.output_offset
            target = motif if batch.condition == "motif" else control
            target[batch.motif_index, list(batch.background_indices)] = profiles
            progress.update(len(batch.background_indices))
    if motif is None:
        raise ValueError("No marginalization sample batches were provided")

    lengths = [len(m.sequence) for m in design.motifs]
    x = [
        offset + torch.arange(motif.shape[-1]) - (int(start) + length / 2)
        for start, length in zip(design.insertion_starts, lengths)
    ]
    return _pool(
        [m.name for m in design.motifs],
        list(design.orientations),
        lengths,
        x,
        list(motif),
        list(control),
        combine_orientations,
    )


def _plot_curve(axis, x, profiles, *, statistic, interval, color, label, linestyle):
    center, lower, upper = _summary(profiles, statistic, interval)
    finite = torch.isfinite(center)
    axis.plot(
        x[finite].numpy(),
        center[finite].numpy(),
        color=color,
        label=label,
        linestyle=linestyle,
    )
    if lower is not None and upper is not None:
        band = finite & torch.isfinite(lower) & torch.isfinite(upper)
        axis.fill_between(
            x[band].numpy(),
            lower[band].numpy(),
            upper[band].numpy(),
            color=color,
            alpha=0.15,
            linewidth=0,
        )


def plot_marginalizations(
    profiles: list[MotifProfiles],
    *,
    smoothing_window: int = 31,
    statistic: Statistic = "mean",
    interval: tuple[float, float] | None = None,
    xlim_bp: int | None = 400,
    effect_window_bp: int | None = 100,
    columns: int = 4,
) -> Any:
    """Plot every motif as one panel of a grid with a shared y axis.

    Profiles are smoothed with an odd ``smoothing_window`` (bp), then summarized
    over backgrounds by ``statistic``, with an optional quantile ``interval``
    band. The motif is shaded, and ``effect_window_bp`` annotates each panel
    with :func:`effect`. ``None`` turns off the axis limit or the annotation.
    """

    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd number")
    plt = import_pyplot()
    columns = min(columns, len(profiles))
    rows = math.ceil(len(profiles) / columns)
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(2.6 * columns, 2.0 * rows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for axis, motif in zip(axes.flat, profiles):
            axis.axvspan(-motif.length / 2, motif.length / 2, color=EXPECTED_FILL, linewidth=0)
            for condition, color, linestyle in (
                ("control", CONTROL_COLOR, (0, (3, 2))),
                ("motif", MOTIF_COLOR, "-"),
            ):
                _plot_curve(
                    axis,
                    motif.x,
                    _smooth(getattr(motif, condition), smoothing_window),
                    statistic=statistic,
                    interval=interval,
                    color=color,
                    label=condition.capitalize(),
                    linestyle=linestyle,
                )
            if effect_window_bp is not None:
                axis.text(
                    0.97,
                    0.05,
                    f"Δ {effect(motif, effect_window_bp):+.2f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                )
            if xlim_bp is not None:
                axis.set_xlim(-xlim_bp, xlim_bp)
            axis.set_title(motif.name)
        for axis in axes.flat[len(profiles) :]:
            axis.set_visible(False)
        for axis in axes[-1]:
            axis.set_xlabel("Position from motif center (bp)")
        for axis in axes[:, 0]:
            axis.set_ylabel(f"{statistic.capitalize()} SMF")
        axes.flat[0].legend(loc="upper left")
    return figure


def plot_marginalizations_h5(
    path: str | PathLike[str],
    *,
    combine_orientations: bool = True,
    **options: Any,
) -> Any:
    """Plot every motif of a marginalization result file on one figure.

    ``options`` are those of :func:`plot_marginalizations`.
    """

    return plot_marginalizations(
        marginalization_profiles_h5(path, combine_orientations=combine_orientations),
        **options,
    )
