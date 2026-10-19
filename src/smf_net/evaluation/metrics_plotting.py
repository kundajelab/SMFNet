"""Optional plotting helpers for evaluation metric tables.

The plots show Pearson agreement only; the Spearman columns stay in the
metrics table, where they tell the same story. Every figure uses the shared
style in :mod:`smf_net.evaluation.plot_style`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smf_net.evaluation.covariation import (
    DECAY_PREFIX,
    SeparationBins,
    pooled_decay,
)
from smf_net.evaluation.plot_style import (
    ACCENT,
    BASELINE,
    GREYS,
    POINT,
    PSEUDO_REP,
    REPORT_FONT_SIZE,
    SMFNET,
    SMFNET_ALL_READS,
    import_pyplot,
    plot_style,
)
from smf_net.evaluation.run_lengths import (
    HISTOGRAM_PREFIX,
    RUN_VALUES,
    RunLengthBins,
    pooled_histogram,
)
from smf_net.evaluation.summary import (
    gap_closed_columns,
    gap_closed_fraction,
    metric_unit,
)

# Observed reads, and the two observed shuffles drawn against them.
OBSERVED = POINT
SHUFFLE = GREYS[0]
READ_AXIS_SHUFFLE = BASELINE


def _finite_values(table: pd.DataFrame, column: str) -> np.ndarray:
    values = table[column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _title_with_gap_closed(table: pd.DataFrame, column: str, title: str) -> str:
    """Annotate a panel with where the pooled model median sits.

    Per-region gap-closed ratios are unusable because the baseline and floor
    are each within sampling noise of the model for a single region, so the
    annotation reports one pooled number for the regions in the panel.
    """

    if column not in gap_closed_columns(table.columns):
        return title
    fraction = gap_closed_fraction(table, column)
    if not np.isfinite(fraction):
        return title
    return f"{title}\n{fraction:.0%} of shuffle gap closed"


@dataclass(frozen=True)
class Series:
    """One histogram of a :class:`Panel`.

    A filled series is a reference distribution, drawn underneath the outlined
    series it is read against.
    """

    column: str
    label: str
    color: str
    filled: bool = False


PSEUDO_REPLICATE_LABEL = "Pseudo-replicate"


def _pseudo_replicate(column: str) -> Series:
    return Series(column, PSEUDO_REPLICATE_LABEL, PSEUDO_REP, filled=True)


@dataclass(frozen=True)
class Panel:
    """One per-region histogram panel of :func:`plot_metrics`.

    The panel is drawn only when the column of its first series is present.
    Every series in a panel is scored against the same observed reference, so
    they can be read against each other. ``ceiling`` names a pseudo-replicate
    column whose noise-free ceiling (see :func:`noise_free_ceiling`) is marked
    on the panel, and ``gap_closed`` asks for the pooled gap-closed annotation.
    """

    title: str
    series: tuple[Series, ...]
    ceiling: str | None = None
    gap_closed: bool = False


def noise_free_ceiling(table: pd.DataFrame, replicate_column: str) -> np.ndarray:
    """Per-region agreement a noise-free predictor can reach with every read.

    The pseudo-replicate agreement ``R`` is the reliability of a profile
    estimated from the matched read count. Every observed read is ``k`` times
    as many, which the Spearman-Brown formula turns into a reliability of
    ``k R / (1 + (k - 1) R)``. A predictor with no noise of its own is
    attenuated only by the observed side, so it can reach the square root of
    that. The model's own reads are treated as noise free, so this is a
    slight overestimate for it.
    """

    reliability = table[replicate_column].to_numpy(dtype=float).clip(0.0, 1.0)
    k = table["n_observed_reads"].to_numpy(dtype=float) / table["n_matched_reads"].to_numpy(
        dtype=float
    )
    return np.sqrt(k * reliability / (1 + (k - 1) * reliability))


def _matched_and_all_reads_panels(
    name: str,
    column: str,
    matched: str,
    baselines: tuple[Series, ...] = (),
) -> tuple[Panel, Panel]:
    """A matched-read panel and an every-read panel for one agreement metric.

    The matched model and the pseudo-replicate are scored against one observed
    half at the same read count, while the every-read model and the train-set
    baselines carry no sampling noise of their own and are scored against
    every observed read. Splitting them keeps each panel like for like.
    """

    replicate = f"{matched}_pseudo_replicate"
    return (
        Panel(
            f"{name}\nmatched reads (Pearson)",
            (Series(matched, "Model", SMFNET), _pseudo_replicate(replicate)),
        ),
        Panel(
            f"{name}\nall reads (Pearson)",
            (Series(column, "Model", SMFNET_ALL_READS), *baselines),
            ceiling=replicate,
        ),
    )


PROFILE_PANELS = _matched_and_all_reads_panels(
    "Profile agreement",
    "pearson",
    "pearson_matched",
    (
        Series("pearson_positional_baseline", "Positional baseline (train)", BASELINE),
        Series("pearson_flank_baseline", "Flank baseline (train)", ACCENT),
    ),
)

FLANK_PANELS = _matched_and_all_reads_panels(
    "Profile agreement, flank baseline removed",
    "pearson_flank_residual",
    "pearson_flank_residual_matched",
)

# Run-length distances only exist on the matched samples, where the shuffles
# and the pseudo-replicate bound them from both sides.
RUN_LENGTH_PANELS = tuple(
    Panel(
        f"{name.capitalize()} run-length Wasserstein (bp)",
        (
            Series(f"run_length_{name}_wasserstein", "Model", SMFNET),
            Series(f"run_length_{name}_wasserstein_shuffle", "Observed shuffle", SHUFFLE),
            Series(
                f"run_length_{name}_wasserstein_read_axis_shuffle",
                "Observed read-axis shuffle",
                READ_AXIS_SHUFFLE,
            ),
            _pseudo_replicate(f"run_length_{name}_wasserstein_pseudo_replicate"),
        ),
        gap_closed=True,
    )
    for name in RUN_VALUES
)

COVARIATION_PANELS = _matched_and_all_reads_panels(
    "Pairwise covariation agreement",
    "covariation_corr_of_corr_pearson_all_reads",
    "covariation_corr_of_corr_pearson",
    (
        Series(
            "covariation_corr_of_corr_pearson_positional_baseline",
            "Positional baseline (train)",
            BASELINE,
        ),
    ),
) + _matched_and_all_reads_panels(
    "Residual covariation agreement, baseline removed",
    "covariation_residual_corr_of_corr_pearson_all_reads",
    "covariation_residual_corr_of_corr_pearson",
)

PANELS = PROFILE_PANELS + FLANK_PANELS + RUN_LENGTH_PANELS + COVARIATION_PANELS


def _metric_panels(metrics: pd.DataFrame) -> tuple[Panel, ...]:
    """Keep each panel only when the metrics include its first column."""

    return tuple(panel for panel in PANELS if panel.series[0].column in metrics)


def _plot_distribution(
    axis: Any,
    table: pd.DataFrame,
    panel: Panel,
    *,
    bins: int,
) -> None:
    first_column = panel.series[0].column
    present = [
        series
        for series in panel.series
        if series.column in table and _finite_values(table, series.column).size
    ]
    # One set of bin edges for every series keeps their heights comparable.
    edges = np.histogram_bin_edges(
        np.concatenate([_finite_values(table, series.column) for series in present])
        if present
        else np.zeros(0),
        bins=bins,
    )
    # Reference distributions go underneath, so outlines stay visible on top.
    for series in sorted(present, key=lambda series: not series.filled):
        values = _finite_values(table, series.column)
        median = np.median(values)
        axis.hist(
            values,
            bins=edges,
            histtype="stepfilled" if series.filled else "step",
            color=series.color,
            linewidth=1.2,
            label=f"{series.label} (median {median:.3f})",
        )
        axis.axvline(
            median,
            # A reference fill is too light to read as a line.
            color="black" if series.filled else series.color,
            linestyle="--",
            linewidth=1,
        )

    if panel.ceiling is not None and {panel.ceiling, "n_observed_reads", "n_matched_reads"} <= set(
        table
    ):
        ceiling = noise_free_ceiling(table, panel.ceiling)
        ceiling = ceiling[np.isfinite(ceiling)]
        if ceiling.size:
            median = np.median(ceiling)
            axis.axvline(
                median,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"Noise-free ceiling, est. (median {median:.3f})",
            )

    title = panel.title
    if panel.gap_closed:
        title = _title_with_gap_closed(table, first_column, title)
    axis.set_title(title)
    axis.set_ylabel("Regions")
    if unit := metric_unit(first_column):
        axis.set_xlabel(unit)
    if axis.has_data():
        axis.legend(frameon=False)


def plot_metrics(
    metrics: pd.DataFrame,
    *,
    min_correlation_sites: int | None = None,
    bins: int = 40,
) -> Any:
    """Plot per-region metric distributions and coverage diagnostics on one page.

    Every panel counts regions, including the run-length distance panels, so the
    pooled run-length distributions over individual runs are plotted separately
    by :func:`plot_run_lengths`.

    Agreement metrics come in rows of two panels. The left one holds the
    matched model and the pseudo-replicate, scored against one observed half
    at the same read count. The right one holds the every-read model and the
    train-set baselines, scored against every observed read, with a dotted
    line at the median noise-free ceiling.

    No region is filtered here. Statistics that rest on too little data are
    already NaN in the table, so each panel shows exactly the regions its
    metric was scored on. ``min_correlation_sites`` only marks the floor the
    profile correlations used on the coverage panel.
    """

    if bins < 1:
        raise ValueError("bins must be positive")

    panels = _metric_panels(metrics)

    plt = import_pyplot()
    rows = -(-(len(panels) + 1) // 2)
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(
            rows,
            2,
            figsize=(11, 3.4 * rows),
            squeeze=False,
            layout="constrained",
        )
        for axis, panel in zip(axes.flat, panels, strict=False):
            _plot_distribution(axis, metrics, panel, bins=bins)

        coverage_axis = axes.flat[len(panels)]
        for unused_axis in axes.flat[len(panels) + 1 :]:
            unused_axis.set_visible(False)
        coverage = _finite_values(metrics, "n_valid_positions")
        if coverage.size:
            coverage_axis.hist(coverage, bins=bins, color=GREYS[1])
            if min_correlation_sites is not None:
                coverage_axis.axvline(
                    min_correlation_sites,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label=f"Minimum for correlations: {min_correlation_sites}",
                )
                coverage_axis.legend(frameon=False)
        coverage_axis.set_title("Valid positions per region")
        coverage_axis.set_xlabel("Positions")
        coverage_axis.set_ylabel("Regions")

        figure.suptitle(f"Evaluation metrics ({len(metrics):,} regions)")
    return figure


def plot_site_correlations(summary: pd.DataFrame) -> Any:
    """Plot pooled site Pearson correlations for full and matched read targets."""

    plt = import_pyplot()
    with plot_style(REPORT_FONT_SIZE):
        return _plot_site_correlations(plt, summary)


def _plot_site_correlations(plt: Any, summary: pd.DataFrame) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    indexed = summary.set_index("comparison")
    for axis, (group, title, model_color) in zip(
        axes,
        (
            ("all_reads", "All reads", SMFNET_ALL_READS),
            ("matched", "Matched reads", SMFNET),
        ),
    ):
        series = (
            ("model", "Model", model_color),
            ("flank_baseline", "Flank baseline", ACCENT),
            ("pseudo_replicate", PSEUDO_REPLICATE_LABEL, PSEUDO_REP),
        )
        labels = []
        for name, label, color in series:
            key = f"{name}_{group}"
            if key not in indexed.index:
                continue
            result = indexed.loc[key]
            position = len(labels)
            labels.append(f"{label}\n{int(result.n_sites):,} sites")
            value = result.pearson
            if np.isfinite(value):
                axis.bar(position, value, color=color, edgecolor="black", width=0.6)
                axis.annotate(
                    f"{value:.3f}",
                    (position, value),
                    xytext=(0, 4 if value >= 0 else -4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                )
            else:
                axis.text(position, 0.03, "Undefined", ha="center")
        axis.set_xticks(range(len(labels)), labels)
        axis.set_xlim(-0.6, max(len(labels) - 0.4, 0.6))
        axis.set_ylim(-1.12, 1.12)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_ylabel("Correlation across sites")
        axis.set_title(f"{title}: Pearson")
        if not labels:
            axis.text(0.5, 0.5, "No eligible sites", transform=axis.transAxes, ha="center")
    figure.suptitle("Pooled site methylation correlations")
    return figure


# Column suffix, legend label, colour and line style of each pooled curve.
_CURVE_SERIES = (
    ("observed", "Observed", OBSERVED, "-"),
    ("model", "Model", SMFNET_ALL_READS, "-"),
    ("shuffle", "Observed within-read shuffle", SHUFFLE, "--"),
    ("read_axis_shuffle", "Observed read-axis shuffle", READ_AXIS_SHUFFLE, "--"),
)


def _plot_pooled_run_lengths(
    axis: Any,
    metrics: pd.DataFrame,
    name: str,
    bins: RunLengthBins,
    *,
    weight_by_coverage: bool,
) -> None:
    centers = bins.edges[:-1] + bins.bin_width_bp / 2
    for suffix, label, color, linestyle in _CURVE_SERIES:
        column = f"{HISTOGRAM_PREFIX}{name}_{suffix}"
        if column not in metrics:
            continue
        pooled = pooled_histogram(
            metrics,
            column,
            weight_by_coverage=weight_by_coverage,
        )
        total = pooled.sum()
        if total == 0:
            continue
        axis.step(
            centers,
            pooled / total,
            where="mid",
            label=label,
            color=color,
            linestyle=linestyle,
        )

    weighting = "coverage-weighted" if weight_by_coverage else "equal-weighted"
    axis.set_title(f"{name.capitalize()} run lengths ({weighting})")
    axis.set_xlabel(f"Run length (bp, last bin is >={bins.max_bp})")
    axis.set_ylabel("Fraction of runs")
    if axis.has_data():
        axis.legend(frameon=False)


def plot_run_lengths(
    metrics: pd.DataFrame,
    bins: RunLengthBins = RunLengthBins(),
) -> Any:
    """Plot run-length distributions pooled over every region.

    Each row pools the per-region histograms, once weighting regions by their
    read and site coverage and once weighting them equally. Unlike the panels in
    :func:`plot_metrics`, these count individual runs rather than regions. The
    per-region distance against the observed pseudo-replicate floor is a
    ``plot_metrics`` panel.
    """

    plt = import_pyplot()
    names = list(RUN_VALUES)
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(
            2,
            len(names),
            figsize=(5.5 * len(names), 7),
            squeeze=False,
            layout="constrained",
        )
        for column_index, name in enumerate(names):
            for row_index, weight_by_coverage in enumerate((True, False)):
                _plot_pooled_run_lengths(
                    axes[row_index][column_index],
                    metrics,
                    name,
                    bins,
                    weight_by_coverage=weight_by_coverage,
                )

        figure.suptitle(f"Pooled run lengths ({len(metrics):,} regions)")
    return figure


def plot_covariation(
    metrics: pd.DataFrame,
    bins: SeparationBins = SeparationBins(),
) -> Any:
    """Plot how pairwise correlation decays with genomic separation.

    The left panel pools every region's binned correlations, so a region
    contributes in proportion to how many usable position pairs it has. The
    right panel shows how many pairs support each bin, because the tail of the
    curve rests on far fewer pairs than its head.

    The read-axis shuffle is the curve to read against: it keeps every position
    marginal and destroys only the coupling between positions, so it sits at
    zero and marks where the observed curve stops carrying real covariation.
    """

    plt = import_pyplot()
    with plot_style(REPORT_FONT_SIZE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4),
            squeeze=False,
            layout="constrained",
        )
        decay_axis, support_axis = axes[0]
        centers = bins.centers

        for suffix, label, color, linestyle in _CURVE_SERIES:
            if f"{DECAY_PREFIX}sum_{suffix}" not in metrics:
                continue
            style = {"label": label, "color": color, "linestyle": linestyle}
            decay_axis.plot(centers, pooled_decay(metrics, suffix), **style)
            counts = np.stack(
                [np.asarray(value) for value in metrics[f"{DECAY_PREFIX}count_{suffix}"]]
            ).sum(axis=0)
            support_axis.plot(centers, counts, **style)

        decay_axis.axhline(0.0, color="black", linewidth=0.8)
        decay_axis.set_title("Pairwise correlation vs separation")
        decay_axis.set_xlabel("Separation between positions (bp)")
        decay_axis.set_ylabel("Mean correlation")
        if decay_axis.has_data():
            decay_axis.legend(frameon=False)

        support_axis.set_title("Position pairs per bin")
        support_axis.set_xlabel("Separation between positions (bp)")
        support_axis.set_ylabel("Pairs")
        support_axis.set_yscale("log")

        figure.suptitle(f"Pairwise covariation ({len(metrics):,} regions)")
    return figure
