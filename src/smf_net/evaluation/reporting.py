"""Reporting helpers for paired-sample evaluation artifacts."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from smf_net.evaluation.covariation import decay_columns
from smf_net.evaluation.flank_baselines import FlankBaseline, fit_flank_baseline
from smf_net.evaluation.metrics import (
    MetricRecord,
    MetricsConfig,
    compute_metrics_table,
    metrics_table,
    region_generator,
    score_region,
)
from smf_net.evaluation.positional_baselines import (
    PositionalBaselines,
    fit_baselines_for_samples,
    load_training_split,
)
from smf_net.evaluation.run_lengths import RUN_VALUES, histogram_columns
from smf_net.evaluation.sample_io import (
    SourceRegistry,
    read_sample_region,
    read_samples_h5,
    read_source_registry,
)
from smf_net.evaluation.site_correlations import SITE_PREFIX, site_correlation_summary
from smf_net.evaluation.summary import gap_closed_summary

_WORKER_SAMPLES_HANDLE: h5py.File | None = None
_WORKER_METRICS_CONFIG: MetricsConfig | None = None
_WORKER_SOURCE_REGISTRY: SourceRegistry | None = None
_WORKER_BASELINES: PositionalBaselines | None = None
_WORKER_FLANK_BASELINE: FlankBaseline | None = None


def _initialize_metrics_worker(
    samples_path: str,
    config: MetricsConfig,
    baselines: PositionalBaselines | None,
    flank_baseline: FlankBaseline | None,
) -> None:
    global _WORKER_SAMPLES_HANDLE, _WORKER_METRICS_CONFIG, _WORKER_SOURCE_REGISTRY
    global _WORKER_BASELINES, _WORKER_FLANK_BASELINE
    torch.set_num_threads(1)
    _WORKER_SAMPLES_HANDLE = h5py.File(samples_path, "r")
    _WORKER_METRICS_CONFIG = config
    _WORKER_SOURCE_REGISTRY = read_source_registry(_WORKER_SAMPLES_HANDLE)
    _WORKER_BASELINES = baselines
    _WORKER_FLANK_BASELINE = flank_baseline


def _compute_h5_region_metrics(task: tuple[int, str]) -> MetricRecord | str:
    region_index, region_key = task
    if (
        _WORKER_SAMPLES_HANDLE is None
        or _WORKER_METRICS_CONFIG is None
        or _WORKER_SOURCE_REGISTRY is None
    ):
        raise RuntimeError("Metrics worker was not initialized")
    region = read_sample_region(
        _WORKER_SAMPLES_HANDLE["regions"][region_key],
        _WORKER_SOURCE_REGISTRY,
    )
    return score_region(
        region,
        _WORKER_METRICS_CONFIG,
        region_generator(_WORKER_METRICS_CONFIG.seed, region_index),
        _WORKER_BASELINES,
        _WORKER_FLANK_BASELINE,
    )


def _parallel_metrics_from_h5(
    samples_path: str,
    region_keys: list[str],
    config: MetricsConfig,
    baselines: PositionalBaselines | None,
    flank_baseline: FlankBaseline | None,
) -> pd.DataFrame:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        mp_context=context,
        initializer=_initialize_metrics_worker,
        initargs=(samples_path, config, baselines, flank_baseline),
    ) as executor:
        records = list(
            tqdm(
                executor.map(
                    _compute_h5_region_metrics,
                    enumerate(region_keys),
                    chunksize=1,
                ),
                total=len(region_keys),
                desc="computing metrics",
                unit="region",
                disable=not config.progress,
            )
        )
    return metrics_table(records)


def train_baselines_for(
    samples_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
) -> tuple[PositionalBaselines | None, FlankBaseline | None]:
    """Fit the train-set baselines the config includes for a samples file.

    The training data is loaded once for both. Pass the results to
    :func:`metrics_from_h5` or :func:`write_metrics_tsv`.
    """

    positional = config.include_positional_baselines
    flank = config.include_flank_baselines
    if not (positional or flank):
        return None, None
    training = load_training_split(samples_path)
    positional_baselines = (
        fit_baselines_for_samples(
            samples_path,
            config.positional_baseline_settings,
            progress=config.progress,
            training=training,
        )
        if positional
        else None
    )
    flank_baseline = (
        fit_flank_baseline(
            training.reads,
            training.positions,
            training.dna,
            training.config.assay_type,
            training.config.train_chrm,
            config.flank_baseline_settings,
        )
        if flank
        else None
    )
    return positional_baselines, flank_baseline


def metrics_from_h5(
    samples_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
    baselines: PositionalBaselines | None = None,
    flank_baseline: FlankBaseline | None = None,
) -> pd.DataFrame:
    """Score every region of a samples file.

    The train-set baselines the config includes must be fit first, with
    :func:`train_baselines_for`, and passed in.
    """

    missing = [
        name
        for name, included, fitted in (
            ("positional", config.include_positional_baselines, baselines),
            ("flank", config.include_flank_baselines, flank_baseline),
        )
        if included and fitted is None
    ]
    if missing:
        raise ValueError(
            f"The config includes {' and '.join(missing)} baselines, but none were "
            "passed; fit them with train_baselines_for or turn them off"
        )

    samples_path = str(Path(samples_path).resolve())
    with h5py.File(samples_path, "r") as handle:
        if handle.attrs.get("schema") != "smfnet.samples":
            raise ValueError("Expected an smfnet.samples HDF5 file")
        total_regions = int(handle.attrs.get("num_regions", len(handle["regions"])))
        region_keys = sorted(handle["regions"])

    if config.num_workers == 0:
        metrics = compute_metrics_table(
            read_samples_h5(samples_path),
            config,
            total_regions=total_regions,
            baselines=baselines,
            flank_baseline=flank_baseline,
        )
    else:
        metrics = _parallel_metrics_from_h5(
            samples_path,
            region_keys,
            config,
            baselines,
            flank_baseline,
        )

    return metrics


# Metric group, the column whose NaNs it classifies, the count that column
# rests on, and the floor that count must reach. Shuffle columns share these
# floors, so they would only repeat the same rows.
def _drop_checks(config: MetricsConfig) -> list[tuple[str, str, str, int, str]]:
    checks = [
        (
            "profile",
            "pearson",
            "n_valid_positions",
            config.min_correlation_sites,
            "sites",
        ),
        (
            "profile, matched reads",
            "pearson_matched",
            "n_matched_correlation_sites",
            config.min_correlation_sites,
            "sites",
        ),
        (
            "covariation",
            "covariation_corr_of_corr_pearson",
            "n_covariation_pairs",
            config.min_correlation_pairs,
            "pairs",
        ),
        (
            "covariation, all reads",
            "covariation_corr_of_corr_pearson_all_reads",
            "n_covariation_pairs_all_reads",
            config.min_correlation_pairs,
            "pairs",
        ),
        (
            "covariation, pseudo-replicate",
            "covariation_corr_of_corr_pearson_pseudo_replicate",
            "n_covariation_pairs_pseudo_replicate",
            config.min_correlation_pairs,
            "pairs",
        ),
    ]
    for name in RUN_VALUES:
        checks.append(
            (
                f"run lengths ({name})",
                f"run_length_{name}_wasserstein",
                f"n_run_length_{name}_matched_min",
                config.min_distribution_size,
                "runs",
            )
        )
    return checks


def drop_summary(
    metrics: pd.DataFrame,
    config: MetricsConfig = MetricsConfig(),
) -> pd.DataFrame:
    """Why regions, sites and pairs are missing from the metrics.

    Each region row counts, per metric group, the regions scored, the regions
    whose statistic is NaN because it rests on fewer than the configured
    floor, the regions where it is NaN anyway (for example a constant profile,
    whose correlation is undefined), and the regions skipped before scoring
    because they share no predicted and observed positions. Fractions are over
    every region in the samples file. Site and pair rows total what the
    profile and covariation statistics could have used and what they did.
    """

    skipped = len(metrics.attrs.get("skipped_regions", []))
    total = len(metrics) + skipped
    table = metrics.copy()
    for name in RUN_VALUES:
        # A run-length distance needs enough runs in both of its samples.
        model = f"n_run_length_{name}_matched_model"
        observed = f"n_run_length_{name}_matched_observed"
        if model in table and observed in table:
            table[f"n_run_length_{name}_matched_min"] = np.minimum(table[model], table[observed])

    rows = []

    def add(metric, unit, reason, count, denominator, detail=""):
        rows.append(
            {
                "metric": metric,
                "unit": unit,
                "reason": reason,
                "detail": detail,
                "count": int(count),
                "fraction": count / denominator if denominator else np.nan,
            }
        )

    for metric, column, count_column, floor, items in _drop_checks(config):
        if column not in table or count_column not in table:
            continue
        missing = table[column].isna()
        too_few = missing & (table[count_column] < floor)
        add(metric, "regions", "scored", (~missing).sum(), total)
        add(
            metric,
            "regions",
            "too few",
            too_few.sum(),
            total,
            detail=f"fewer than {floor} {items}",
        )
        add(
            metric,
            "regions",
            "undefined",
            (missing & ~too_few).sum(),
            total,
            detail=("enough data but NaN, e.g. a constant profile or reads that share no sites"),
        )
        add(metric, "regions", "no shared positions", skipped, total)

    if {"n_shared_positions", "n_valid_positions"} <= set(table):
        shared = table["n_shared_positions"].sum()
        add("profile", "sites", "shared by model and observed", shared, shared)
        if "n_low_coverage_positions" in table:
            add(
                "profile",
                "sites",
                "low coverage",
                table["n_low_coverage_positions"].sum(),
                shared,
                detail=f"fewer than {config.min_site_coverage} observed reads",
            )
        add("profile", "sites", "used", table["n_valid_positions"].sum(), shared)
        if "n_matched_correlation_sites" in table:
            add(
                "profile, matched reads",
                "sites",
                "used",
                table["n_matched_correlation_sites"].sum(),
                shared,
            )
    if {"n_shared_positions", "n_covariation_pairs"} <= set(table):
        shared = table["n_shared_positions"]
        pairs = (shared * (shared - 1) // 2).sum()
        used = table["n_covariation_pairs"].sum()
        add("covariation", "pairs", "between shared positions", pairs, pairs)
        add(
            "covariation",
            "pairs",
            "not usable",
            pairs - used,
            pairs,
            detail=(
                f"fewer than {config.covariation_min_joint_reads} joint reads "
                "in the model or observed sample, or undefined"
            ),
        )
        add("covariation", "pairs", "used", used, pairs)
    return pd.DataFrame.from_records(rows)


def write_drop_summary(
    metrics: pd.DataFrame,
    output_dir: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
) -> pd.DataFrame:
    """Write ``metrics_drops.tsv`` and, if any, ``skipped_regions.txt``."""

    output_dir = Path(output_dir)
    summary = drop_summary(metrics, config)
    summary.to_csv(output_dir / "metrics_drops.tsv", sep="\t", index=False)
    skipped = metrics.attrs.get("skipped_regions", [])
    if skipped:
        (output_dir / "skipped_regions.txt").write_text("\n".join(skipped) + "\n")
    return summary


def array_columns(columns: pd.Index | list[str]) -> list[str]:
    """Columns holding one array per region rather than one scalar."""

    return (
        histogram_columns(columns)
        + decay_columns(columns)
        + [column for column in columns if column.startswith(SITE_PREFIX)]
    )


def write_metrics_tsv(
    samples_path: str | PathLike[str],
    output_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
    baselines: PositionalBaselines | None = None,
    flank_baseline: FlankBaseline | None = None,
) -> pd.DataFrame:
    """Write scalar metrics as a TSV and return the full table.

    Histogram, decay, and site-mean arrays are excluded from the TSV. They
    remain in the returned table for the histogram, decay, and pooled site
    correlation reporting helpers.
    """

    metrics = metrics_from_h5(samples_path, config, baselines, flank_baseline)
    arrays = array_columns(metrics.columns)
    metrics.drop(columns=arrays).to_csv(output_path, sep="\t", index=False)
    return metrics


def write_gap_closed_tsv(
    metrics: pd.DataFrame,
    output_path: str | PathLike[str],
) -> pd.DataFrame:
    """Write one pooled gap-closed row per metric and return the summary.

    This is an aggregate over regions rather than a per-region column because a
    single region's model, baseline and floor are all within sampling noise of
    each other, so their per-region ratio has no stable value.
    """

    summary = gap_closed_summary(metrics)
    summary.to_csv(output_path, sep="\t", index=False)
    return summary


def write_site_correlations_tsv(
    metrics: pd.DataFrame,
    output_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
) -> pd.DataFrame:
    """Write pooled site correlations and their contributing site counts."""

    summary = site_correlation_summary(metrics, config.min_correlation_sites)
    summary.to_csv(output_path, sep="\t", index=False)
    return summary


def _write_region_arrays(
    metrics: pd.DataFrame,
    output_path: str | PathLike[str],
    columns: list[str],
    bin_edges: np.ndarray,
) -> list[str]:
    """Stack per-region arrays onto a ``[region, bin]`` grid in one ``.npz``.

    Rows are aligned with ``region_ids``, so pooling over regions is a sum along
    the region axis.
    """

    if not columns:
        return []
    np.savez_compressed(
        output_path,
        bin_edges=bin_edges,
        region_ids=metrics["region"].to_numpy(dtype=str),
        **{
            column: np.stack([np.asarray(value) for value in metrics[column]]) for column in columns
        },
    )
    return columns


def write_run_length_histograms(
    metrics: pd.DataFrame,
    output_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
) -> list[str]:
    """Write per-region run-length histograms and their bin edges to an ``.npz``."""

    return _write_region_arrays(
        metrics,
        output_path,
        histogram_columns(metrics.columns),
        config.run_length_bins.edges,
    )


def write_covariation_decay(
    metrics: pd.DataFrame,
    output_path: str | PathLike[str],
    config: MetricsConfig = MetricsConfig(),
) -> list[str]:
    """Write per-region correlation decay sums and counts to an ``.npz``.

    Sums and counts are stored separately rather than as a mean so that the
    pooled curve is the summed numerator over the summed denominator.
    """

    return _write_region_arrays(
        metrics,
        output_path,
        decay_columns(metrics.columns),
        config.separation_bins.edges,
    )
