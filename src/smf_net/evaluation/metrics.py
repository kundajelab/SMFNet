"""Metrics for stored predicted and observed SMF samples."""

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from smf_net.data import site_means
from smf_net.evaluation.correlation import pearson as _pearson
from smf_net.evaluation.correlation import spearman as _spearman
from smf_net.evaluation.covariation import (
    CovariationShuffles,
    PairIndex,
    SeparationBins,
    covariation_metrics,
)
from smf_net.evaluation.flank_baselines import FlankBaseline, FlankBaselineSettings
from smf_net.evaluation.positional_baselines import (
    PositionalBaselines,
    PositionalBaselineSettings,
)
from smf_net.evaluation.run_lengths import (
    RunLengthBins,
    RunLengthShuffles,
    donate_missingness,
    run_length_metrics,
)
from smf_net.evaluation.sample_io import StoredSampleRegion
from smf_net.evaluation.site_correlations import site_mean_pairs

MetricRecord = dict[str, str | int | float | np.ndarray]


@dataclass(frozen=True)
class MetricsConfig:
    """Controls which per-region metrics and reference points are computed.

    ``include_shuffle`` adds ``*_shuffle`` baselines built from permuted
    observed reads. They describe the data, so every model shares them and they
    are what the plots draw as reference lines.

    ``include_positional_baselines`` fits sequence-blind train-set baselines
    for the profile and pair correlations (see
    :mod:`smf_net.evaluation.positional_baselines`) and adds
    ``*_positional_baseline`` and residual covariation columns. The fit needs
    the training data behind the samples file, so it happens in
    :func:`smf_net.evaluation.reporting.train_baselines_for`; the
    ``positional_baseline_*`` fields set its smoothing and subsampling.

    ``include_flank_baselines`` fits a train-set model of each site's mean call
    from its flanking bases (see :mod:`smf_net.evaluation.flank_baselines`) and
    adds ``*_flank_baseline`` profile correlations, plus ``*_flank_residual``
    correlations of model and observed means after subtracting it, which only
    learning more than local context can earn. It is fit from the same
    training data as the positional baselines; the ``flank_baseline_*`` fields
    set its context width, regularization and subsampling.

    ``min_site_coverage`` drops sites that too few observed reads cover from
    every profile correlation. Short reads cover sites very unevenly, and a
    site seen by a handful of reads has a near-binary mean that dominates an
    unweighted correlation. Only observed coverage decides, so the model and
    every reference are scored on the same sites. Covariation has its own
    threshold, ``covariation_min_joint_reads``.

    The ``min_correlation_*`` and ``min_distribution_size`` floors make a
    statistic NaN rather than report it from too little data: a profile
    correlation needs that many sites, a correlation of correlations that many
    pairs, and a run-length distance that many runs in each of its two
    samples. A correlation over a handful of points is pushed toward
    +/-1, not just made noisier, so these regions would bias the pooled medians.
    """

    include_shuffle: bool = True
    seed: int = 0
    progress: bool = True
    num_workers: int = 0
    include_run_lengths: bool = True
    run_length_max_bp: int = 400
    run_length_bin_width_bp: int = 10
    include_covariation: bool = True
    covariation_max_separation_bp: int = 500
    covariation_separation_bin_bp: int = 10
    covariation_min_joint_reads: int = 10
    min_site_coverage: int = 10
    min_correlation_sites: int = 10
    min_correlation_pairs: int = 10
    min_distribution_size: int = 10
    include_positional_baselines: bool = True
    positional_baseline_profile_bandwidth_bp: float = 50.0
    positional_baseline_separation_bandwidth_bp: float = 2.0
    positional_baseline_midpoint_bandwidth_bp: float = 50.0
    positional_baseline_max_pair_regions: int | None = 10_000
    include_flank_baselines: bool = True
    flank_baseline_bp: int = 5
    flank_baseline_l2: float = 1e-4
    flank_baseline_max_sites: int | None = 2_000_000

    @property
    def flank_baseline_settings(self) -> FlankBaselineSettings:
        return FlankBaselineSettings(
            flank_bp=self.flank_baseline_bp,
            l2=self.flank_baseline_l2,
            max_sites=self.flank_baseline_max_sites,
            seed=self.seed,
        )

    @property
    def positional_baseline_settings(self) -> PositionalBaselineSettings:
        return PositionalBaselineSettings(
            profile_bandwidth_bp=self.positional_baseline_profile_bandwidth_bp,
            separation_bandwidth_bp=self.positional_baseline_separation_bandwidth_bp,
            midpoint_bandwidth_bp=self.positional_baseline_midpoint_bandwidth_bp,
            max_pair_regions=self.positional_baseline_max_pair_regions,
            min_joint_reads=self.covariation_min_joint_reads,
            seed=self.seed,
        )

    @property
    def run_length_bins(self) -> RunLengthBins:
        return RunLengthBins(
            max_bp=self.run_length_max_bp,
            bin_width_bp=self.run_length_bin_width_bp,
        )

    @property
    def separation_bins(self) -> SeparationBins:
        return SeparationBins(
            max_bp=self.covariation_max_separation_bp,
            bin_width_bp=self.covariation_separation_bin_bp,
        )


class UnscorableRegionError(ValueError):
    """A region with nothing for its samples to be compared on.

    A sampling window can hold none of a region's observed sites even though
    the region has observed reads elsewhere. Such a region is skipped rather
    than failing the whole table, unlike structural problems such as samples
    without any observed reads.
    """


@dataclass(frozen=True)
class AlignedSampleRegion:
    region_id: str
    positions: torch.Tensor
    predicted_smf: torch.Tensor
    observed_smf: torch.Tensor
    observed_mean: torch.Tensor


def align_sample_region(region: StoredSampleRegion) -> AlignedSampleRegion:
    """Restrict predicted and observed samples to their shared coordinates."""

    if (
        region.observed_smf is None
        or region.observed_positions is None
        or region.observed_mean is None
    ):
        raise ValueError(
            f"Region {region.region_id!r} has no observed reads; "
            "metrics require samples generated from a data split"
        )

    positions, predicted_idx, observed_idx = np.intersect1d(
        region.predicted_positions.numpy(),
        region.observed_positions.numpy(),
        return_indices=True,
    )
    if not len(positions):
        raise UnscorableRegionError(
            f"Region {region.region_id!r} has no shared predicted and observed positions"
        )
    predicted_idx = torch.from_numpy(predicted_idx)
    observed_idx = torch.from_numpy(observed_idx)
    return AlignedSampleRegion(
        region_id=region.region_id,
        positions=torch.from_numpy(positions),
        predicted_smf=region.predicted_smf[:, predicted_idx],
        observed_smf=region.observed_smf[:, observed_idx],
        observed_mean=region.observed_mean[observed_idx],
    )


def _sample_mean(reads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-site means and which sites any read calls."""

    mean, counts = site_means(reads)
    return mean, counts > 0


def _shuffle_reads(
    reads: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle valid calls across positions independently within each read."""

    shuffled = reads.clone()
    for read, shuffled_read in zip(reads, shuffled, strict=True):
        valid = read != -1
        values = read[valid]
        order = torch.randperm(values.numel(), generator=generator)
        shuffled_read[valid] = values[order]
    return shuffled


def _shuffle_along_read_axis(
    reads: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle valid calls across reads independently at each position."""

    shuffled = reads.clone()
    for position, shuffled_position in zip(
        reads.mT,
        shuffled.mT,
        strict=True,
    ):
        valid = position != -1
        values = position[valid]
        order = torch.randperm(values.numel(), generator=generator)
        shuffled_position[valid] = values[order]
    return shuffled


def _prepare_matched_samples(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create matched model, observed, and observed pseudo-replicate samples."""

    reads_per_sample = min(predicted.shape[0], observed.shape[0] // 2)

    predicted_order = torch.randperm(predicted.shape[0], generator=generator)
    observed_order = torch.randperm(observed.shape[0], generator=generator)
    predicted_sample = predicted[predicted_order[:reads_per_sample]]
    observed_sample = observed[observed_order[:reads_per_sample]]
    observed_replicate = observed[observed_order[reads_per_sample : 2 * reads_per_sample]]
    return predicted_sample, observed_sample, observed_replicate


def _covered(reads: torch.Tensor, min_coverage: int) -> torch.Tensor:
    """Sites that at least ``min_coverage`` of these reads have a call at."""

    return (reads != -1).sum(dim=0) >= min_coverage


def _correlations(
    predicted: torch.Tensor,
    observed_mean: torch.Tensor,
    sites: torch.Tensor | None = None,
    min_sites: int = 0,
) -> tuple[float, float, int]:
    predicted_mean, has_prediction = _sample_mean(predicted)
    valid = has_prediction & (observed_mean != -1)
    if sites is not None:
        valid &= sites
    n_valid = int(valid.sum())
    if n_valid < min_sites:
        return float("nan"), float("nan"), n_valid
    x = predicted_mean[valid]
    y = observed_mean[valid]
    return _pearson(x, y), _spearman(x, y), n_valid


def _matched_correlations(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    replicate: torch.Tensor,
    shuffled_replicate: torch.Tensor | None,
    min_site_coverage: int = 1,
    min_sites: int = 0,
    flank_prediction: torch.Tensor | None = None,
) -> MetricRecord:
    """Profile correlations scored against one observed half at one read count.

    The headline correlations score every model read against the mean of every
    observed read, and no observed reference can be scored that way without
    reusing its own target. Here the model sample, the replicate and its
    shuffle each hold as many reads as the observed half they are scored
    against and share none of its reads, so the replicate is the agreement a
    perfect model reaches in expectation.

    A site counts only if both observed halves cover it well enough. The
    within-read shuffle keeps every read's missingness, so it covers the same
    sites as the replicate it was built from.

    ``flank_prediction`` adds ``*_flank_residual_matched*`` columns, the same
    comparisons after subtracting the flank baseline, so the residual has a
    replicate ceiling too.
    """

    target_mean, has_target = _sample_mean(observed)
    target_mean = torch.where(has_target, target_mean, -1.0)
    sites = _covered(observed, min_site_coverage) & _covered(replicate, min_site_coverage)
    samples = {"": predicted, "_pseudo_replicate": replicate}
    if shuffled_replicate is not None:
        samples["_shuffle"] = shuffled_replicate

    record: MetricRecord = {
        "n_matched_correlation_sites": int((sites & has_target).sum()),
    }
    for suffix, reads in samples.items():
        pearson, spearman, _ = _correlations(reads, target_mean, sites, min_sites)
        record[f"pearson_matched{suffix}"] = pearson
        record[f"spearman_matched{suffix}"] = spearman
        if flank_prediction is not None:
            pearson, spearman = _residual_correlations(
                reads, target_mean, flank_prediction, sites, min_sites
            )
            record[f"pearson_flank_residual_matched{suffix}"] = pearson
            record[f"spearman_flank_residual_matched{suffix}"] = spearman
    return record


def _positional_baseline_correlations(
    baselines: PositionalBaselines,
    positions: torch.Tensor,
    observed_mean: torch.Tensor,
    sites: torch.Tensor,
    min_sites: int = 0,
) -> MetricRecord:
    """Train-set mean profile scored like the headline correlations."""

    baseline = baselines.profile_at(positions)
    valid = (observed_mean != -1) & torch.isfinite(baseline) & sites
    if int(valid.sum()) < min_sites:
        return {
            "pearson_positional_baseline": float("nan"),
            "spearman_positional_baseline": float("nan"),
        }
    x = baseline[valid]
    y = observed_mean[valid]
    return {
        "pearson_positional_baseline": _pearson(x, y),
        "spearman_positional_baseline": _spearman(x, y),
    }


def _residual_correlations(
    predicted: torch.Tensor,
    observed_mean: torch.Tensor,
    baseline: torch.Tensor,
    sites: torch.Tensor,
    min_sites: int = 0,
) -> tuple[float, float]:
    """Correlate predicted and observed means after subtracting ``baseline``."""

    predicted_mean, has_prediction = _sample_mean(predicted)
    valid = has_prediction & (observed_mean != -1) & torch.isfinite(baseline) & sites
    if int(valid.sum()) < min_sites:
        return float("nan"), float("nan")
    x = predicted_mean[valid] - baseline[valid]
    y = observed_mean[valid] - baseline[valid]
    return _pearson(x, y), _spearman(x, y)


def _flank_baseline_correlations(
    baseline: torch.Tensor,
    predicted: torch.Tensor,
    observed_mean: torch.Tensor,
    sites: torch.Tensor,
    min_sites: int = 0,
) -> MetricRecord:
    """Flank-model profile, and model agreement beyond it.

    ``baseline`` is the flank prediction at each site. The residual columns
    correlate the model's and the observed means after subtracting it from
    both, on the sites the headline correlations use.
    """

    valid = (observed_mean != -1) & torch.isfinite(baseline) & sites
    record: MetricRecord = {
        "pearson_flank_baseline": float("nan"),
        "spearman_flank_baseline": float("nan"),
    }
    if int(valid.sum()) >= min_sites:
        record["pearson_flank_baseline"] = _pearson(baseline[valid], observed_mean[valid])
        record["spearman_flank_baseline"] = _spearman(baseline[valid], observed_mean[valid])
    (
        record["pearson_flank_residual"],
        record["spearman_flank_residual"],
    ) = _residual_correlations(predicted, observed_mean, baseline, sites, min_sites)
    return record


def _check_baseline_assay(region: StoredSampleRegion, kind: str, baseline_assay: str) -> None:
    if region.assay_type is not None and region.assay_type != baseline_assay:
        raise ValueError(
            f"Region {region.region_id!r} has assay {region.assay_type!r} but the "
            f"{kind} baselines were fit on {baseline_assay!r}"
        )


def compute_region_metrics(
    region: StoredSampleRegion,
    config: MetricsConfig,
    generator: torch.Generator,
    baselines: PositionalBaselines | None = None,
    flank_baseline: FlankBaseline | None = None,
) -> MetricRecord:
    """Score one region.

    ``baselines`` adds the positional baseline columns. Baselines are indexed
    by position in the full source region, so the region's sites are shifted by
    its ``window_start`` before every lookup. ``flank_baseline`` adds the flank
    columns, which read flanks from the region's stored window DNA.
    """

    aligned = align_sample_region(region)
    if baselines is not None:
        _check_baseline_assay(region, "positional", baselines.assay_type)
    if flank_baseline is not None:
        _check_baseline_assay(region, "flank", flank_baseline.assay_type)
    region_positions = aligned.positions + region.window_start
    covered_sites = _covered(aligned.observed_smf, config.min_site_coverage)
    pearson, spearman, n_valid = _correlations(
        aligned.predicted_smf,
        aligned.observed_mean,
        covered_sites,
        config.min_correlation_sites,
    )
    record: MetricRecord = {
        "region": aligned.region_id,
        "assay_type": region.assay_type or "",
        "source_type": region.source.kind if region.source is not None else "",
        "source_path": region.source.path if region.source is not None else "",
        "reference_fasta_path": (
            region.source.reference_fasta_path or "" if region.source is not None else ""
        ),
        "n_predicted_reads": aligned.predicted_smf.shape[0],
        "n_observed_reads": aligned.observed_smf.shape[0],
        "n_predicted_positions": region.predicted_positions.numel(),
        "n_observed_positions": region.observed_positions.numel(),
        "n_shared_positions": aligned.positions.numel(),
        "n_low_coverage_positions": int(((aligned.observed_mean != -1) & ~covered_sites).sum()),
        "n_valid_positions": n_valid,
        "pearson": pearson,
        "spearman": spearman,
    }
    if baselines is not None:
        record.update(
            _positional_baseline_correlations(
                baselines,
                region_positions,
                aligned.observed_mean,
                covered_sites,
                config.min_correlation_sites,
            )
        )
    flank_prediction = (
        flank_baseline.predict(region.dna, aligned.positions)
        if flank_baseline is not None
        else None
    )
    if flank_prediction is not None:
        record.update(
            _flank_baseline_correlations(
                flank_prediction,
                aligned.predicted_smf,
                aligned.observed_mean,
                covered_sites,
                config.min_correlation_sites,
            )
        )

    # One matched split serves every comparison with a pseudo-replicate: the
    # model sample, the observed half it is scored against and the held-out
    # replicate all hold the same number of reads, so the replicate is the
    # agreement a perfect model reaches at that read count. Comparisons
    # without a replicate use every read on both sides instead.
    predicted_matched, observed_matched, replicate_matched = _prepare_matched_samples(
        aligned.predicted_smf,
        aligned.observed_smf,
        generator,
    )
    record["n_matched_reads"] = predicted_matched.shape[0]

    # Shuffling observed rather than model reads makes every baseline a property
    # of the data instead of the model, so the same reference line applies to
    # every model and a region where the baseline already matches the observed
    # sample is a region with no positional structure to learn. The shuffle
    # donor is the held-out replicate: it stays disjoint from the observed
    # sample the baselines are scored against, and it carries the same read
    # count as the model's matched sample.
    shuffled_observed: torch.Tensor | None = None
    read_axis_shuffled_observed: torch.Tensor | None = None
    if config.include_shuffle:
        shuffled_observed = _shuffle_reads(replicate_matched, generator)
        read_axis_shuffled_observed = _shuffle_along_read_axis(
            replicate_matched,
            generator,
        )

    record.update(
        _matched_correlations(
            predicted_matched,
            observed_matched,
            replicate_matched,
            shuffled_observed,
            config.min_site_coverage,
            config.min_correlation_sites,
            flank_prediction,
        )
    )

    # Donating observed per-read missingness is what makes predicted and observed
    # reads comparable, so every predicted read set fed to the run length or
    # covariation metrics carries a donated pattern. The shuffled baselines need
    # no donor because they are built from observed reads and already carry their
    # real missingness.
    if config.include_run_lengths or config.include_covariation:
        donated = donate_missingness(
            aligned.predicted_smf,
            aligned.observed_smf,
            generator,
        )
        matched_donated = donate_missingness(
            predicted_matched,
            aligned.observed_smf,
            generator,
        )

    if config.include_run_lengths:
        record.update(
            run_length_metrics(
                positions=aligned.positions,
                predicted=donated,
                observed=aligned.observed_smf,
                matched_predicted=matched_donated,
                matched_observed=observed_matched,
                matched_replicate=replicate_matched,
                shuffles=RunLengthShuffles(
                    observed_within_read=shuffled_observed,
                    observed_read_axis=read_axis_shuffled_observed,
                ),
                bins=config.run_length_bins,
                min_runs=config.min_distribution_size,
            )
        )

    if config.include_covariation:
        record.update(
            covariation_metrics(
                positions=aligned.positions,
                matched_predicted=matched_donated,
                matched_observed=observed_matched,
                matched_replicate=replicate_matched,
                shuffles=CovariationShuffles(
                    within_read=shuffled_observed,
                    read_axis=read_axis_shuffled_observed,
                ),
                bins=config.separation_bins,
                min_joint_reads=config.covariation_min_joint_reads,
                min_pairs=config.min_correlation_pairs,
                # Every model read, with the same donated missingness the run
                # lengths use, scored against every observed read.
                all_predicted=donated,
                all_observed=aligned.observed_smf,
                pair_baseline=(
                    baselines.pairs_at(
                        region_positions,
                        PairIndex.from_positions(aligned.positions),
                    )
                    if baselines is not None
                    else None
                ),
            )
        )

    # Pool site means even when a region cannot support its own correlation.
    full_mean, has_full = _sample_mean(aligned.predicted_smf)
    full_predictions = {"model_all_reads": full_mean}
    matched_mean, has_matched = _sample_mean(predicted_matched)
    target_mean, has_target = _sample_mean(observed_matched)
    replicate_mean, has_replicate = _sample_mean(replicate_matched)
    matched_predictions = {
        "model_matched": matched_mean,
        "pseudo_replicate_matched": replicate_mean,
    }
    if flank_prediction is not None:
        full_predictions["flank_baseline_all_reads"] = flank_prediction
        matched_predictions["flank_baseline_matched"] = flank_prediction
    record.update(
        site_mean_pairs(aligned.observed_mean, full_predictions, covered_sites & has_full)
    )
    record.update(
        site_mean_pairs(
            target_mean,
            matched_predictions,
            has_matched
            & has_target
            & has_replicate
            & _covered(observed_matched, config.min_site_coverage)
            & _covered(replicate_matched, config.min_site_coverage),
        )
    )
    return record


def region_generator(seed: int, region_index: int) -> torch.Generator:
    """Create a deterministic generator independent of region scheduling."""

    region_seed = (seed + region_index) % (2**63)
    return torch.Generator().manual_seed(region_seed)


def compute_metrics_table(
    regions: Iterable[StoredSampleRegion],
    config: MetricsConfig = MetricsConfig(),
    total_regions: int | None = None,
    baselines: PositionalBaselines | None = None,
    flank_baseline: FlankBaseline | None = None,
) -> pd.DataFrame:
    """Compute one row of metrics per stored region.

    Regions with nothing to compare are left out of the table with a warning;
    see :class:`UnscorableRegionError`.
    """

    records = [
        score_region(
            region,
            config,
            region_generator(config.seed, region_index),
            baselines,
            flank_baseline,
        )
        for region_index, region in enumerate(
            tqdm(
                regions,
                total=total_regions,
                desc="computing metrics",
                unit="region",
                disable=not config.progress,
            )
        )
    ]
    return metrics_table(records)


def score_region(
    region: StoredSampleRegion,
    config: MetricsConfig,
    generator: torch.Generator,
    baselines: PositionalBaselines | None = None,
    flank_baseline: FlankBaseline | None = None,
) -> MetricRecord | str:
    """Region metrics, or the region's id when it cannot be scored."""

    try:
        return compute_region_metrics(region, config, generator, baselines, flank_baseline)
    except UnscorableRegionError:
        return region.region_id


def metrics_table(records: Sequence[MetricRecord | str]) -> pd.DataFrame:
    """Assemble scored regions into a table and report the skipped ones.

    Skipped regions have no row, so their ids travel with the table in
    ``attrs["skipped_regions"]`` for the drop summary.
    """

    skipped = [record for record in records if isinstance(record, str)]
    if skipped:
        shown = ", ".join(skipped[:10]) + (", ..." if len(skipped) > 10 else "")
        warnings.warn(
            f"Skipped {len(skipped)} of {len(records)} regions with no shared "
            f"predicted and observed positions: {shown}",
            stacklevel=2,
        )
    table = pd.DataFrame.from_records([record for record in records if not isinstance(record, str)])
    table.attrs["skipped_regions"] = skipped
    return table
