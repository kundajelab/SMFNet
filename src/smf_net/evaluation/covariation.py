"""Pairwise position-position covariation for observed and generated reads.

A read's accessibility calls are not independent across positions, and that
coupling is what a per-read generative model has to earn beyond matching the
mean profile. Two comparisons live here.

:func:`correlation_decay` bins each pair of positions by how far apart they sit
and averages the correlations within a bin, which pools across regions into a
curve of how co-accessibility falls off with genomic distance.

:func:`correlation_of_correlations` keeps every pair in place instead and asks
how well the model's pairwise correlations track the observed ones inside one
region, which is a single number per region.

Both work on the matched read samples rather than on every model read. A
correlation estimated from ten reads carries roughly +/-0.3 of sampling noise,
so the only readable version of either number is its distance from the observed
pseudo-replicate reference computed at the same read count. The exceptions are
the all-read and positional baseline references of
:func:`covariation_metrics`, which carry no noise of their own and are scored
against every observed read rather than one half.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from smf_net.evaluation.correlation import pearson, spearman

DECAY_PREFIX = "covariation_decay_"


@dataclass(frozen=True)
class SeparationBins:
    """Genomic separation bins for the correlation decay curve.

    Pairs separated by more than ``max_bp`` are dropped rather than gathered
    into a final bin. A decay curve is read as a function of distance, so an
    overflow bin would plot a separation that no pair in it actually has.
    """

    max_bp: int = 500
    bin_width_bp: int = 10

    def __post_init__(self) -> None:
        if self.bin_width_bp < 1:
            raise ValueError("bin_width_bp must be positive")
        if self.max_bp < self.bin_width_bp:
            raise ValueError("max_bp cannot be smaller than bin_width_bp")
        if self.max_bp % self.bin_width_bp:
            raise ValueError("max_bp must be a multiple of bin_width_bp")

    @property
    def edges(self) -> np.ndarray:
        return np.arange(0, self.max_bp + self.bin_width_bp, self.bin_width_bp)

    @property
    def centers(self) -> np.ndarray:
        return self.edges[:-1] + self.bin_width_bp / 2

    @property
    def num_bins(self) -> int:
        return self.max_bp // self.bin_width_bp


@dataclass(frozen=True)
class CovariationShuffles:
    """Permuted observed read samples that earn a pooled decay curve.

    The two permutations bound different things. Within-read shuffling moves
    each read's calls across positions, destroying the profile along with the
    coupling, but it keeps each read's overall accessibility, so read-to-read
    heterogeneity lifts every pair's correlation by a similar amount. Read-axis
    shuffling permutes calls across reads at each position, which leaves every
    position marginal exactly intact and removes only the coupling between
    positions, so its curve sits at zero up to sampling noise.

    Neither earns a correlation-of-correlations column. One leaves the pair
    correlations pure noise and the other offsets them by a near constant that
    a correlation cannot see, so both baselines are zero by construction.
    """

    within_read: torch.Tensor | None = None
    read_axis: torch.Tensor | None = None


# Field name and the decay-curve suffix each shuffle is written under.
_SHUFFLE_COLUMNS = (
    ("within_read", "shuffle"),
    ("read_axis", "read_axis_shuffle"),
)


@dataclass(frozen=True)
class PairIndex:
    """Upper-triangle pair layout shared by every read sample in a region.

    The pairs and their separations depend only on the region's positions, so
    they are built once and reused for the model, the observed sample, and each
    permuted baseline.
    """

    rows: torch.Tensor
    cols: torch.Tensor
    separation: torch.Tensor

    @classmethod
    def from_positions(cls, positions: torch.Tensor) -> "PairIndex":
        rows, cols = torch.triu_indices(
            positions.numel(),
            positions.numel(),
            offset=1,
        )
        return cls(
            rows=rows,
            cols=cols,
            separation=(positions[cols] - positions[rows]).abs(),
        )

    @property
    def num_pairs(self) -> int:
        return self.rows.numel()


@dataclass(frozen=True)
class PairCorrelations:
    """One read sample's correlation and read count for every position pair."""

    correlation: torch.Tensor
    joint_reads: torch.Tensor

    def usable(self, min_joint_reads: int) -> torch.Tensor:
        """Pairs with a defined correlation and enough reads behind it.

        The threshold only removes pairs too degenerate to have a value; it
        cannot remove sampling noise, which is what pooling and the replicate
        reference are for.
        """

        return torch.isfinite(self.correlation) & (self.joint_reads >= min_joint_reads)


def pairwise_correlation(reads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise-complete correlation between every pair of positions.

    Each pair uses only the reads carrying a call at both of its positions, so
    the number of contributing reads varies by pair and is returned alongside
    the correlations. A pair with no shared reads, or where either position is
    constant across them, is NaN.

    Returns ``(correlation, joint_reads)``, both ``[positions, positions]``.
    """

    valid = (reads != -1).float()
    calls = torch.where(reads == -1, 0, reads).float()

    joint_reads = valid.mT @ valid
    counts = joint_reads.clamp_min(1)
    # Calls are binary, so x**2 == x and the conditional mean doubles as the
    # second moment. One masked product per moment is therefore enough.
    mean_x = (calls.mT @ valid) / counts
    mean_y = mean_x.mT
    covariance = (calls.mT @ calls) / counts - mean_x * mean_y
    scale = ((mean_x - mean_x.square()) * (mean_y - mean_y.square())).sqrt()

    defined = (joint_reads > 0) & (scale > 0)
    correlation = torch.where(
        defined,
        covariance / scale.clamp_min(torch.finfo(scale.dtype).tiny),
        torch.nan,
    )
    return correlation, joint_reads


def pair_correlations(reads: torch.Tensor, index: PairIndex) -> PairCorrelations:
    """Flatten a sample's correlation matrix onto the shared pair layout."""

    correlation, joint_reads = pairwise_correlation(reads)
    return PairCorrelations(
        correlation=correlation[index.rows, index.cols],
        joint_reads=joint_reads[index.rows, index.cols],
    )


def correlation_decay(
    pairs: PairCorrelations,
    index: PairIndex,
    bins: SeparationBins,
    min_joint_reads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of pair correlations and pair count per separation bin.

    Returning sums and counts rather than means is what lets regions pool: the
    pooled curve is the summed numerator over the summed denominator, so a
    region contributes in proportion to how many usable pairs it has.
    """

    keep = (
        pairs.usable(min_joint_reads) & (index.separation > 0) & (index.separation <= bins.max_bp)
    )
    sums = torch.zeros(bins.num_bins)
    counts = torch.zeros(bins.num_bins)
    if keep.any():
        # Separations equal to an edge fall in the upper bin, and ``max_bp``
        # itself falls in the last one, matching ``numpy.histogram``.
        bin_index = torch.div(
            index.separation[keep],
            bins.bin_width_bp,
            rounding_mode="floor",
        ).clamp_max(bins.num_bins - 1)
        sums.scatter_add_(0, bin_index, pairs.correlation[keep].float())
        counts.scatter_add_(0, bin_index, torch.ones_like(bin_index, dtype=counts.dtype))
    return sums.numpy(), counts.numpy().astype(np.int64)


def correlation_of_correlations(
    model: PairCorrelations,
    observed: PairCorrelations,
    min_joint_reads: int,
    min_pairs: int = 0,
) -> tuple[float, float, int]:
    """Agreement between two samples' pairwise correlations in one region.

    Only pairs usable in both samples contribute, so the two vectors describe
    the same pairs. Returns ``(pearson, spearman, n_pairs)``; both correlations
    are NaN when fewer than ``min_pairs`` pairs remain, because a correlation
    over a handful of points is pushed toward +/-1 rather than just widened.

    This deliberately reports no regression slope. Sampling noise in the
    observed correlations attenuates any slope fitted against them toward zero,
    so a naive slope reads as model under-dispersion even for a perfect model.
    """

    keep = model.usable(min_joint_reads) & observed.usable(min_joint_reads)
    n_pairs = int(keep.sum())
    if n_pairs < min_pairs:
        return float("nan"), float("nan"), n_pairs
    x = model.correlation[keep]
    y = observed.correlation[keep]
    return pearson(x, y), spearman(x, y), n_pairs


def _residual(pairs: PairCorrelations, baseline: torch.Tensor) -> PairCorrelations:
    """Pair correlations with the baseline's prediction removed.

    A pair the baseline never saw is NaN here, so it drops out of the residual
    agreement like any other unusable pair.
    """

    return PairCorrelations(
        correlation=pairs.correlation - baseline.to(pairs.correlation.dtype),
        joint_reads=pairs.joint_reads,
    )


def covariation_metrics(
    positions: torch.Tensor,
    matched_predicted: torch.Tensor,
    matched_observed: torch.Tensor,
    matched_replicate: torch.Tensor,
    shuffles: CovariationShuffles,
    bins: SeparationBins,
    min_joint_reads: int,
    all_predicted: torch.Tensor,
    all_observed: torch.Tensor,
    pair_baseline: torch.Tensor | None = None,
    min_pairs: int = 0,
) -> dict[str, float | int | np.ndarray]:
    """Compare predicted and observed pairwise covariation for one region.

    The matched samples all hold the same number of reads, so the
    pseudo-replicate column is a like-for-like reference: it is the agreement
    two halves of the real data reach with each other, which is the most a
    model scored from that many of its own reads can reach.

    Two references sit off that footing on purpose. ``all_predicted`` scores
    the model from every one of its reads, which carry almost no sampling
    noise, and ``pair_baseline`` is a noise-free sequence-blind prediction for
    every pair. Neither has a replicate to be matched to, so both are scored
    against ``all_observed``, every observed read, which leaves the least
    noise on the observed side. With ``R`` the replicate agreement and ``k``
    the ratio of observed reads to matched reads, a noise-free predictor can
    reach about ``sqrt(k R / (1 + (k - 1) R))``. The baseline is also
    subtracted from every sample's pair correlations for the
    ``covariation_residual_*`` columns. Those keep only region-specific
    coupling, which the baseline scores zero on by construction, so they have
    a replicate floor but no gap to close.
    """

    index = PairIndex.from_positions(positions)
    observed_pairs = pair_correlations(matched_observed, index)
    model_pairs = pair_correlations(matched_predicted, index)

    model_pearson, model_spearman, n_pairs = correlation_of_correlations(
        model_pairs,
        observed_pairs,
        min_joint_reads,
        min_pairs,
    )
    record: dict[str, float | int | np.ndarray] = {
        "n_covariation_pairs": n_pairs,
        "covariation_corr_of_corr_pearson": model_pearson,
        "covariation_corr_of_corr_spearman": model_spearman,
    }
    for suffix, sample in (
        ("model", model_pairs),
        ("observed", observed_pairs),
    ):
        sums, counts = correlation_decay(sample, index, bins, min_joint_reads)
        record[f"{DECAY_PREFIX}sum_{suffix}"] = sums
        record[f"{DECAY_PREFIX}count_{suffix}"] = counts

    replicate_pairs = pair_correlations(matched_replicate, index)
    floor_pearson, floor_spearman, n_replicate_pairs = correlation_of_correlations(
        replicate_pairs,
        observed_pairs,
        min_joint_reads,
        min_pairs,
    )
    record["covariation_corr_of_corr_pearson_pseudo_replicate"] = floor_pearson
    record["covariation_corr_of_corr_spearman_pseudo_replicate"] = floor_spearman
    record["n_covariation_pairs_pseudo_replicate"] = n_replicate_pairs

    # Suffix, pair correlations and observed reference of every sample.
    samples = [
        ("", model_pairs, observed_pairs),
        ("_pseudo_replicate", replicate_pairs, observed_pairs),
    ]
    all_observed_pairs = pair_correlations(all_observed, index)
    all_pairs = pair_correlations(all_predicted, index)
    samples.append(("_all_reads", all_pairs, all_observed_pairs))
    all_pearson, all_spearman, n_all_pairs = correlation_of_correlations(
        all_pairs,
        all_observed_pairs,
        min_joint_reads,
        min_pairs,
    )
    record["covariation_corr_of_corr_pearson_all_reads"] = all_pearson
    record["covariation_corr_of_corr_spearman_all_reads"] = all_spearman
    record["n_covariation_pairs_all_reads"] = n_all_pairs

    if pair_baseline is not None:
        baseline_pairs = PairCorrelations(
            correlation=pair_baseline,
            joint_reads=torch.full_like(pair_baseline, torch.inf),
        )
        baseline_pearson, baseline_spearman, _ = correlation_of_correlations(
            baseline_pairs,
            all_observed_pairs,
            min_joint_reads,
            min_pairs,
        )
        record["covariation_corr_of_corr_pearson_positional_baseline"] = baseline_pearson
        record["covariation_corr_of_corr_spearman_positional_baseline"] = baseline_spearman

        for suffix, pairs, reference in samples:
            residual_pearson, residual_spearman, _ = correlation_of_correlations(
                _residual(pairs, pair_baseline),
                _residual(reference, pair_baseline),
                min_joint_reads,
                min_pairs,
            )
            record[f"covariation_residual_corr_of_corr_pearson{suffix}"] = residual_pearson
            record[f"covariation_residual_corr_of_corr_spearman{suffix}"] = residual_spearman

    for field, suffix in _SHUFFLE_COLUMNS:
        reads = getattr(shuffles, field)
        if reads is None:
            continue
        sums, counts = correlation_decay(
            pair_correlations(reads, index),
            index,
            bins,
            min_joint_reads,
        )
        record[f"{DECAY_PREFIX}sum_{suffix}"] = sums
        record[f"{DECAY_PREFIX}count_{suffix}"] = counts
    return record


def decay_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column.startswith(DECAY_PREFIX)]


def pooled_decay(table: pd.DataFrame, suffix: str) -> np.ndarray:
    """Mean pair correlation per separation bin, pooled over every region.

    Bins with no usable pair in any region are NaN so the curve breaks rather
    than dropping to zero where there is no data.
    """

    sums = np.stack([np.asarray(value) for value in table[f"{DECAY_PREFIX}sum_{suffix}"]])
    counts = np.stack([np.asarray(value) for value in table[f"{DECAY_PREFIX}count_{suffix}"]])
    total_counts = counts.sum(axis=0)
    return np.where(
        total_counts > 0,
        sums.sum(axis=0) / np.maximum(total_counts, 1),
        np.nan,
    )
