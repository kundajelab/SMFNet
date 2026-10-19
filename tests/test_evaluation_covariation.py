import numpy as np
import pandas as pd
import pytest
import torch

from smf_net.evaluation.covariation import (
    DECAY_PREFIX,
    CovariationShuffles,
    PairIndex,
    SeparationBins,
    correlation_decay,
    correlation_of_correlations,
    covariation_metrics,
    pair_correlations,
    pairwise_correlation,
    pooled_decay,
)
from smf_net.evaluation.metrics import MetricsConfig, compute_metrics_table
from smf_net.evaluation.sample_io import StoredSampleRegion


def test_pairwise_correlation_recovers_known_relationships():
    # Position 1 copies position 0, position 2 is its opposite, position 3 is
    # constant and so has no correlation with anything.
    reads = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
        ],
        dtype=torch.int8,
    )

    correlation, joint_reads = pairwise_correlation(reads)

    assert correlation[0, 1] == pytest.approx(1.0)
    assert correlation[0, 2] == pytest.approx(-1.0)
    assert torch.isnan(correlation[0, 3])
    assert joint_reads.tolist() == [[4.0] * 4] * 4


def test_pairwise_correlation_matches_numpy_on_complete_reads():
    generator = torch.Generator().manual_seed(4)
    reads = (torch.rand((64, 5), generator=generator) < 0.5).to(torch.int8)

    correlation, _ = pairwise_correlation(reads)

    expected = np.corrcoef(reads.numpy().astype(float), rowvar=False)
    np.testing.assert_allclose(correlation.numpy(), expected, atol=1e-5)


def test_pairwise_correlation_uses_only_jointly_observed_reads():
    # The third read would break the perfect agreement between positions 0 and
    # 1, but it has no call at position 1 so that pair never sees it.
    reads = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [1, -1],
            [0, 0],
        ],
        dtype=torch.int8,
    )

    correlation, joint_reads = pairwise_correlation(reads)

    assert correlation[0, 1] == pytest.approx(1.0)
    assert joint_reads[0, 1] == pytest.approx(3.0)
    assert joint_reads[0, 0] == pytest.approx(4.0)


def test_pairwise_correlation_is_nan_without_shared_reads():
    reads = torch.tensor([[0, -1], [-1, 1]], dtype=torch.int8)

    correlation, joint_reads = pairwise_correlation(reads)

    assert joint_reads[0, 1] == pytest.approx(0.0)
    assert torch.isnan(correlation[0, 1])


def test_pair_index_records_separations_in_base_pairs():
    index = PairIndex.from_positions(torch.tensor([0, 10, 40]))

    assert index.num_pairs == 3
    assert index.separation.tolist() == [10, 40, 30]


def _uniform_pairs(correlations: list[float], joint_reads: float = 10.0):
    from smf_net.evaluation.covariation import PairCorrelations

    return PairCorrelations(
        correlation=torch.tensor(correlations),
        joint_reads=torch.full((len(correlations),), joint_reads),
    )


def test_correlation_decay_bins_by_separation():
    index = PairIndex.from_positions(torch.tensor([0, 5, 25]))
    # Separations are 5, 25 and 20 bp for the three pairs in that order.
    pairs = _uniform_pairs([1.0, 0.25, 0.5])

    sums, counts = correlation_decay(
        pairs,
        index,
        SeparationBins(max_bp=30, bin_width_bp=10),
        min_joint_reads=1,
    )

    assert counts.tolist() == [1, 0, 2]
    assert sums.tolist() == pytest.approx([1.0, 0.0, 0.75])


def test_correlation_decay_drops_remote_and_unsupported_pairs():
    index = PairIndex.from_positions(torch.tensor([0, 10, 500]))
    # Pair separations: 10 bp, 500 bp, 490 bp. Only the first is within range.
    pairs = _uniform_pairs([0.5, 0.5, 0.5])

    _, in_range = correlation_decay(
        pairs,
        index,
        SeparationBins(max_bp=20, bin_width_bp=10),
        min_joint_reads=1,
    )
    _, unsupported = correlation_decay(
        pairs,
        index,
        SeparationBins(max_bp=20, bin_width_bp=10),
        min_joint_reads=11,
    )

    assert in_range.sum() == 1
    assert unsupported.sum() == 0


def test_correlation_decay_ignores_undefined_correlations():
    index = PairIndex.from_positions(torch.tensor([0, 10]))
    pairs = _uniform_pairs([float("nan")])

    sums, counts = correlation_decay(
        pairs,
        index,
        SeparationBins(max_bp=20, bin_width_bp=10),
        min_joint_reads=1,
    )

    assert counts.sum() == 0
    assert sums.sum() == pytest.approx(0.0)


def test_correlation_of_correlations_is_one_for_an_identical_sample():
    generator = torch.Generator().manual_seed(7)
    reads = (torch.rand((32, 6), generator=generator) < 0.5).to(torch.int8)
    index = PairIndex.from_positions(torch.arange(6) * 10)
    pairs = pair_correlations(reads, index)

    pearson, spearman, n_pairs = correlation_of_correlations(
        pairs,
        pairs,
        min_joint_reads=5,
    )

    assert pearson == pytest.approx(1.0)
    assert spearman == pytest.approx(1.0)
    assert n_pairs == 15


def test_correlation_of_correlations_counts_only_pairs_usable_in_both():
    model = _uniform_pairs([0.5, 0.5, 0.5], joint_reads=10.0)
    observed = _uniform_pairs([0.4, 0.6, 0.5], joint_reads=2.0)

    _, _, n_pairs = correlation_of_correlations(model, observed, min_joint_reads=5)

    assert n_pairs == 0


def _structured_reads(
    rows: int,
    positions: int,
    generator: torch.Generator,
    block: int = 4,
) -> torch.Tensor:
    """Reads whose calls are constant within blocks of adjacent positions."""

    blocks = torch.rand((rows, positions // block + 1), generator=generator) < 0.5
    return blocks.repeat_interleave(block, dim=-1)[:, :positions].to(torch.int8)


def test_read_axis_shuffling_removes_the_covariation_it_baselines():
    generator = torch.Generator().manual_seed(11)
    index = PairIndex.from_positions(torch.arange(16) * 10)
    reads = _structured_reads(200, 16, generator)
    shuffled = reads.clone()
    for column in shuffled.mT:
        column.copy_(column[torch.randperm(column.numel(), generator=generator)])

    observed = pair_correlations(reads, index)
    replicate = pair_correlations(
        _structured_reads(200, 16, generator),
        index,
    )
    honest, _, _ = correlation_of_correlations(replicate, observed, 5)
    null, _, _ = correlation_of_correlations(
        pair_correlations(shuffled, index),
        observed,
        5,
    )

    # A second draw from the same process tracks the observed correlations; the
    # read-axis shuffle keeps the marginals but carries none of the coupling.
    assert honest > 0.5
    assert abs(null) < 0.2


def test_pooled_decay_divides_summed_sums_by_summed_counts():
    table = pd.DataFrame(
        {
            f"{DECAY_PREFIX}sum_model": [np.array([1.0, 0.0]), np.array([2.0, 0.0])],
            f"{DECAY_PREFIX}count_model": [np.array([2, 0]), np.array([2, 0])],
        }
    )

    pooled = pooled_decay(table, "model")

    assert pooled[0] == pytest.approx(0.75)
    assert np.isnan(pooled[1])


def test_separation_bins_centers_sit_between_edges():
    bins = SeparationBins(max_bp=30, bin_width_bp=10)

    assert bins.edges.tolist() == [0, 10, 20, 30]
    assert bins.centers.tolist() == [5.0, 15.0, 25.0]
    assert bins.num_bins == 3


def test_covariation_metrics_reports_scalars_and_decay_curves():
    generator = torch.Generator().manual_seed(3)
    positions = torch.arange(12) * 10
    bins = SeparationBins(max_bp=60, bin_width_bp=10)

    record = covariation_metrics(
        positions=positions,
        matched_predicted=_structured_reads(20, 12, generator),
        matched_observed=_structured_reads(20, 12, generator),
        matched_replicate=_structured_reads(20, 12, generator),
        shuffles=CovariationShuffles(
            within_read=_structured_reads(20, 12, generator),
            read_axis=_structured_reads(20, 12, generator),
        ),
        bins=bins,
        min_joint_reads=5,
        all_predicted=_structured_reads(40, 12, generator),
        all_observed=_structured_reads(40, 12, generator),
    )

    assert record["n_covariation_pairs"] == 66
    for suffix in ("", "_pseudo_replicate"):
        assert np.isfinite(record[f"covariation_corr_of_corr_pearson{suffix}"])
        assert np.isfinite(record[f"covariation_corr_of_corr_spearman{suffix}"])
    # Both shuffles are zero by construction for an agreement between pair
    # correlations, so they earn a decay curve but no scalar.
    assert not [
        column
        for column in record
        if column.startswith("covariation_corr_of_corr") and "shuffle" in column
    ]
    for suffix in ("model", "observed", "shuffle", "read_axis_shuffle"):
        assert record[f"{DECAY_PREFIX}sum_{suffix}"].shape == (bins.num_bins,)
        assert record[f"{DECAY_PREFIX}count_{suffix}"].shape == (bins.num_bins,)


def _stored_region(region_id: str = "region_a", seed: int = 5) -> StoredSampleRegion:
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(24) * 10
    predicted = _structured_reads(64, 24, generator)
    observed = _structured_reads(32, 24, generator)
    observed[torch.rand((32, 24), generator=generator) < 0.2] = -1
    return StoredSampleRegion(
        region_id=region_id,
        dna=torch.zeros(24, dtype=torch.int8),
        predicted_smf=predicted,
        predicted_positions=positions,
        observed_smf=observed,
        observed_positions=positions,
        observed_mean=torch.full((24,), 0.5),
        window_start=0,
    )


def test_compute_metrics_table_reports_covariation_columns():
    config = MetricsConfig(progress=False)

    row = compute_metrics_table([_stored_region()], config).iloc[0]

    assert row["n_covariation_pairs"] > 0
    assert np.isfinite(row["covariation_corr_of_corr_pearson"])
    assert np.isfinite(row["covariation_corr_of_corr_pearson_pseudo_replicate"])
    assert row[f"{DECAY_PREFIX}sum_model"].shape == (config.separation_bins.num_bins,)


def _covariation_record(**kwargs):
    generator = torch.Generator().manual_seed(7)
    kwargs.setdefault("all_predicted", _structured_reads(40, 12, generator))
    kwargs.setdefault("all_observed", _structured_reads(40, 12, generator))
    return covariation_metrics(
        positions=torch.arange(12) * 10,
        matched_predicted=_structured_reads(20, 12, generator),
        matched_observed=_structured_reads(20, 12, generator),
        matched_replicate=_structured_reads(20, 12, generator),
        shuffles=CovariationShuffles(),
        bins=SeparationBins(max_bp=60, bin_width_bp=10),
        min_joint_reads=5,
        **kwargs,
    )


def test_covariation_metrics_score_every_model_read_and_the_baseline():
    generator = torch.Generator().manual_seed(8)
    index = PairIndex.from_positions(torch.arange(12) * 10)
    baseline = torch.linspace(1, 0, index.num_pairs)

    record = _covariation_record(
        all_predicted=_structured_reads(200, 12, generator),
        pair_baseline=baseline,
    )

    for method in ("pearson", "spearman"):
        assert np.isfinite(record[f"covariation_corr_of_corr_{method}_all_reads"])
        assert np.isfinite(record[f"covariation_corr_of_corr_{method}_positional_baseline"])
        for suffix in ("", "_pseudo_replicate", "_all_reads"):
            assert np.isfinite(record[f"covariation_residual_corr_of_corr_{method}{suffix}"])


def test_residual_agreement_against_a_zero_baseline_is_the_raw_agreement():
    generator = torch.Generator().manual_seed(9)
    index = PairIndex.from_positions(torch.arange(12) * 10)

    record = _covariation_record(
        all_predicted=_structured_reads(200, 12, generator),
        pair_baseline=torch.zeros(index.num_pairs),
    )

    for method in ("pearson", "spearman"):
        for suffix in ("", "_pseudo_replicate", "_all_reads"):
            assert record[f"covariation_residual_corr_of_corr_{method}{suffix}"] == pytest.approx(
                record[f"covariation_corr_of_corr_{method}{suffix}"]
            )


def test_all_reads_and_baseline_are_scored_against_every_observed_read():
    generator = torch.Generator().manual_seed(10)
    positions = torch.arange(12) * 10
    index = PairIndex.from_positions(positions)
    all_observed = _structured_reads(200, 12, generator)
    observed_pairs = pair_correlations(all_observed, index)

    # A model that reproduces every observed read, and a baseline that is the
    # observed pair correlations, agree perfectly only with the full set; the
    # matched half they would otherwise be scored against is unrelated noise.
    record = _covariation_record(
        all_predicted=all_observed,
        all_observed=all_observed,
        pair_baseline=observed_pairs.correlation,
    )

    for method in ("pearson", "spearman"):
        assert record[f"covariation_corr_of_corr_{method}_all_reads"] == pytest.approx(1.0)
        assert record[f"covariation_corr_of_corr_{method}_positional_baseline"] == pytest.approx(
            1.0
        )
        assert record[f"covariation_corr_of_corr_{method}"] < 0.99


def test_correlation_of_correlations_needs_enough_pairs():
    pairs = _uniform_pairs([0.1, 0.4, 0.2, 0.8])

    pearson, _, n_pairs = correlation_of_correlations(pairs, pairs, 5, min_pairs=4)
    assert pearson == pytest.approx(1.0)

    pearson, spearman, n_pairs = correlation_of_correlations(pairs, pairs, 5, min_pairs=5)
    assert np.isnan(pearson) and np.isnan(spearman)
    assert n_pairs == 4
