import numpy as np
import pandas as pd
import pytest
import torch

from smf_net.evaluation.metrics import (
    MetricsConfig,
    _shuffle_along_read_axis,
    _shuffle_reads,
    compute_metrics_table,
)
from smf_net.evaluation.run_lengths import (
    RunLengthBins,
    RunLengthShuffles,
    donate_missingness,
    pooled_histogram,
    run_length_bp,
    run_length_histogram,
    run_length_metrics,
)
from smf_net.evaluation.sample_io import StoredSampleRegion


def _positions() -> torch.Tensor:
    return torch.tensor([0, 10, 20, 30, 40, 50])


def test_run_length_bp_measures_span_between_first_and_last_position():
    reads = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.int8)

    lengths = run_length_bp(reads, _positions(), value=1)

    assert lengths.tolist() == [20]


def test_run_length_bp_breaks_runs_at_missing_calls():
    bridged = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.int8)
    broken = torch.tensor([[0, 1, 1, -1, 1, 0]], dtype=torch.int8)

    assert run_length_bp(bridged, _positions(), value=1).tolist() == [30]
    assert sorted(run_length_bp(broken, _positions(), value=1).tolist()) == [0, 10]


def test_run_length_bp_spans_zero_for_a_single_position_run():
    reads = torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.int8)

    assert run_length_bp(reads, _positions(), value=1).tolist() == [0, 0]


def test_run_length_bp_drops_runs_touching_the_window_edges():
    reads = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.int8)

    assert run_length_bp(reads, _positions(), value=1).numel() == 0
    assert run_length_bp(reads, _positions(), value=0).tolist() == [10]


def test_run_length_bp_reports_every_read():
    reads = torch.tensor(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
        ],
        dtype=torch.int8,
    )

    assert sorted(run_length_bp(reads, _positions(), value=1).tolist()) == [10, 20]


def test_donate_missingness_copies_observed_patterns_without_changing_calls():
    predicted = torch.ones((8, 6), dtype=torch.int8)
    observed = torch.tensor(
        [
            [1, 1, -1, -1, 1, 1],
            [-1, 0, 0, 0, 0, -1],
        ],
        dtype=torch.int8,
    )

    donated = donate_missingness(predicted, observed, torch.Generator().manual_seed(3))

    observed_patterns = {tuple((row == -1).tolist()) for row in observed}
    for row in donated:
        assert tuple((row == -1).tolist()) in observed_patterns
        assert set(row[row != -1].tolist()) == {1}


def test_donate_missingness_without_observed_reads_is_a_passthrough():
    predicted = torch.ones((2, 3), dtype=torch.int8)

    donated = donate_missingness(
        predicted,
        torch.zeros((0, 3), dtype=torch.int8),
        torch.Generator().manual_seed(0),
    )

    assert torch.equal(donated, predicted)


def test_run_length_histogram_collects_long_runs_in_the_final_bin():
    bins = RunLengthBins(max_bp=40, bin_width_bp=10)

    counts = run_length_histogram(torch.tensor([0, 15, 35, 400, 4000]), bins)

    # Bins are [0, 10), [10, 20), [20, 30), [30, 40], and the clipped long runs
    # join the 35 bp run in the closed final bin.
    assert counts.tolist() == [1, 1, 0, 3]
    assert bins.num_bins == 4


def test_pooled_histogram_weighting_changes_region_contributions():
    table = pd.DataFrame(
        {"hist": [np.array([100, 0]), np.array([0, 1])]},
    )

    coverage_weighted = pooled_histogram(table, "hist", weight_by_coverage=True)
    equal_weighted = pooled_histogram(table, "hist", weight_by_coverage=False)

    assert coverage_weighted.tolist() == [100.0, 1.0]
    assert equal_weighted.tolist() == [0.5, 0.5]


def test_pooled_histogram_ignores_regions_without_runs():
    table = pd.DataFrame({"hist": [np.array([0, 0]), np.array([2, 2])]})

    assert pooled_histogram(table, "hist", weight_by_coverage=False).tolist() == [
        0.5,
        0.5,
    ]


def _stored_region() -> StoredSampleRegion:
    generator = torch.Generator().manual_seed(5)
    positions = torch.arange(24)
    predicted = (torch.rand((32, 24), generator=generator) < 0.5).to(torch.int8)
    observed = (torch.rand((16, 24), generator=generator) < 0.5).to(torch.int8)
    observed[torch.rand((16, 24), generator=generator) < 0.3] = -1
    return StoredSampleRegion(
        region_id="region_a",
        dna=torch.zeros(24, dtype=torch.int8),
        predicted_smf=predicted,
        predicted_positions=positions,
        observed_smf=observed,
        observed_positions=positions,
        observed_mean=torch.full((24,), 0.5),
        window_start=0,
    )


def test_compute_metrics_table_reports_run_length_columns():
    config = MetricsConfig(progress=False)

    row = compute_metrics_table([_stored_region()], config).iloc[0]

    for name in ("zeros", "ones"):
        assert row[f"n_run_length_{name}_model"] > 0
        assert row[f"n_run_length_{name}_observed"] > 0
        assert pd.notna(row[f"run_length_{name}_mean_model"])
        assert pd.notna(row[f"run_length_{name}_wasserstein"])
        assert pd.notna(row[f"run_length_{name}_wasserstein_pseudo_replicate"])
        assert pd.notna(row[f"run_length_{name}_wasserstein_all_reads"])
        assert pd.notna(row[f"run_length_{name}_wasserstein_shuffle"])
        assert pd.notna(row[f"run_length_{name}_wasserstein_read_axis_shuffle"])
        histogram = row[f"run_length_hist_{name}_model"]
        assert histogram.shape == (config.run_length_bins.num_bins,)
        assert histogram.sum() == row[f"n_run_length_{name}_model"]
        for suffix in ("shuffle", "read_axis_shuffle"):
            baseline = row[f"run_length_hist_{name}_{suffix}"]
            assert baseline.shape == (config.run_length_bins.num_bins,)
            assert baseline.sum() > 0


def test_read_axis_shuffle_keeps_the_run_structure_held_by_the_profile():
    """The two baselines bound different structure on profile-driven reads.

    Reads drawn independently from a strong positional profile hold run
    structure entirely in that profile, so the read-axis shuffle reproduces
    their run lengths while the within-read shuffle flattens the profile and
    loses them.
    """

    generator = torch.Generator().manual_seed(4)
    positions = torch.arange(24) * 10
    rates = torch.full((24,), 0.05)
    rates[8:16] = 0.95
    observed = (torch.rand((32, 24), generator=generator) < rates).to(torch.int8)

    record = run_length_metrics(
        positions=positions,
        predicted=observed,
        observed=observed,
        matched_predicted=observed,
        matched_observed=observed,
        matched_replicate=observed,
        shuffles=RunLengthShuffles(
            observed_within_read=_shuffle_reads(observed, generator),
            observed_read_axis=_shuffle_along_read_axis(observed, generator),
        ),
        bins=RunLengthBins(),
    )

    for name in ("zeros", "ones"):
        assert (
            record[f"run_length_{name}_wasserstein_read_axis_shuffle"]
            < record[f"run_length_{name}_wasserstein_shuffle"]
        ), name


def test_donated_missingness_shortens_dense_predicted_runs():
    region = _stored_region()
    positions = region.predicted_positions
    dense = run_length_bp(region.predicted_smf, positions, value=1)

    donated = run_length_bp(
        donate_missingness(
            region.predicted_smf,
            region.observed_smf,
            torch.Generator().manual_seed(1),
        ),
        positions,
        value=1,
    )

    # Without donation the model's runs are measured on fully covered reads and
    # are longer than anything the sparse observed reads could produce.
    assert donated.float().mean() < dense.float().mean()


def test_run_length_distances_need_enough_runs_in_both_samples():
    positions = torch.arange(6) * 10
    # Two interior runs of ones.
    reads = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.int8)

    def distance(min_runs):
        return run_length_metrics(
            positions=positions,
            predicted=reads,
            observed=reads,
            matched_predicted=reads,
            matched_observed=reads,
            matched_replicate=reads,
            shuffles=RunLengthShuffles(),
            bins=RunLengthBins(),
            min_runs=min_runs,
        )["run_length_ones_wasserstein"]

    assert distance(2) == pytest.approx(0.0)
    assert np.isnan(distance(3))
