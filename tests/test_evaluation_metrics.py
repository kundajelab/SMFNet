from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from smf_net.data import dna_str_to_tensor
from smf_net.evaluation.metrics import (
    MetricsConfig,
    _prepare_matched_samples,
    _shuffle_along_read_axis,
    align_sample_region,
    compute_metrics_table,
    compute_region_metrics,
    region_generator,
)
from smf_net.evaluation.sample_io import StoredSampleRegion

# The toy region has four reads over three shared sites, below every default
# floor on how much data a statistic may rest on.
TOY_FLOORS = {
    "min_site_coverage": 1,
    "min_correlation_sites": 0,
    "min_correlation_pairs": 0,
    "min_distribution_size": 0,
}


def _stored_region() -> StoredSampleRegion:
    predicted_smf = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 1, 1],
        ],
        dtype=torch.int8,
    )
    observed_smf = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.int8,
    )
    return StoredSampleRegion(
        region_id="region_a",
        dna=torch.tensor([0, 1, 2, 3] * 2, dtype=torch.int8),
        predicted_smf=predicted_smf,
        predicted_positions=torch.tensor([1, 3, 5, 7]),
        observed_smf=observed_smf,
        observed_positions=torch.tensor([0, 3, 5, 7]),
        observed_mean=torch.tensor([0.5, 0.0, 0.5, 1.0]),
        window_start=10,
        assay_type="gch_accessibility",
    )


def test_align_sample_region_uses_position_intersection():
    aligned = align_sample_region(_stored_region())

    assert aligned.positions.tolist() == [3, 5, 7]
    assert aligned.predicted_smf.shape == (4, 3)
    assert aligned.observed_smf.shape == (4, 3)
    assert aligned.observed_mean.tolist() == [0.0, 0.5, 1.0]


def test_compute_metrics_table_reports_correlations():
    config = MetricsConfig(include_shuffle=False, **TOY_FLOORS)
    metrics = compute_metrics_table([_stored_region()], config)
    row = metrics.iloc[0]

    assert row["region"] == "region_a"
    assert row["assay_type"] == "gch_accessibility"
    assert row["n_predicted_positions"] == 4
    assert row["n_observed_positions"] == 4
    assert row["n_shared_positions"] == 3
    assert row["n_valid_positions"] == 3
    assert row["pearson"] == pytest.approx(1.0)
    assert row["spearman"] == pytest.approx(1.0)
    assert row["n_matched_reads"] == 2


def test_prepare_matched_samples_uses_disjoint_equal_observed_groups():
    predicted = torch.arange(30).reshape(10, 3)
    observed = torch.arange(24).reshape(8, 3)

    predicted_sample, observed_sample, observed_replicate = _prepare_matched_samples(
        predicted,
        observed,
        generator=torch.Generator().manual_seed(11),
    )

    assert predicted_sample.shape == (4, 3)
    assert observed_sample.shape == (4, 3)
    assert observed_replicate.shape == (4, 3)
    observed_rows = {tuple(row.tolist()) for row in observed_sample}
    replicate_rows = {tuple(row.tolist()) for row in observed_replicate}
    assert observed_rows.isdisjoint(replicate_rows)


def test_metrics_are_reproducible_with_a_seed():
    # Shuffles, matched splits and donated missingness all draw from the seed.
    config = MetricsConfig(seed=17, progress=False)
    first = compute_metrics_table([_stored_region()], config)
    second = compute_metrics_table([_stored_region()], config)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("flag", "removed"),
    [
        ("include_shuffle", lambda column: "shuffle" in column),
        ("include_run_lengths", lambda column: column.startswith("run_length_")),
        ("include_covariation", lambda column: column.startswith("covariation_")),
    ],
)
def test_metric_groups_can_be_disabled(flag, removed):
    every = compute_metrics_table([_stored_region()], MetricsConfig(progress=False))
    without = compute_metrics_table(
        [_stored_region()], MetricsConfig(progress=False, **{flag: False})
    )

    assert [column for column in every.columns if removed(column)]
    assert not [column for column in without.columns if removed(column)]


def test_read_axis_shuffle_preserves_position_marginals_and_missingness():
    reads = torch.tensor(
        [
            [0, 0, -1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, -1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int8,
    )

    shuffled = _shuffle_along_read_axis(
        reads,
        torch.Generator().manual_seed(17),
    )

    assert torch.equal(shuffled == -1, reads == -1)
    for position in range(reads.shape[1]):
        valid = reads[:, position] != -1
        assert torch.equal(
            shuffled[valid, position].sort().values,
            reads[valid, position].sort().values,
        )
    assert not torch.equal(shuffled, reads)
    assert not torch.equal(shuffled[:, 0], shuffled[:, 1])


def _region_with_predicted(predicted: torch.Tensor) -> StoredSampleRegion:
    """Build a region with fixed observed reads and caller-supplied model reads."""

    generator = torch.Generator().manual_seed(3)
    observed = (torch.rand((16, 24), generator=generator) < 0.4).to(torch.int8)
    observed[torch.rand((16, 24), generator=generator) < 0.2] = -1
    valid = observed != -1
    counts = valid.sum(dim=0)
    positions = torch.arange(24)
    return StoredSampleRegion(
        region_id="region_a",
        dna=torch.zeros(24, dtype=torch.int8),
        predicted_smf=predicted,
        predicted_positions=positions,
        observed_smf=observed,
        observed_positions=positions,
        observed_mean=torch.where(
            counts > 0,
            (observed * valid).sum(dim=0) / counts.clamp_min(1),
            torch.tensor(-1.0),
        ),
        window_start=0,
    )


def test_shuffle_baselines_do_not_depend_on_the_model():
    # Eight-read matched samples can hold fewer runs than the default
    # distribution floor, which would make the run-length distances NaN and
    # the comparisons below pass vacuously.
    config = MetricsConfig(
        seed=5,
        progress=False,
        min_distribution_size=0,
    )
    generator = torch.Generator().manual_seed(9)
    structured = (torch.rand((12, 24), generator=generator) < 0.4).to(torch.int8)
    degenerate = torch.zeros((12, 24), dtype=torch.int8)

    first = compute_metrics_table([_region_with_predicted(structured)], config).iloc[0]
    second = compute_metrics_table([_region_with_predicted(degenerate)], config).iloc[0]

    shuffle_columns = [column for column in first.index if column.endswith("_shuffle")]
    assert "pearson_matched_shuffle" in shuffle_columns
    assert "run_length_zeros_wasserstein_shuffle" in shuffle_columns
    assert "run_length_zeros_wasserstein_read_axis_shuffle" in shuffle_columns
    for column in shuffle_columns:
        assert np.array_equal(
            np.asarray(first[column], dtype=float),
            np.asarray(second[column], dtype=float),
            equal_nan=True,
        ), column

    # The two models must actually score differently, or the baselines above
    # would match for the wrong reason.
    assert np.isfinite(first["run_length_zeros_wasserstein"])
    assert first["run_length_zeros_wasserstein"] != second["run_length_zeros_wasserstein"]


def test_matched_samples_use_every_read_the_observed_split_allows():
    generator = torch.Generator().manual_seed(9)
    predicted = (torch.rand((12, 24), generator=generator) < 0.4).to(torch.int8)

    row = compute_metrics_table(
        [_region_with_predicted(predicted)],
        MetricsConfig(progress=False, min_site_coverage=1),
    ).iloc[0]

    # Sixteen observed reads split into two halves of eight, and the model
    # sample is cut to the same size.
    assert row["n_matched_reads"] == 8
    for name in ("pearson", "spearman"):
        for suffix in ("", "_pseudo_replicate", "_shuffle"):
            assert np.isfinite(row[f"{name}_matched{suffix}"]), name + suffix


def test_matched_replicate_is_a_ceiling_a_faithful_model_reaches():
    generator = torch.Generator().manual_seed(21)
    profile = torch.rand(40, generator=generator)
    regions = []
    for index in range(40):
        observed = (torch.rand((64, 40), generator=generator) < profile).to(torch.int8)
        predicted = (torch.rand((64, 40), generator=generator) < profile).to(torch.int8)
        regions.append(
            StoredSampleRegion(
                region_id=f"region_{index}",
                dna=torch.zeros(40, dtype=torch.int8),
                predicted_smf=predicted,
                predicted_positions=torch.arange(40),
                observed_smf=observed,
                observed_positions=torch.arange(40),
                observed_mean=observed.float().mean(dim=0),
                window_start=0,
            )
        )

    table = compute_metrics_table(
        regions,
        MetricsConfig(
            include_run_lengths=False,
            include_covariation=False,
            progress=False,
        ),
    )

    # A model drawing from the observed profile matches the replicate, and a
    # within-read shuffle keeps almost none of the profile.
    model = table["pearson_matched"].median()
    replicate = table["pearson_matched_pseudo_replicate"].median()
    assert model == pytest.approx(replicate, abs=0.05)
    assert table["pearson_matched_shuffle"].median() < 0.3 < replicate


def test_read_axis_shuffle_has_no_profile_correlation():
    metrics = compute_metrics_table(
        [_stored_region()],
        MetricsConfig(
            include_shuffle=True,
            progress=False,
            **TOY_FLOORS,
        ),
    )

    assert "run_length_zeros_wasserstein_read_axis_shuffle" in metrics
    assert "pearson_read_axis_shuffle" not in metrics
    assert "spearman_read_axis_shuffle" not in metrics


def _baselines_with_profile(profile: np.ndarray, assay_type: str = "cg_gc_accessibility"):
    from smf_net.evaluation.positional_baselines import (
        PositionalBaselines,
        PositionalBaselineSettings,
    )

    return PositionalBaselines(
        profile=profile.astype(np.float32),
        pair_grid=np.zeros((profile.size, profile.size // 2 + 1), dtype=np.float32),
        region_length=profile.size,
        assay_type=assay_type,
        train_chromosomes=("chr2",),
        n_profile_regions=1,
        n_pair_regions=1,
        settings=PositionalBaselineSettings(),
    )


def test_positional_baseline_profile_is_looked_up_in_full_region_coordinates():
    region = replace(_stored_region(), window_start=10)
    observed_mean = align_sample_region(region).observed_mean
    # Defined only where the window's sites land once shifted by window_start,
    # so a lookup that ignored the shift would hit nothing but NaN.
    profile = np.full(30, np.nan)
    profile[[13, 15, 17]] = observed_mean.numpy()

    row = compute_metrics_table(
        [region],
        MetricsConfig(progress=False, **TOY_FLOORS),
        baselines=_baselines_with_profile(profile, region.assay_type),
    ).iloc[0]

    assert row["pearson_positional_baseline"] == pytest.approx(1.0)
    assert row["spearman_positional_baseline"] == pytest.approx(1.0)


def test_positional_baseline_columns_need_a_fit():
    metrics = compute_metrics_table([_stored_region()], MetricsConfig(progress=False))

    assert not [column for column in metrics.columns if "positional_baseline" in column]
    assert not [column for column in metrics.columns if "residual" in column]
    # The all-read covariation reference needs no fit.
    assert "covariation_corr_of_corr_pearson_all_reads" in metrics.columns


def test_regions_with_no_shared_positions_are_skipped_with_a_warning():
    scorable = _stored_region()
    # A sampling window can miss every observed site of a region.
    unscorable = replace(
        _stored_region(),
        region_id="region_b",
        observed_smf=torch.empty((4, 0), dtype=torch.int8),
        observed_positions=torch.empty(0, dtype=torch.long),
        observed_mean=torch.empty(0),
    )
    config = MetricsConfig(progress=False)

    with pytest.warns(UserWarning, match="Skipped 1 of 2 regions.*region_b"):
        metrics = compute_metrics_table([unscorable, scorable], config)

    assert metrics["region"].tolist() == ["region_a"]
    # Each region keeps the generator of its own index, so skipping one does
    # not change the values of the others.
    direct = compute_region_metrics(
        scorable,
        config,
        region_generator(config.seed, 1),
    )
    pd.testing.assert_frame_equal(metrics, pd.DataFrame.from_records([direct]))


def test_profile_correlations_skip_sites_few_observed_reads_cover():
    generator = torch.Generator().manual_seed(4)
    profile = torch.linspace(0.1, 0.9, 24)
    observed = (torch.rand((60, 24), generator=generator) < profile).to(torch.int8)
    # Only two reads cover the last four sites, the way short reads leave some
    # sites nearly uncovered.
    observed[2:, 20:] = -1
    predicted = (torch.rand((60, 24), generator=generator) < profile).to(torch.int8)
    valid = observed != -1
    region = StoredSampleRegion(
        region_id="region_a",
        dna=torch.zeros(24, dtype=torch.int8),
        predicted_smf=predicted,
        predicted_positions=torch.arange(24),
        observed_smf=observed,
        observed_positions=torch.arange(24),
        observed_mean=(observed * valid).sum(dim=0) / valid.sum(dim=0),
        window_start=0,
    )

    def row(min_site_coverage):
        return compute_metrics_table(
            [region],
            MetricsConfig(
                progress=False,
                min_site_coverage=min_site_coverage,
                include_run_lengths=False,
                include_covariation=False,
            ),
        ).iloc[0]

    assert row(1)["n_valid_positions"] == 24
    filtered = row(10)
    assert filtered["n_valid_positions"] == 20
    assert filtered["n_matched_correlation_sites"] == 20


def test_statistics_resting_on_too_little_data_are_nan():
    toy = _stored_region()  # three shared sites, two reads per matched sample

    def row(region=toy, **floors):
        config = MetricsConfig(
            progress=False,
            **{**TOY_FLOORS, **floors},
        )
        return compute_metrics_table([region], config).iloc[0]

    assert np.isfinite(row()["pearson"])
    assert np.isnan(row(min_correlation_sites=4)["pearson"])
    # The site count is still reported, so the reason is visible.
    assert row(min_correlation_sites=4)["n_valid_positions"] == 3

    # The toy region is too small to hold runs, so distances use one that does.
    generator = torch.Generator().manual_seed(9)
    predicted = (torch.rand((12, 24), generator=generator) < 0.4).to(torch.int8)
    runs = _region_with_predicted(predicted)
    assert np.isfinite(row(runs)["run_length_ones_wasserstein"])
    assert np.isnan(row(runs, min_distribution_size=1000)["run_length_ones_wasserstein"])


def test_flank_baseline_columns_score_the_flank_model_and_what_remains():
    from smf_net.evaluation.correlation import pearson
    from smf_net.evaluation.flank_baselines import (
        FlankBaseline,
        FlankBaselineSettings,
        site_types,
    )

    dna = dna_str_to_tensor("TAGCAAGCTTGCAAGCATAGCTTGCA")
    sites = torch.nonzero(site_types(dna, "gch_accessibility") >= 0).flatten()
    generator = torch.Generator().manual_seed(0)
    reads = lambda: (torch.rand((16, sites.numel()), generator=generator) < 0.5).to(torch.int8)
    observed = reads()
    region = StoredSampleRegion(
        region_id="region_k",
        dna=dna,
        predicted_smf=reads(),
        predicted_positions=sites,
        observed_smf=observed,
        observed_positions=sites,
        observed_mean=observed.float().mean(dim=0),
        window_start=0,
        assay_type="gch_accessibility",
    )
    flank = FlankBaseline(
        weights=torch.randn((1, 4, 4), generator=generator).double().numpy(),
        bias=np.zeros(1),
        base_frequencies=np.full((1, 4, 4), 0.25),
        n_sites=np.ones(1, dtype=np.int64),
        assay_type="gch_accessibility",
        train_chromosomes=("chr2",),
        n_regions=1,
        settings=FlankBaselineSettings(flank_bp=2),
    )
    config = MetricsConfig(progress=False, **TOY_FLOORS)

    row = compute_metrics_table([region], config, flank_baseline=flank).iloc[0]

    baseline = flank.predict(dna, sites)
    predicted_mean = region.predicted_smf.float().mean(dim=0)
    assert row["pearson_flank_baseline"] == pytest.approx(pearson(baseline, region.observed_mean))
    assert row["pearson_flank_residual"] == pytest.approx(
        pearson(predicted_mean - baseline, region.observed_mean - baseline)
    )
    # Matched residuals carry a replicate ceiling and a shuffle like the
    # matched profile correlations.
    for suffix in ("", "_pseudo_replicate", "_shuffle"):
        assert np.isfinite(row[f"pearson_flank_residual_matched{suffix}"])
