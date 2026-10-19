from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

import smf_net.evaluation.metrics as metrics_module
from smf_net.evaluation.metrics import MetricsConfig, compute_metrics_table
from smf_net.evaluation.reporting import array_columns, write_site_correlations_tsv
from smf_net.evaluation.sample_io import StoredSampleRegion
from smf_net.evaluation.site_correlations import site_correlation_summary

CONFIG = MetricsConfig(
    progress=False,
    include_run_lengths=False,
    include_covariation=False,
    include_shuffle=False,
    min_site_coverage=2,
    min_correlation_sites=4,
)


def region(name, reads):
    reads = torch.tensor(reads, dtype=torch.int8)
    valid = reads != -1
    means = (reads * valid).sum(0) / valid.sum(0).clamp_min(1)
    return StoredSampleRegion(
        region_id=name,
        dna=torch.zeros(reads.shape[1], dtype=torch.int8),
        predicted_smf=reads.clone(),
        observed_smf=reads,
        predicted_positions=torch.arange(reads.shape[1]),
        observed_positions=torch.arange(reads.shape[1]),
        observed_mean=means,
        window_start=0,
    )


def test_pooling_includes_small_constant_regions_and_filters_low_coverage(tmp_path):
    regions = [region("low", [[0, 0, -1]] * 4), region("high", [[1, 1, -1]] * 4)]
    table = compute_metrics_table(regions, CONFIG)
    assert table.pearson.isna().all()
    summary = write_site_correlations_tsv(table, tmp_path / "sites.tsv", CONFIG)
    assert summary.n_sites.tolist() == [4, 4, 4]
    assert summary.pearson.tolist() == pytest.approx([1, 1, 1])
    assert summary.spearman.tolist() == pytest.approx([1, 1, 1])
    pd.testing.assert_frame_equal(summary, pd.read_csv(tmp_path / "sites.tsv", sep="\t"))
    assert set(table.filter(like="site_means_").columns) <= set(array_columns(table.columns))
    assert site_correlation_summary(table, min_sites=5).pearson.isna().all()


def test_matched_pooling_reuses_samples_and_common_mask(monkeypatch):
    captured = []
    original = metrics_module._prepare_matched_samples

    def capture(*args, **kwargs):
        samples = original(*args, **kwargs)
        captured.append(samples)
        return samples

    monkeypatch.setattr(metrics_module, "_prepare_matched_samples", capture)
    observed = [
        [0, 0, 1, -1, 0],
        [0, 1, 1, -1, 1],
        [1, 0, 1, -1, 0],
        [0, 1, 0, -1, 1],
        [0, 0, 1, 1, 0],
        [1, 1, 1, -1, 1],
        [0, 0, 0, -1, 0],
        [0, 1, 1, -1, 1],
    ]
    example = region("a", observed)
    # A missing model site must be removed for every comparison.
    example.predicted_smf[:, 4] = -1
    baseline_values = torch.tensor([0.2, 0.5, 0.8, 0.3, 0.9])
    flank = SimpleNamespace(assay_type=None, predict=lambda dna, sites: baseline_values)
    config = replace(CONFIG, min_correlation_sites=2)
    table = compute_metrics_table([example], config, flank_baseline=flank)
    predicted, target, replicate = captured[-1]
    mask = ((target != -1).sum(0) >= 2) & ((replicate != -1).sum(0) >= 2)
    mask &= (predicted != -1).any(0)
    target_mean = metrics_module._sample_mean(target)[0][mask].numpy()
    expected = {
        "model_matched": metrics_module._sample_mean(predicted)[0][mask].numpy(),
        "pseudo_replicate_matched": metrics_module._sample_mean(replicate)[0][mask].numpy(),
        "flank_baseline_matched": baseline_values[mask].numpy(),
    }
    summary = site_correlation_summary(table, 2).set_index("comparison")
    for comparison, prediction in expected.items():
        assert summary.loc[comparison, "n_sites"] == int(mask.sum())
        assert summary.loc[comparison, "pearson"] == pytest.approx(
            np.corrcoef(prediction, target_mean)[0, 1]
        )
        assert summary.loc[comparison, "spearman"] == pytest.approx(
            spearmanr(prediction, target_mean).statistic
        )
    assert summary.loc["flank_baseline_all_reads", "n_sites"] == 3
    assert summary.loc["flank_baseline_all_reads", "pearson"] == pytest.approx(
        np.corrcoef(baseline_values[:3], example.observed_mean[:3])[0, 1]
    )


@pytest.mark.parametrize("reads", [[[0, 0]] * 4, [[-1, -1]] * 4])
def test_undefined_correlations(reads):
    table = compute_metrics_table([region("a", reads)], CONFIG)
    assert site_correlation_summary(table, 0).pearson.isna().all()
    assert site_correlation_summary(table, 0).spearman.isna().all()
