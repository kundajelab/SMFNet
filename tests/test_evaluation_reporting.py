from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from smf_net.data import RegionSource, SparseSMFRegion
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.covariation import DECAY_PREFIX
from smf_net.evaluation.metrics import MetricsConfig
from smf_net.evaluation.reporting import (
    write_covariation_decay,
    write_drop_summary,
    write_gap_closed_tsv,
    write_metrics_tsv,
    write_run_length_histograms,
)
from smf_net.evaluation.sample_io import write_samples_h5
from smf_net.evaluation.sampling import SamplingConfig, sample_observed_regions
from smf_net.evaluation.site_correlations import site_correlation_summary

# The baselines need a training split, which these synthetic samples lack.
NO_BASELINES = {"include_positional_baselines": False, "include_flank_baselines": False}


class FakeModel:
    device = torch.device("cpu")
    cfg = SimpleNamespace(input_length=8, assay_type="cg_gc_accessibility")

    def sample(self, dna, reads_per_region, generator=None, **kwargs):
        samples = torch.randint(
            0, 2, (dna.shape[0], reads_per_region, dna.shape[-1] // 2), generator=generator
        )
        samples[..., 0] = MASK_TOKEN
        return samples


def _region(index: int, positions: list[int], reads: int = 8, source=None) -> SparseSMFRegion:
    return SparseSMFRegion(
        region_id=f"region_{index}",
        dna=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int8),
        smf=torch.randint(
            0, 2, (reads, len(positions)), generator=torch.Generator().manual_seed(index)
        ).to(torch.int8),
        positions=torch.tensor(positions, dtype=torch.long),
        window_start=index * 10,
        source=source,
    )


def _write_samples(path, regions, **metadata):
    write_samples_h5(
        path,
        sample_observed_regions(
            FakeModel(), regions, SamplingConfig(reads_per_region=8, progress=False)
        ),
        metadata=metadata,
    )
    return path


def test_parallel_metrics_match_sequential_metrics(tmp_path):
    source = RegionSource("observed_h5", str(tmp_path / "observed.h5"))
    # One region has reads but no observed site in its window.
    regions = [_region(0, [1, 2, 3], source=source), _region(1, [], source=source)]
    regions += [_region(index, [1, 2, 3], source=source) for index in (2, 3)]
    samples_path = _write_samples(tmp_path / "samples.h5", regions)

    tables = []
    for num_workers in (0, 2):
        with pytest.warns(UserWarning, match="Skipped 1 of 4 regions.*region_1"):
            tables.append(
                write_metrics_tsv(
                    samples_path,
                    tmp_path / f"metrics_{num_workers}.tsv",
                    MetricsConfig(seed=23, progress=False, num_workers=num_workers, **NO_BASELINES),
                )
            )
    sequential, parallel = tables

    pd.testing.assert_frame_equal(sequential, parallel)
    assert parallel["region"].tolist() == ["region_0", "region_2", "region_3"]
    assert parallel["source_path"].tolist() == [source.path] * 3
    pd.testing.assert_frame_equal(
        site_correlation_summary(sequential), site_correlation_summary(parallel)
    )


def test_written_outputs_keep_tables_and_arrays_apart(tmp_path):
    source = RegionSource("observed_h5", str(tmp_path / "observed.h5"))
    samples_path = _write_samples(
        tmp_path / "samples.h5",
        [_region(index, [1, 2, 3], source=source) for index in range(2)],
        assay_type="gch_accessibility",
    )
    config = MetricsConfig(progress=False, include_shuffle=False, **NO_BASELINES)

    metrics = write_metrics_tsv(samples_path, tmp_path / "metrics.tsv", config)

    written = pd.read_csv(tmp_path / "metrics.tsv", sep="\t")
    assert set(written["assay_type"]) == {"gch_accessibility"}
    assert written["source_type"].tolist() == ["observed_h5"] * 2
    assert written["source_path"].tolist() == [source.path] * 2
    # Per-region arrays go to the npz files, not the table.
    assert not [
        column
        for column in written.columns
        if column.startswith(("run_length_hist_", DECAY_PREFIX, "site_means_"))
    ]
    # Without shuffles there is no baseline to normalize against.
    assert write_gap_closed_tsv(metrics, tmp_path / "metrics_summary.tsv").empty

    assert write_run_length_histograms(metrics, tmp_path / "run_lengths.npz", config)
    with np.load(tmp_path / "run_lengths.npz") as histograms:
        assert histograms["region_ids"].tolist() == ["region_0", "region_1"]
        assert histograms["run_length_hist_ones_model"].shape == (
            2,
            config.run_length_bins.num_bins,
        )
        assert histograms["bin_edges"].tolist() == config.run_length_bins.edges.tolist()

    assert write_covariation_decay(metrics, tmp_path / "decay.npz", config)
    with np.load(tmp_path / "decay.npz") as decay:
        assert decay["region_ids"].tolist() == ["region_0", "region_1"]
        assert decay[f"{DECAY_PREFIX}sum_model"].shape == (
            2,
            config.separation_bins.num_bins,
        )
        assert decay["bin_edges"].tolist() == config.separation_bins.edges.tolist()


def test_drop_summary_explains_every_missing_region_and_site(tmp_path):
    samples_path = _write_samples(
        tmp_path / "samples.h5", [_region(0, [0, 1, 2, 3], reads=4), _region(1, [], reads=4)]
    )
    config = MetricsConfig(progress=False, **NO_BASELINES)
    with pytest.warns(UserWarning, match="Skipped 1 of 2"):
        metrics = write_metrics_tsv(samples_path, tmp_path / "metrics.tsv", config)

    drops = write_drop_summary(metrics, tmp_path, config)

    assert (tmp_path / "metrics_drops.tsv").exists()
    assert (tmp_path / "skipped_regions.txt").read_text() == "region_1\n"
    profile = drops[(drops["metric"] == "profile") & (drops["unit"] == "regions")]
    # Four reads per site sit below the default coverage and site floors.
    assert dict(zip(profile["reason"], profile["count"])) == {
        "scored": 0,
        "too few": 1,
        "undefined": 0,
        "no shared positions": 1,
    }
    # Every region in the samples file is accounted for in every group.
    region_rows = drops[drops["unit"] == "regions"]
    assert (region_rows.groupby("metric")["count"].sum() == 2).all()
    sites = drops[(drops["metric"] == "profile") & (drops["unit"] == "sites")]
    sites = dict(zip(sites["reason"], sites["count"]))
    assert sites["low coverage"] == sites["shared by model and observed"]
