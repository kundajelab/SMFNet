import json

import h5py
import numpy as np
import pytest
import torch

from smf_net.evaluation.covariation import PairIndex
from smf_net.evaluation.metrics import MetricsConfig
from smf_net.evaluation.positional_baselines import (
    PositionalBaselines,
    PositionalBaselineSettings,
    check_regions_outside_training,
    data_config_from_samples,
    fit_positional_baselines,
)
from smf_net.evaluation.reporting import train_baselines_for

UNSMOOTHED = PositionalBaselineSettings(
    profile_bandwidth_bp=0,
    separation_bandwidth_bp=0,
    midpoint_bandwidth_bp=0,
    max_pair_regions=None,
    min_joint_reads=2,
)


def _baselines(profile: np.ndarray, pair_grid: np.ndarray) -> PositionalBaselines:
    return PositionalBaselines(
        profile=profile.astype(np.float32),
        pair_grid=pair_grid.astype(np.float32),
        region_length=profile.size,
        assay_type="cg_gc_accessibility",
        train_chromosomes=("chr2",),
        n_profile_regions=1,
        n_pair_regions=1,
        settings=PositionalBaselineSettings(),
    )


def test_profile_fit_recovers_a_positional_bump():
    generator = torch.Generator().manual_seed(0)
    length = 40
    positions = torch.arange(0, length, 2)
    truth = 0.2 + 0.6 * torch.exp(-((positions - length / 2) ** 2) / 50.0)
    reads = [
        (torch.rand((200, positions.numel()), generator=generator) < truth).to(torch.int8)
        for _ in range(20)
    ]

    baselines = fit_positional_baselines(
        reads,
        [positions] * len(reads),
        length,
        "cg_gc_accessibility",
        ["chr2"],
        UNSMOOTHED,
        progress=False,
    )

    np.testing.assert_allclose(baselines.profile[positions], truth, atol=0.03)
    # Positions no region has a site at stay undefined instead of reading as 0.
    assert np.isnan(baselines.profile[1])


def test_pair_fit_recovers_coupling_that_falls_with_separation():
    generator = torch.Generator().manual_seed(1)
    positions = torch.arange(16) * 10
    # Calls are shared within blocks of four adjacent sites, so near pairs are
    # perfectly coupled and pairs in different blocks are independent.
    reads = [
        (torch.rand((100, 4), generator=generator) < 0.5).repeat_interleave(4, dim=1).to(torch.int8)
        for _ in range(10)
    ]

    grid = fit_positional_baselines(
        reads,
        [positions] * len(reads),
        200,
        "cg_gc_accessibility",
        ["chr2"],
        UNSMOOTHED,
        progress=False,
    ).pair_grid

    near = grid[10][np.isfinite(grid[10])]
    far = grid[150][np.isfinite(grid[150])]
    assert near.mean() > 0.3
    assert abs(far.mean()) < 0.1


def test_pair_lookup_uses_separation_and_midpoint_offset():
    length = 41  # centre at 20
    grid = np.add.outer(np.arange(length) * 1000.0, np.arange(length // 2 + 1))
    baselines = _baselines(np.zeros(length), grid)
    positions = torch.tensor([0, 10, 30])

    values = baselines.pairs_at(positions, PairIndex.from_positions(positions))

    # (0, 10): 10 bp apart, midpoint 15 bp from centre; (0, 30): 30 and 5;
    # (10, 30): 20 and 0.
    assert values.tolist() == [10_015.0, 30_005.0, 20_000.0]


def test_only_the_pair_fit_is_subsampled():
    positions = torch.arange(0, 20, 2)
    reads = [torch.zeros((4, positions.numel()), dtype=torch.int8)] * 30

    baselines = fit_positional_baselines(
        reads,
        [positions] * len(reads),
        20,
        "cg_gc_accessibility",
        ["chr2"],
        PositionalBaselineSettings(max_pair_regions=7),
        progress=False,
    )

    assert baselines.n_profile_regions == 30
    assert baselines.n_pair_regions == 7


def test_save_writes_the_fit_and_its_provenance(tmp_path):
    baselines = _baselines(np.arange(10.0), np.ones((10, 6)))
    path = tmp_path / "positional_baselines.npz"

    baselines.save(path)

    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["profile"], baselines.profile)
        assert saved["pair_grid"].shape == (10, 6)
        metadata = json.loads(str(saved["metadata"]))
    assert metadata["train_chromosomes"] == ["chr2"]
    assert metadata["settings"]["separation_bandwidth_bp"] == 2.0


def test_regions_on_training_chromosomes_are_refused():
    check_regions_outside_training(["chr1_1_41", "chr3_5_45"], ["chr2"])

    with pytest.raises(ValueError, match="chr2"):
        check_regions_outside_training(["chr1_1_41", "chr2_1_41"], ["chr2"])


def _write_samples_metadata(path, region_ids, **attrs) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "smfnet.samples"
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.create_dataset(
            "region_ids",
            data=np.asarray(region_ids, dtype=object),
            dtype=h5py.string_dtype(),
        )


def _data_config(fasta_path, h5_path, length) -> str:
    return json.dumps(
        {
            "fasta_path": str(fasta_path),
            "h5_paths": [str(h5_path)],
            "assay_type": "cg_gc_accessibility",
            "h5_region_length": length,
            "train_chrm": ["chr2"],
            "val_chrm": ["chr8"],
            "test_chrm": ["chr1"],
        }
    )


def test_data_config_comes_from_the_samples_file(tmp_path):
    path = tmp_path / "samples.h5"
    _write_samples_metadata(
        path,
        ["chr1_1_41"],
        source="split",
        data_config=_data_config("genome.fa", "regions.h5", 40),
    )

    config = data_config_from_samples(path)

    assert config.train_chrm == ["chr2"]
    assert config.h5_paths == ["regions.h5"]
    assert config.h5_region_length == 40


def test_baselines_are_fit_from_the_training_data_a_samples_file_records(tmp_path):
    length = 40
    genome = "CG" * (length // 2)
    fasta_path = tmp_path / "genome.fa"
    fasta_path.write_text(f">chr1\n{genome}\n>chr2\n{genome}\n")
    h5_path = tmp_path / "regions.h5"
    generator = torch.Generator().manual_seed(2)
    positions = np.arange(0, length, 2)
    with h5py.File(h5_path, "w") as handle:
        for chromosome in ("chr1", "chr2"):
            region = f"{chromosome}_1_{length + 1}"
            handle.create_dataset(f"smf_pos/{region}", data=positions)
            handle.create_dataset(
                f"smf_mat/{region}",
                data=(torch.rand((8, positions.size), generator=generator) < 0.5)
                .to(torch.int8)
                .numpy(),
            )
    samples_path = tmp_path / "samples.h5"
    _write_samples_metadata(
        samples_path,
        [f"chr1_1_{length + 1}"],
        source="split",
        data_config=_data_config(fasta_path, h5_path, length),
    )

    baselines, flank = train_baselines_for(samples_path, MetricsConfig(progress=False))

    # Only the chr2 training region contributes, to both fits.
    assert baselines.n_profile_regions == 1
    assert baselines.train_chromosomes == ("chr2",)
    assert np.isfinite(baselines.profile[positions]).all()
    assert flank.n_regions == 1
    assert flank.n_sites.sum() == positions.size
