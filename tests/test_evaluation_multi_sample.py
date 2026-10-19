import json
from contextlib import ExitStack

import h5py
import numpy as np
import pytest
import torch

from smf_net.data import site_means
from smf_net.evaluation.multi_sample import (
    SharedSiteConfig,
    open_sample_files,
    read_regions,
    shared_region_ids,
    shared_sites,
    validate_sample_files,
)
from smf_net.evaluation.sample_io import read_source_registry

CONDITIONS = ("NP", "ES")


def _write_samples(path, condition, region_ids, *, reads=40, sample_steps=32):
    rng = np.random.default_rng(len(condition))
    data_config = {
        "assay_type": "gch_accessibility",
        "h5_region_length": 24,
        "train_chrm": ["chr2"],
        "val_chrm": ["chr8"],
        "test_chrm": ["chr1"],
        "fasta_path": "/reference/mm10.fa",
        "h5_paths": [f"/observed/{condition}.h5"],
    }
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            {
                "schema": "smfnet.samples",
                "schema_version": 2,
                "num_regions": len(region_ids),
                "source": "split",
                "split": "val",
                "assay_type": "gch_accessibility",
                "sample_steps": sample_steps,
                "reads_per_region": reads,
                "model_input_length": 48,
                "dtype": "bfloat16",
                "checkpoint": f"/models/{condition}.ckpt",
                "data_config": json.dumps(data_config),
            }
        )
        h5.create_dataset("region_ids", data=np.asarray(region_ids, dtype=h5py.string_dtype()))
        for number, region_id in enumerate(region_ids):
            group = h5.create_group(f"regions/{number:08d}")
            group.attrs["region_id"] = region_id
            group.attrs["window_start"] = 12
            group.create_dataset("dna", data=np.tile(np.arange(4, dtype=np.int8), 6))
            for label, positions in (("observed", np.arange(12)), ("predicted", np.arange(1, 12))):
                smf = (rng.random((reads, len(positions))) < 0.5).astype(np.int8)
                if label == "observed":
                    smf[:, 0] = -1  # position 0 is never observed in this condition
                sub = group.create_group(label)
                sub.create_dataset("positions", data=positions)
                sub.create_dataset("smf", data=smf)
                if label == "observed":
                    sub.create_dataset("mean", data=smf.mean(axis=0))


def _paths(tmp_path, **kwargs):
    paths = {}
    for i, condition in enumerate(CONDITIONS):
        paths[condition] = tmp_path / f"{condition}.h5"
        order = ["chr8:a", "chr8:b"] if i else ["chr8:b", "chr8:a"]
        _write_samples(paths[condition], condition, order, **kwargs)
    return paths


def _regions(paths, region_id):
    with ExitStack() as stack:
        handles = open_sample_files(stack, paths)
        _, indexes = validate_sample_files(handles)
        registries = {c: read_source_registry(h) for c, h in handles.items()}
        assert shared_region_ids(indexes) == ["chr8:a", "chr8:b"]
        return read_regions(handles, indexes, registries, region_id)


def test_shared_sites_keep_covered_sites_and_split_molecules(tmp_path):
    regions = _regions(_paths(tmp_path), "chr8:a")
    assert list(regions) == list(CONDITIONS)

    sites = shared_sites(regions, SharedSiteConfig(min_site_coverage=10))

    # Position 0 is never observed and is not predicted, so only 1..11 remain.
    assert sites.positions.tolist() == list(range(1, 12))
    assert [len(x) for x in sites.reads["selection"]] == [12, 12]
    assert [len(x) for x in sites.reads["evaluation"]] == [28, 28]
    for source, by_condition in sites.reads.items():
        assert torch.equal(
            sites.means[source], torch.stack([site_means(x)[0] for x in by_condition])
        )

    # Raising the floor above the selection set's 12 molecules drops every site.
    empty = shared_sites(regions, SharedSiteConfig(min_site_coverage=50))
    assert empty.positions.size == 0
    assert empty.means["selection"].shape == (2, 0)


def test_molecule_split_is_seeded_by_seed_and_region(tmp_path):
    paths = _paths(tmp_path)
    a = shared_sites(_regions(paths, "chr8:a"))
    again = shared_sites(_regions(paths, "chr8:a"))
    reseeded = shared_sites(_regions(paths, "chr8:a"), SharedSiteConfig(seed=1))

    assert all(torch.equal(x, y) for x, y in zip(a.reads["selection"], again.reads["selection"]))
    assert not all(
        torch.equal(x, y) for x, y in zip(a.reads["selection"], reseeded.reads["selection"])
    )


def test_incompatible_sample_files_are_refused(tmp_path):
    paths = _paths(tmp_path)
    _write_samples(paths["ES"], "ES", ["chr8:a", "chr8:b"], sample_steps=16)
    with ExitStack() as stack, pytest.raises(ValueError, match="sample_steps"):
        validate_sample_files(open_sample_files(stack, paths))
