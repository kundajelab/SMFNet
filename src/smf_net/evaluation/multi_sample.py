"""Compare sample files of several conditions over their shared regions.

Each condition (for example a cell type) has its own model and a
``samples.h5`` file from ``scripts/sample.py``, sampled on the same data
split. :func:`validate_sample_files` checks that the files can be compared,
:func:`read_regions` reads one region from every file, and
:func:`shared_sites` finds the sites every condition covers well and splits
each condition's observed molecules into a selection and an evaluation set.
The split lets one set choose or bin regions while the other scores them, so
no region is selected and scored on the same molecules.
"""

import json
import math
import zlib
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from os import PathLike

import h5py
import numpy as np
import torch

from smf_net.data import site_means
from smf_net.evaluation.sample_io import (
    SourceRegistry,
    StoredSampleRegion,
    read_sample_region,
)

# Settings every compared file must share: top-level sampling metadata, and
# the data configuration that decides the split's regions and sites.
SHARED_METADATA = (
    "source",
    "split",
    "assay_type",
    "sample_steps",
    "reads_per_region",
    "model_input_length",
    "dtype",
)
SHARED_DATA_CONFIG = (
    "assay_type",
    "h5_region_length",
    "train_chrm",
    "val_chrm",
    "test_chrm",
    "fasta_path",
)


@dataclass(frozen=True)
class SharedSiteConfig:
    """Site coverage floors and the molecule split of :func:`shared_sites`.

    ``min_site_coverage`` is stated for all of a region's observed molecules
    and scaled by each set's share: with ``selection_fraction=0.3``, a 30-call
    floor becomes 9 calls in the selection set and 21 in the evaluation set.
    ``min_predicted_site_coverage`` applies to the model's reads unscaled. The
    split is seeded by ``seed`` and the region ID.
    """

    min_site_coverage: int = 30
    min_predicted_site_coverage: int = 10
    selection_fraction: float = 0.3
    seed: int = 0

    def observed_floor(self, floor: int, selection: bool) -> int:
        """An all-molecule floor scaled to the selection or evaluation share."""

        share = self.selection_fraction if selection else 1 - self.selection_fraction
        return max(1, math.ceil(floor * share))


def _attribute(handle: h5py.File, name: str, condition: str):
    if name not in handle.attrs:
        raise ValueError(f"{condition}: missing {name!r} metadata; is sampling complete?")
    value = handle.attrs[name]
    return value.item() if isinstance(value, np.generic) else value


def _manifest(handle: h5py.File, condition: str) -> tuple[dict, dict[str, int]]:
    if handle.attrs.get("schema") != "smfnet.samples":
        raise ValueError(f"{condition}: expected an smfnet.samples file")
    if int(handle.attrs.get("schema_version", -1)) != 2:
        raise ValueError(f"{condition}: expected sample schema version 2")
    # sample_io writes this attribute only after all region groups are written.
    count = int(_attribute(handle, "num_regions", condition))
    if "region_ids" not in handle or "regions" not in handle:
        raise ValueError(f"{condition}: incomplete sample file")
    if len(handle["region_ids"]) != count or len(handle["regions"]) != count:
        raise ValueError(f"{condition}: num_regions does not match the file contents")
    ids = handle["region_ids"].asstr()[:].tolist()
    if len(set(ids)) != count:
        raise ValueError(f"{condition}: duplicate region IDs")
    attrs = {key: _attribute(handle, key, condition) for key in SHARED_METADATA}
    if attrs["source"] != "split" or attrs["split"] not in {"val", "test"}:
        raise ValueError(f"{condition}: expected samples of a validation or test split")
    data_config = json.loads(_attribute(handle, "data_config", condition))
    for key in SHARED_DATA_CONFIG:
        if key not in data_config:
            raise ValueError(f"{condition}: data_config lacks {key}")
    if data_config["assay_type"] != attrs["assay_type"]:
        raise ValueError(f"{condition}: data and model assay types disagree")
    attrs["data_config"] = data_config
    attrs["checkpoint"] = _attribute(handle, "checkpoint", condition)
    attrs["num_regions"] = count
    return attrs, {region_id: i for i, region_id in enumerate(ids)}


def validate_sample_files(
    handles: Mapping[str, h5py.File],
) -> tuple[dict[str, dict], dict[str, dict[str, int]]]:
    """Check that completed sample files can be compared.

    Returns each condition's provenance and its region-ID-to-record index.
    """

    conditions = tuple(handles)
    if len(conditions) < 2 or any(not str(name).strip() for name in conditions):
        raise ValueError("Provide at least two sample files with nonempty names")
    manifests, indexes = {}, {}
    for condition in conditions:
        manifests[condition], indexes[condition] = _manifest(handles[condition], condition)
    reference = manifests[conditions[0]]
    for condition in conditions[1:]:
        current = manifests[condition]
        for key in SHARED_METADATA:
            if current[key] != reference[key]:
                raise ValueError(
                    f"{condition}: {key}={current[key]!r} differs from "
                    f"{conditions[0]} ({reference[key]!r})"
                )
        for key in SHARED_DATA_CONFIG:
            if current["data_config"][key] != reference["data_config"][key]:
                raise ValueError(f"{condition}: data_config.{key} differs from {conditions[0]}")
    return manifests, indexes


def open_sample_files(
    stack: ExitStack,
    sample_paths: Mapping[str, str | PathLike[str]],
) -> dict[str, h5py.File]:
    """Open each condition's samples file for reading, closed with ``stack``."""

    handles = {}
    for condition, path in sample_paths.items():
        try:
            handles[condition] = stack.enter_context(h5py.File(path, "r"))
        except OSError as exc:
            raise ValueError(
                f"{condition}: cannot read {path}; sampling may still be writing"
            ) from exc
    return handles


def shared_region_ids(indexes: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Region IDs present in every file, sorted."""

    shared = sorted(set.intersection(*(set(index) for index in indexes.values())))
    if not shared:
        raise ValueError("The sample files have no shared region IDs")
    return shared


def read_regions(
    handles: Mapping[str, h5py.File],
    indexes: Mapping[str, Mapping[str, int]],
    registries: Mapping[str, SourceRegistry],
    region_id: str,
) -> dict[str, StoredSampleRegion]:
    """Read one region from every condition's file."""

    return {
        condition: read_sample_region(
            handles[condition]["regions"][f"{indexes[condition][region_id]:08d}"],
            registries[condition],
        )
        for condition in handles
    }


def _validate_region(regions: Mapping[str, StoredSampleRegion]) -> None:
    first = next(iter(regions.values()))
    for condition, region in regions.items():
        if region.region_id != first.region_id:
            raise ValueError(f"Mismatched region IDs for {condition}")
        if region.window_start != first.window_start or not torch.equal(region.dna, first.dna):
            raise ValueError(f"{first.region_id}: output coordinates or DNA differ for {condition}")
        if region.observed_smf is None or region.observed_positions is None:
            raise ValueError(f"{first.region_id}: {condition} has no observed reads")


@dataclass(frozen=True)
class SharedSites:
    """One region's well-covered shared sites, with every condition's reads there.

    ``reads`` and ``means`` map each source (``predicted``, ``evaluation``,
    ``selection``) to per-condition values at ``positions``, in the order of
    ``conditions``: a list of ``[reads, sites]`` tensors, and a
    ``[conditions, sites]`` tensor of mean calls. ``generator`` continues the
    region's seeded stream after the split, for further random draws.
    """

    region_id: str
    conditions: tuple[str, ...]
    window_start: int
    positions: np.ndarray
    reads: dict[str, list[torch.Tensor]]
    means: dict[str, torch.Tensor]
    generator: torch.Generator


def _split(reads: torch.Tensor, fraction: float, generator: torch.Generator):
    """Randomly assign a region's molecules to selection and evaluation sets."""

    order = torch.randperm(len(reads), generator=generator)
    cut = round(fraction * len(reads))
    return reads[order[:cut]], reads[order[cut:]]


def shared_sites(
    regions: Mapping[str, StoredSampleRegion],
    config: SharedSiteConfig = SharedSiteConfig(),
) -> SharedSites:
    """Split one region's molecules and keep the sites every source covers.

    A site is kept when it is observed and predicted in every condition and
    reaches the coverage floors in every condition's predicted, evaluation and
    selection reads. The split is seeded by ``config.seed`` and the region ID,
    so it does not depend on which other regions are analyzed.
    """

    conditions = tuple(regions)
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required")
    _validate_region(regions)
    region_id = regions[conditions[0]].region_id
    # Torch's CPU generator keeps only a seed's low 32 bits, so hash both together.
    generator = torch.Generator().manual_seed(zlib.crc32(f"{config.seed}:{region_id}".encode()))
    positions = np.asarray(regions[conditions[0]].observed_positions)
    for region in regions.values():
        positions = np.intersect1d(positions, region.observed_positions.numpy())
        positions = np.intersect1d(positions, region.predicted_positions.numpy())
    reads: dict[str, list[torch.Tensor]] = {"predicted": [], "evaluation": [], "selection": []}
    for region in regions.values():
        observed = region.observed_smf[
            :, np.searchsorted(region.observed_positions.numpy(), positions)
        ]
        selection, evaluation = _split(observed, config.selection_fraction, generator)
        reads["selection"].append(selection)
        reads["evaluation"].append(evaluation)
        reads["predicted"].append(
            region.predicted_smf[:, np.searchsorted(region.predicted_positions.numpy(), positions)]
        )
    floors = {
        "predicted": config.min_predicted_site_coverage,
        "evaluation": config.observed_floor(config.min_site_coverage, selection=False),
        "selection": config.observed_floor(config.min_site_coverage, selection=True),
    }
    keep = np.ones(len(positions), dtype=bool)
    for source, by_condition in reads.items():
        for x in by_condition:
            keep &= (x != -1).sum(0).numpy() >= floors[source]
    sites = torch.as_tensor(np.nonzero(keep)[0])
    reads = {source: [x[:, sites] for x in by_condition] for source, by_condition in reads.items()}
    means = {
        source: torch.stack([site_means(x)[0] for x in by_condition])
        for source, by_condition in reads.items()
    }
    return SharedSites(
        region_id=region_id,
        conditions=conditions,
        window_start=regions[conditions[0]].window_start,
        positions=positions[keep],
        reads=reads,
        means=means,
        generator=generator,
    )
