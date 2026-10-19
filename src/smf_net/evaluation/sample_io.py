"""Streaming storage for paired model and observed SMF samples."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from os import PathLike

import h5py
import torch

from smf_net.data import RegionSource
from smf_net.evaluation.sampling import PairedSampleBatch, SampleBatch


@dataclass(frozen=True)
class StoredSampleRegion:
    region_id: str
    dna: torch.Tensor
    predicted_smf: torch.Tensor
    predicted_positions: torch.Tensor
    observed_smf: torch.Tensor | None
    observed_positions: torch.Tensor | None
    observed_mean: torch.Tensor | None
    window_start: int
    source: RegionSource | None = None
    assay_type: str | None = None


SourceRegistry = tuple[tuple[str, ...], tuple[str, ...]]


def _write_reads(group: h5py.Group, name: str, reads: torch.Tensor) -> None:
    values = reads.cpu().to(torch.int8).numpy()
    kwargs = {"compression": "gzip", "shuffle": True} if values.size else {}
    group.create_dataset(name, data=values, **kwargs)


def write_samples_h5(
    path: str | PathLike[str],
    batches: Iterable[PairedSampleBatch | SampleBatch],
    metadata: Mapping[str, str | int | float | bool] | None = None,
) -> int:
    """Write the ``smfnet.samples`` HDF5 format incrementally.

    Format::

        attrs: schema, schema_version, num_regions, and run metadata
        sources/paths                 UTF-8 [source]
        sources/kinds                 UTF-8 [source]
        region_ids                    UTF-8 [N], in record order
        regions/00000000/
            attrs: region_id, window_start, model_input_length,
                   model_output_offset, source_index (optional),
                   reference_source_index (optional)
            dna                       int8 [output_length]
            predicted/positions       int [predicted_positions]
            predicted/smf             int8 [predicted_reads, positions]
            observed/positions        int [observed_positions]       (optional)
            observed/smf              int8 [observed_reads, positions] (optional)
            observed/mean             float [observed_positions]     (optional)

    Per-region groups preserve the ragged read and position dimensions.
    ``region_ids`` provides a direct index into the numbered record groups.
    """

    region_count = 0
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "smfnet.samples"
        handle.attrs["schema_version"] = 2
        if metadata is not None:
            for key, value in metadata.items():
                handle.attrs[key] = str(value) if isinstance(value, str) else value
        regions_group = handle.create_group("regions")
        sources_group = handle.create_group("sources")
        source_paths = sources_group.create_dataset(
            "paths",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        source_kinds = sources_group.create_dataset(
            "kinds",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        source_indices: dict[tuple[str, str], int] = {}

        def register_source(kind: str, source_path: str) -> int:
            key = (kind, source_path)
            if key in source_indices:
                return source_indices[key]
            source_index = len(source_indices)
            source_paths.resize(source_index + 1, axis=0)
            source_kinds.resize(source_index + 1, axis=0)
            source_paths[source_index] = source_path
            source_kinds[source_index] = kind
            source_indices[key] = source_index
            return source_index

        region_ids = handle.create_dataset(
            "region_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        for batch in batches:
            samples = batch.samples if isinstance(batch, PairedSampleBatch) else batch
            observed_regions = (
                batch.observed_regions
                if isinstance(batch, PairedSampleBatch)
                else (None,) * samples.batch_size
            )
            for batch_idx, observed in enumerate(observed_regions):
                region_id = str(samples.region_ids[batch_idx])
                predicted = samples.reads[batch_idx]
                predicted_mask = (predicted != -1).any(dim=0)

                region_group = regions_group.create_group(f"{region_count:08d}")
                region_group.attrs["region_id"] = region_id
                region_group.attrs["window_start"] = (
                    observed.window_start if observed is not None else samples.output_offset
                )
                region_group.attrs["model_input_length"] = samples.input_length
                region_group.attrs["model_output_offset"] = samples.output_offset
                source = observed.source if observed is not None else None
                if source is None:
                    source = samples.sources[batch_idx]
                if source is not None:
                    region_group.attrs["source_index"] = register_source(
                        source.kind,
                        source.path,
                    )
                    if source.reference_fasta_path is not None:
                        region_group.attrs["reference_source_index"] = register_source(
                            "reference_fasta",
                            source.reference_fasta_path,
                        )

                dna = (
                    observed.dna
                    if observed is not None
                    else samples.dna[batch_idx, samples.output_slice]
                )
                region_group.create_dataset(
                    "dna",
                    data=dna.cpu().to(torch.int8).numpy(),
                )

                predicted_group = region_group.create_group("predicted")
                predicted_group.create_dataset(
                    "positions",
                    data=torch.nonzero(predicted_mask).squeeze(1).cpu().numpy(),
                )
                _write_reads(
                    predicted_group,
                    "smf",
                    predicted[:, predicted_mask],
                )

                if observed is not None:
                    observed_group = region_group.create_group("observed")
                    observed_group.create_dataset(
                        "positions",
                        data=observed.positions.cpu().numpy(),
                    )
                    _write_reads(observed_group, "smf", observed.smf)
                    observed_group.create_dataset(
                        "mean",
                        data=observed.mean().cpu().numpy(),
                    )
                region_ids.resize(region_count + 1, axis=0)
                region_ids[region_count] = region_id
                region_count += 1

        handle.attrs["num_regions"] = region_count
    return region_count


def read_source_registry(handle: h5py.File) -> SourceRegistry:
    if "sources" not in handle:
        return (), ()
    sources = handle["sources"]
    return (
        tuple(sources["paths"].asstr()[:]),
        tuple(sources["kinds"].asstr()[:]),
    )


def _read_region_source(
    region: h5py.Group,
    registry: SourceRegistry,
) -> RegionSource | None:
    if "source_index" not in region.attrs:
        return None
    paths, kinds = registry
    source_index = int(region.attrs["source_index"])
    reference_path = None
    if "reference_source_index" in region.attrs:
        reference_index = int(region.attrs["reference_source_index"])
        reference_path = paths[reference_index]
    return RegionSource(
        kind=kinds[source_index],
        path=paths[source_index],
        reference_fasta_path=reference_path,
    )


def read_sample_region(
    region: h5py.Group,
    registry: SourceRegistry | None = None,
) -> StoredSampleRegion:
    if registry is None:
        registry = read_source_registry(region.file)
    predicted = region["predicted"]
    observed = region.get("observed")
    assay_type = region.file.attrs.get("assay_type")
    return StoredSampleRegion(
        region_id=str(region.attrs["region_id"]),
        dna=torch.from_numpy(region["dna"][:]),
        predicted_smf=torch.from_numpy(predicted["smf"][:]),
        predicted_positions=torch.from_numpy(predicted["positions"][:]),
        observed_smf=(torch.from_numpy(observed["smf"][:]) if observed is not None else None),
        observed_positions=(
            torch.from_numpy(observed["positions"][:]) if observed is not None else None
        ),
        observed_mean=(torch.from_numpy(observed["mean"][:]) if observed is not None else None),
        window_start=int(region.attrs["window_start"]),
        source=_read_region_source(region, registry),
        assay_type=None if assay_type is None else str(assay_type),
    )


def read_samples_h5(
    path: str | PathLike[str],
) -> Iterator[StoredSampleRegion]:
    """Stream regions from a paired-sample HDF5 file."""

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "smfnet.samples":
            raise ValueError("Expected an smfnet.samples HDF5 file")
        registry = read_source_registry(handle)
        for key in sorted(handle["regions"]):
            yield read_sample_region(handle["regions"][key], registry)
