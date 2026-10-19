"""Streaming HDF5 storage for profile ISM results."""

from collections.abc import Iterable, Iterator, Mapping
from os import PathLike

import h5py
import torch

from smf_net.evaluation.ism import ProfileISM, RegionProfileISM


def _create_stacked_dataset(
    handle: h5py.File,
    name: str,
    tensor: torch.Tensor,
) -> h5py.Dataset:
    values = tensor.cpu().numpy()
    return handle.create_dataset(
        name,
        shape=(0, *values.shape),
        maxshape=(None, *values.shape),
        chunks=(1, *values.shape),
        dtype=values.dtype,
        compression="gzip",
        shuffle=True,
    )


def _append(dataset: h5py.Dataset, index: int, tensor: torch.Tensor) -> None:
    dataset.resize(index + 1, axis=0)
    dataset[index] = tensor.cpu().numpy()


def write_profile_ism_h5(
    path: str | PathLike[str],
    regions: Iterable[RegionProfileISM],
    metadata: Mapping[str, str | int | float | bool] | None = None,
) -> int:
    """Write the stacked ``smfnet.profile_ism`` HDF5 format incrementally.

    For ``N`` regions, mutation-window length ``M``, and output length ``M``::

        attrs: schema, schema_version, num_regions, and run metadata
        region_ids             UTF-8 [N]
        dna                    int8  [N, M]
        influence              float [N, M, M]
        scaled_influence       float [N, M, M]
        base_deltas            float [N, M, 4, M]
        reference_logits       float [N, M]
        mutation_start         int64 [N]
        mutation_end           int64 [N]
        output_start           int64 [N]
        output_end             int64 [N]

    Numerical datasets are resizable and chunked by region, retaining streaming
    writes and efficient single-region reads without per-region HDF5 objects.
    """

    region_count = 0
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "smfnet.profile_ism"
        handle.attrs["schema_version"] = 1
        if metadata is not None:
            for key, value in metadata.items():
                handle.attrs[key] = str(value) if isinstance(value, str) else value
        region_ids = handle.create_dataset(
            "region_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        coordinate_datasets = {
            name: handle.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype="int64",
            )
            for name in (
                "mutation_start",
                "mutation_end",
                "output_start",
                "output_end",
            )
        }
        arrays: dict[str, h5py.Dataset] | None = None

        for region in regions:
            result = region.result
            tensors = {
                "dna": result.dna,
                "influence": result.influence,
                "scaled_influence": result.scaled_influence,
                "base_deltas": result.base_deltas,
                "reference_logits": result.reference_logits,
            }
            if arrays is None:
                arrays = {
                    name: _create_stacked_dataset(handle, name, tensor)
                    for name, tensor in tensors.items()
                }

            region_ids.resize(region_count + 1, axis=0)
            region_ids[region_count] = str(region.region_id)
            for name, tensor in tensors.items():
                _append(arrays[name], region_count, tensor)
            for name, value in (
                ("mutation_start", result.mutation_start),
                ("mutation_end", result.mutation_end),
                ("output_start", result.output_start),
                ("output_end", result.output_end),
            ):
                coordinate_datasets[name].resize(region_count + 1, axis=0)
                coordinate_datasets[name][region_count] = value
            region_count += 1

        handle.attrs["num_regions"] = region_count
    return region_count


def read_profile_ism_h5(
    path: str | PathLike[str],
) -> Iterator[RegionProfileISM]:
    """Stream profile ISM regions from an HDF5 result file."""

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "smfnet.profile_ism":
            raise ValueError("Expected an smfnet.profile_ism HDF5 file")
        region_ids = handle["region_ids"].asstr()
        for index in range(len(region_ids)):
            yield RegionProfileISM(
                region_id=region_ids[index],
                result=ProfileISM(
                    influence=torch.from_numpy(handle["influence"][index]),
                    scaled_influence=torch.from_numpy(handle["scaled_influence"][index]),
                    base_deltas=torch.from_numpy(handle["base_deltas"][index]),
                    reference_logits=torch.from_numpy(handle["reference_logits"][index]),
                    dna=torch.from_numpy(handle["dna"][index]),
                    mutation_start=int(handle["mutation_start"][index]),
                    mutation_end=int(handle["mutation_end"][index]),
                    output_start=int(handle["output_start"][index]),
                    output_end=int(handle["output_end"][index]),
                ),
            )
