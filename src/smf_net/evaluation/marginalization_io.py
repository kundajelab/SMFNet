"""Streaming HDF5 storage for motif marginalization samples."""

from collections.abc import Iterable, Mapping
from os import PathLike

import h5py
import torch
from tqdm import tqdm

from smf_net.evaluation.marginalization import (
    MarginalizationDesign,
    MarginalizationSampleBatch,
)


def _write_tensor(
    group: h5py.Group,
    name: str,
    tensor: torch.Tensor,
) -> h5py.Dataset:
    values = tensor.cpu().numpy()
    return group.create_dataset(
        name,
        data=values,
        compression="gzip",
        shuffle=True,
    )


def _write_strings(
    handle: h5py.File,
    name: str,
    values: list[str],
) -> h5py.Dataset:
    dataset = handle.create_dataset(
        name,
        shape=(len(values),),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )
    dataset[:] = [str(value) for value in values]
    return dataset


def _sample_datasets(
    group: h5py.Group,
    design: MarginalizationDesign,
    example: torch.Tensor,
) -> dict[str, h5py.Dataset]:
    reads_per_background, output_length = example.shape[-2:]
    options = {
        "dtype": "int8",
        "compression": "gzip",
        "shuffle": True,
    }
    return {
        "background": group.create_dataset(
            "background",
            shape=(design.n_backgrounds, reads_per_background, output_length),
            chunks=(1, reads_per_background, output_length),
            **options,
        ),
        "motif": group.create_dataset(
            "motif",
            shape=(
                design.n_motifs,
                design.n_backgrounds,
                reads_per_background,
                output_length,
            ),
            chunks=(1, 1, reads_per_background, output_length),
            **options,
        ),
        "control": group.create_dataset(
            "control",
            shape=(
                design.n_motifs,
                design.n_backgrounds,
                reads_per_background,
                output_length,
            ),
            chunks=(1, 1, reads_per_background, output_length),
            **options,
        ),
    }


def write_marginalizations_h5(
    path: str | PathLike[str],
    design: MarginalizationDesign,
    batches: Iterable[MarginalizationSampleBatch],
    metadata: Mapping[str, str | int | float | bool] | None = None,
    *,
    show_progress: bool = True,
) -> None:
    """Write raw samples in the stacked ``smfnet.marginalizations`` format.

    For ``M`` motifs, ``B`` backgrounds, ``R`` reads, input length ``L``, and
    model output length ``O``::

        attrs: schema, schema_version, output_offset, and run metadata
        motif_names                 UTF-8 [M]
        motif_orientations          UTF-8 [M]  "+" or "-" (reverse complement)
        motif_sequences             UTF-8 [M]
        control_sequences           UTF-8 [M]
        insertion_starts            int64 [M]
        dna/background              int8  [B, L]
        dna/motif                   int8  [M, B, L]
        dna/control                 int8  [M, B, L]
        samples/background          int8  [B, R, O]
        samples/motif               int8  [M, B, R, O]
        samples/control             int8  [M, B, R, O]

    With both orientations, each motif appears twice, as ``NAME(+)`` and
    ``NAME(-)``, and ``motif_sequences`` holds the sequence as inserted.

    DNA and raw samples are retained so downstream plotting can choose its own
    read aggregation, summary statistic, and interval. Sample datasets are
    compressed and chunked by motif/background for efficient partial reads.
    """

    total = (1 + 2 * design.n_motifs) * design.n_backgrounds
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "smfnet.marginalizations"
        handle.attrs["schema_version"] = 1
        handle.attrs["num_motifs"] = design.n_motifs
        handle.attrs["num_backgrounds"] = design.n_backgrounds
        handle.attrs["input_length"] = design.sequence_length
        if metadata is not None:
            for key, value in metadata.items():
                handle.attrs[key] = str(value) if isinstance(value, str) else value

        _write_strings(handle, "motif_names", [motif.name for motif in design.motifs])
        _write_strings(handle, "motif_orientations", list(design.orientations))
        _write_strings(
            handle,
            "motif_sequences",
            [motif.sequence for motif in design.motifs],
        )
        _write_strings(
            handle,
            "control_sequences",
            [motif.control_sequence for motif in design.motifs],
        )
        handle.create_dataset(
            "insertion_starts",
            data=design.insertion_starts.cpu().numpy(),
        )
        dna_group = handle.create_group("dna")
        _write_tensor(dna_group, "background", design.background_dna)
        _write_tensor(dna_group, "motif", design.motif_dna)
        _write_tensor(dna_group, "control", design.control_dna)

        samples_group = handle.create_group("samples")
        datasets = None
        with tqdm(
            total=total,
            desc="Marginalization",
            unit="condition/background",
            disable=not show_progress,
        ) as progress:
            for batch in batches:
                if datasets is None:
                    datasets = _sample_datasets(
                        samples_group,
                        design,
                        batch.samples.reads,
                    )
                    handle.attrs["output_length"] = batch.samples.output_length
                    handle.attrs["output_offset"] = batch.samples.output_offset

                indices = list(batch.background_indices)
                values = batch.samples.reads.cpu().to(torch.int8).numpy()
                if batch.condition == "background":
                    datasets["background"][indices] = values
                else:
                    datasets[batch.condition][batch.motif_index, indices] = values
                progress.update(len(indices))
