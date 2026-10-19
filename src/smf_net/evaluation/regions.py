"""Named DNA region sources shared by sampling and attribution workflows."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from os import PathLike
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from pyfaidx import Fasta

from smf_net.data import RegionSource, SparseSMFDataModule, dna_str_to_tensor


@dataclass(frozen=True)
class RegionInputConfig:
    """Region source shared by inference command-line workflows."""

    source: Literal["split", "fasta", "bed"] = "split"
    split: Literal["val", "test"] = "val"
    fasta_path: str | None = None
    bed_path: str | None = None
    reference_fasta: str | None = None


@dataclass(frozen=True)
class SequenceRegion:
    region_id: str
    dna: torch.Tensor
    source: RegionSource | None = None


class SequenceRegionSource(Iterator[SequenceRegion]):
    """One-pass region source with a known total for progress reporting."""

    def __init__(self, regions: Iterator[SequenceRegion], total_regions: int) -> None:
        self._regions = regions
        self.total_regions = total_regions

    def __next__(self) -> SequenceRegion:
        return next(self._regions)


@dataclass(frozen=True)
class RegionBatch:
    """A batch of named, integer-encoded DNA sequences."""

    region_ids: tuple[str, ...]
    dna: torch.Tensor
    sources: tuple[RegionSource | None, ...]

    def __init__(
        self,
        region_ids: Sequence[str],
        dna: torch.Tensor,
        sources: Sequence[RegionSource | None] | None = None,
    ) -> None:
        if dna.ndim != 2:
            raise ValueError(f"DNA must have shape [regions, length], got {tuple(dna.shape)}")
        if dna.shape[0] != len(region_ids):
            raise ValueError(
                f"Got {len(region_ids)} region IDs for a DNA batch of size {dna.shape[0]}"
            )
        if dna.dtype not in {torch.int8, torch.int, torch.long}:
            raise TypeError(f"DNA must be integer encoded, got dtype {dna.dtype}")
        if sources is not None and len(sources) != len(region_ids):
            raise ValueError("sources must match the number of region IDs")

        object.__setattr__(self, "region_ids", tuple(region_ids))
        object.__setattr__(self, "dna", dna)
        object.__setattr__(
            self,
            "sources",
            tuple(sources) if sources is not None else (None,) * len(region_ids),
        )

    @property
    def batch_size(self) -> int:
        return len(self.region_ids)

    @property
    def sequence_length(self) -> int:
        return self.dna.shape[-1]


def batch_regions(
    regions: Iterable[SequenceRegion],
    batch_size: int,
) -> Iterator[RegionBatch]:
    for group in batched(regions, batch_size):
        yield RegionBatch(
            region_ids=[region.region_id for region in group],
            dna=torch.stack([region.dna for region in group]),
            sources=[region.source for region in group],
        )


def _sequence_length(
    lengths: Iterable[int],
    requested: int | None,
    source: str,
) -> int:
    if requested is not None:
        return requested
    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise ValueError(
            f"{source} contains multiple sequence lengths {unique_lengths}; "
            "provide sequence_length to resize them"
        )
    return unique_lengths[0]


def _resize_sequence(sequence: str, length: int) -> str:
    if len(sequence) > length:
        start = (len(sequence) - length) // 2
        return sequence[start : start + length]
    if len(sequence) < length:
        left = (length - len(sequence)) // 2
        return "N" * left + sequence + "N" * (length - len(sequence) - left)
    return sequence


def regions_from_fasta(
    fasta_path: str | PathLike[str],
    sequence_length: int | None = None,
) -> SequenceRegionSource:
    fasta_path = str(Path(fasta_path).resolve())
    source = RegionSource("fasta", fasta_path)
    fasta = Fasta(fasta_path)
    try:
        sequences = [(record.name, str(record[:]).upper()) for record in fasta]
    finally:
        fasta.close()
    length = _sequence_length(
        (len(sequence) for _, sequence in sequences),
        sequence_length,
        "FASTA input",
    )
    regions = (
        SequenceRegion(
            region_id=region_id,
            dna=dna_str_to_tensor(_resize_sequence(sequence, length)),
            source=source,
        )
        for region_id, sequence in sequences
    )
    return SequenceRegionSource(regions, total_regions=len(sequences))


def _resize_interval(start: int, end: int, length: int) -> tuple[int, int]:
    center = (start + end) // 2
    start = max(0, center - length // 2)
    return start, start + length


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def regions_from_bed(
    bed_path: str | PathLike[str],
    reference_fasta: str | PathLike[str],
    sequence_length: int | None = None,
) -> SequenceRegionSource:
    """Regions from a BED file, resized about their centers to sequence_length.

    Region IDs are ``chrom:start-end`` in BED's 0-based, end-exclusive
    coordinates, after resizing. Minus-strand regions (column 6) are
    reverse complemented.
    """
    bed_path = str(Path(bed_path).resolve())
    reference_fasta = str(Path(reference_fasta).resolve())
    source = RegionSource(
        "bed",
        bed_path,
        reference_fasta_path=reference_fasta,
    )
    bed = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    length = _sequence_length(
        (int(row[2]) - int(row[1]) for row in bed.itertuples(index=False, name=None)),
        sequence_length,
        "BED input",
    )

    def iter_regions() -> Iterator[SequenceRegion]:
        genome = Fasta(str(reference_fasta))
        try:
            for row in bed.itertuples(index=False, name=None):
                chromosome, start, end = str(row[0]), int(row[1]), int(row[2])
                strand = str(row[5]) if len(row) > 5 and pd.notna(row[5]) else "+"
                if end - start != length:
                    start, end = _resize_interval(start, end, length)
                sequence = genome[chromosome][start:end].seq.upper()
                if strand == "-":
                    sequence = _reverse_complement(sequence)
                yield SequenceRegion(
                    region_id=f"{chromosome}:{start}-{end}",
                    dna=dna_str_to_tensor(sequence),
                    source=source,
                )
        finally:
            genome.close()

    return SequenceRegionSource(iter_regions(), total_regions=len(bed))


def regions_from_split(
    data: SparseSMFDataModule,
    split: Literal["train", "val", "test"],
) -> Iterator[SequenceRegion]:
    stage = {"train": "fit", "val": "validate", "test": "test"}[split]
    data.setup(stage)
    dataset = getattr(data, f"{split}_dataset")
    for region in dataset.iter_regions():
        if region.region_id is None:
            raise ValueError("Datamodule regions must include region IDs")
        yield SequenceRegion(region.region_id, region.dna, region.source)
