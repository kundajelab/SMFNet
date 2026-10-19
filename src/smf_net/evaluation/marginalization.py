"""Synthetic motif marginalization designs and streaming model sampling."""

import csv
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.resources import files
from itertools import batched, pairwise, tee
from os import PathLike
from typing import Literal

import torch

from smf_net.data import ASSAYS, DNA_MAPPING, dna_str_to_tensor, get_valid_mask
from smf_net.evaluation.regions import RegionBatch
from smf_net.evaluation.sampling import SampleBatch, SamplingConfig, sample_regions
from smf_net.model import SMFNet


@dataclass(frozen=True)
class Motif:
    """A named motif and its explicit matched control sequence."""

    name: str
    sequence: str
    control_sequence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "sequence", str(self.sequence).upper())
        object.__setattr__(
            self,
            "control_sequence",
            str(self.control_sequence).upper(),
        )
        if not self.name:
            raise ValueError("Motif name cannot be empty")
        if not self.sequence or set(self.sequence) - set("ACGT"):
            raise ValueError(f"Motif {self.name!r} must contain only A/C/G/T")
        if len(self.control_sequence) != len(self.sequence) or set(self.control_sequence) - set(
            "ACGT"
        ):
            raise ValueError(
                f"Control for motif {self.name!r} must be an A/C/G/T sequence "
                "with the same length as the motif"
            )


MotifPanel = Literal["jaspar_consensus"]
Orientation = Literal["+", "-"]


@dataclass(frozen=True)
class MotifSourceConfig:
    """Select a packaged motif panel or a user-provided motif TSV.

    The packaged ``jaspar_consensus`` panel holds JASPAR consensus motifs, in
    groups by the cell type they are expected to act in, and two composition
    checks (``group`` column of its TSV). ``names`` optionally keeps only the
    listed motifs, in the listed order.
    """

    path: str | None = None
    panel: MotifPanel = "jaspar_consensus"
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.names is not None:
            # Hydra passes lists; store a tuple so the config stays hashable.
            object.__setattr__(self, "names", tuple(str(n) for n in self.names))


@dataclass(frozen=True)
class MarginalizationConfig:
    """Controls construction of matched synthetic DNA backgrounds.

    The defaults are the settings of the SMFNet paper's marginalizations:
    uniform random bases with no added GpC dinucleotides, and every motif
    inserted in both orientations. ``base_probs`` (A, C, G, T) and
    ``n_gc_dinucleotides`` change the backgrounds' composition. "forward"
    inserts each motif as written, which halves the sampling but suits only
    models trained with reverse-complement augmentation.
    """

    n_backgrounds: int = 200
    n_gc_dinucleotides: int = 0
    base_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    insertion_start: int | None = None
    seed: int = 0
    orientations: Literal["forward", "both"] = "both"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_probs",
            tuple(float(prob) for prob in self.base_probs),
        )
        if self.orientations not in {"forward", "both"}:
            raise ValueError("orientations must be 'forward' or 'both'")
        if self.n_backgrounds < 1:
            raise ValueError("n_backgrounds must be positive")
        if self.n_gc_dinucleotides < 0:
            raise ValueError("n_gc_dinucleotides cannot be negative")
        if len(self.base_probs) != 4 or any(prob < 0 for prob in self.base_probs):
            raise ValueError("base_probs must contain four non-negative values")
        if sum(self.base_probs) <= 0:
            raise ValueError("base_probs must have a positive sum")
        if self.insertion_start is not None and self.insertion_start < 0:
            raise ValueError("insertion_start cannot be negative")


@dataclass(frozen=True)
class MarginalizationDesign:
    """Full DNA for matched background, motif, and control conditions."""

    motifs: tuple[Motif, ...]
    background_dna: torch.Tensor
    motif_dna: torch.Tensor
    control_dna: torch.Tensor
    insertion_starts: torch.Tensor
    orientations: tuple[Orientation, ...]

    def __post_init__(self) -> None:
        if len(self.orientations) != len(self.motifs):
            raise ValueError("orientations must have one entry per motif")

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    @property
    def n_backgrounds(self) -> int:
        return self.background_dna.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.background_dna.shape[1]


@dataclass(frozen=True)
class MarginalizationSampleBatch:
    """One condition and motif slice of a streaming marginalization run."""

    condition: Literal["background", "motif", "control"]
    motif_index: int | None
    motif_name: str | None
    insertion_start: int | None
    background_indices: tuple[int, ...]
    samples: SampleBatch


@dataclass(frozen=True)
class _MarginalizationBatch:
    condition: Literal["background", "motif", "control"]
    motif_index: int | None
    motif_name: str | None
    insertion_start: int | None
    background_indices: tuple[int, ...]
    regions: RegionBatch


def read_motifs(path: str | PathLike[str]) -> list[Motif]:
    """Read ``name``, ``sequence``, and ``control_sequence`` from a TSV file."""

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"name", "sequence", "control_sequence"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Motif TSV must contain name, sequence, and control_sequence columns")
        motifs = [
            Motif(row["name"], row["sequence"], row["control_sequence"])
            for row in reader
            if row["name"].strip()
        ]
    _validate_motifs(motifs)
    return motifs


def read_builtin_motifs(name: MotifPanel = "jaspar_consensus") -> list[Motif]:
    """Read a motif panel distributed with SMFNet."""

    resource = files("smf_net.evaluation").joinpath("motifs", f"{name}.tsv")
    with resource.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        motifs = [Motif(row["name"], row["sequence"], row["control_sequence"]) for row in reader]
    _validate_motifs(motifs)
    return motifs


def select_motifs(
    motifs: Sequence[Motif],
    names: Sequence[str] | None,
) -> list[Motif]:
    """Keep the named motifs in the requested order, or all when ``names`` is None."""

    if names is None:
        return list(motifs)
    by_name = {motif.name: motif for motif in motifs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown motif names {missing}; available: {list(by_name)}")
    selected = [by_name[name] for name in names]
    _validate_motifs(selected)
    return selected


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def oriented_motifs(
    motifs: Sequence[Motif],
    orientations: Literal["forward", "both"],
) -> tuple[tuple[Motif, ...], tuple[Orientation, ...]]:
    """Return motifs as inserted, adding ``NAME(+)``/``NAME(-)`` pairs for "both"."""

    if orientations == "forward":
        return tuple(motifs), ("+",) * len(motifs)
    expanded = []
    labels: list[Orientation] = []
    for motif in motifs:
        expanded.append(Motif(f"{motif.name}(+)", motif.sequence, motif.control_sequence))
        expanded.append(
            Motif(
                f"{motif.name}(-)",
                reverse_complement(motif.sequence),
                reverse_complement(motif.control_sequence),
            )
        )
        labels.extend(["+", "-"])
    return tuple(expanded), tuple(labels)


def assay_site_counts(sequence: str, flank: str = "T") -> dict[str, int]:
    """Count each assay's eligible sites in ``sequence`` between neutral flanks.

    ``T`` qualifies as both the D and H flank of a GCH site and forms no CpG or
    GpC, so the counts reflect the sequence itself.
    """

    dna = dna_str_to_tensor(flank + sequence + flank).unsqueeze(0)
    return {assay: int(get_valid_mask(dna, assay)[:, 1:-1].sum()) for assay in ASSAYS}


def dinucleotide_shuffle(sequence: str, rng: random.Random) -> str:
    """Shuffle a sequence uniformly while preserving its dinucleotide counts.

    Altschul-Erickson: pick a random last outgoing edge per base so that those
    edges form a tree into the final base, shuffle the remaining edges, then
    walk the resulting Eulerian path from the first base.
    """

    if len(sequence) < 3:
        return sequence
    edges: dict[str, list[str]] = {}
    for left, right in pairwise(sequence):
        edges.setdefault(left, []).append(right)
    last = sequence[-1]

    while True:
        last_edges = {base: rng.choice(targets) for base, targets in edges.items() if base != last}
        if all(_reaches(base, last, last_edges) for base in last_edges):
            break

    walks = {}
    for base, targets in edges.items():
        remaining = list(targets)
        if base != last:
            remaining.remove(last_edges[base])
        rng.shuffle(remaining)
        if base != last:
            remaining.append(last_edges[base])
        walks[base] = remaining

    shuffled = [sequence[0]]
    positions = dict.fromkeys(walks, 0)
    for _ in range(len(sequence) - 1):
        base = shuffled[-1]
        shuffled.append(walks[base][positions[base]])
        positions[base] += 1
    return "".join(shuffled)


def _reaches(base: str, target: str, last_edges: dict[str, str]) -> bool:
    seen = set()
    while base != target:
        if base in seen:
            return False
        seen.add(base)
        base = last_edges[base]
    return True


def matched_controls(
    sequence: str,
    seed: int = 0,
    max_tries: int = 100_000,
) -> Iterator[str]:
    """Yield distinct shuffles with the motif's eligible site count in every assay.

    Dinucleotide-preserving shuffles come first; very short motifs often have
    few distinct ones, so plain shuffles follow. Every control differs from the
    motif in both orientations.
    """

    sequence = sequence.upper()
    target = assay_site_counts(sequence)
    seen = {sequence, reverse_complement(sequence)}
    rng = random.Random(seed)
    shuffles = (
        lambda: dinucleotide_shuffle(sequence, rng),
        lambda: "".join(rng.sample(sequence, len(sequence))),
    )
    for shuffle in shuffles:
        for _ in range(max_tries):
            candidate = shuffle()
            if candidate in seen:
                continue
            seen.add(candidate)
            if assay_site_counts(candidate) == target:
                yield candidate


def _validate_motifs(motifs: Sequence[Motif]) -> None:
    if not motifs:
        raise ValueError("At least one motif is required")
    names = [motif.name for motif in motifs]
    if len(names) != len(set(names)):
        raise ValueError("Motif names must be unique")


def _random_backgrounds(
    sequence_length: int,
    config: MarginalizationConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    probabilities = torch.tensor(config.base_probs, dtype=torch.float32)
    probabilities /= probabilities.sum()
    dna = torch.multinomial(
        probabilities,
        config.n_backgrounds * sequence_length,
        replacement=True,
        generator=generator,
    ).view(config.n_backgrounds, sequence_length)

    if config.n_gc_dinucleotides > sequence_length // 2:
        raise ValueError("n_gc_dinucleotides cannot exceed half the sequence length")
    for background in dna:
        n_gc = config.n_gc_dinucleotides
        if n_gc:
            # Choosing ordered positions from a shortened coordinate system and
            # adding their rank gives an exact, random set of non-adjacent starts.
            starts = torch.randperm(
                sequence_length - n_gc,
                generator=generator,
            )[:n_gc]
            starts = starts.sort().values + torch.arange(n_gc)
            background[starts] = DNA_MAPPING["G"]
            background[starts + 1] = DNA_MAPPING["C"]
    return dna.to(torch.int8)


def generate_marginalization_design(
    motifs: Sequence[Motif],
    sequence_length: int,
    config: MarginalizationConfig = MarginalizationConfig(),
) -> MarginalizationDesign:
    """Create matched motif and control insertions in synthetic backgrounds."""

    _validate_motifs(motifs)
    motifs, orientations = oriented_motifs(motifs, config.orientations)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    generator = torch.Generator().manual_seed(config.seed)
    background_dna = _random_backgrounds(sequence_length, config, generator)
    motif_dna = background_dna.repeat(len(motifs), 1, 1)
    control_dna = motif_dna.clone()
    insertion_starts = []

    for motif_index, motif in enumerate(motifs):
        start = (
            (sequence_length - len(motif.sequence)) // 2
            if config.insertion_start is None
            else config.insertion_start
        )
        end = start + len(motif.sequence)
        if start < 0 or end > sequence_length:
            raise ValueError(f"Motif {motif.name!r} falls outside the DNA sequence")
        motif_dna[motif_index, :, start:end] = dna_str_to_tensor(motif.sequence)
        control_dna[motif_index, :, start:end] = dna_str_to_tensor(motif.control_sequence)
        insertion_starts.append(start)

    return MarginalizationDesign(
        motifs=motifs,
        background_dna=background_dna,
        motif_dna=motif_dna,
        control_dna=control_dna,
        insertion_starts=torch.tensor(insertion_starts, dtype=torch.long),
        orientations=orientations,
    )


def _design_batches(
    design: MarginalizationDesign,
    regions_per_batch: int,
) -> Iterator[_MarginalizationBatch]:
    condition_arrays = [("background", None, design.background_dna)]
    condition_arrays.extend(
        (condition, motif_index, dna[motif_index])
        for motif_index in range(design.n_motifs)
        for condition, dna in (
            ("motif", design.motif_dna),
            ("control", design.control_dna),
        )
    )

    for condition, motif_index, dna in condition_arrays:
        motif = None if motif_index is None else design.motifs[motif_index]
        for indices in batched(range(design.n_backgrounds), regions_per_batch):
            label = "background" if motif is None else f"{motif.name}.{condition}"
            yield _MarginalizationBatch(
                condition=condition,
                motif_index=motif_index,
                motif_name=None if motif is None else motif.name,
                insertion_start=(
                    None if motif_index is None else int(design.insertion_starts[motif_index])
                ),
                background_indices=indices,
                regions=RegionBatch(
                    [f"background_{index:06d}.{label}" for index in indices],
                    dna[list(indices)],
                ),
            )


def sample_marginalization(
    model: SMFNet,
    design: MarginalizationDesign,
    config: SamplingConfig,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Iterator[MarginalizationSampleBatch]:
    """Sample a design without retaining the full predicted run in memory."""

    batches = _design_batches(design, config.regions_per_batch)
    batches_for_metadata, batches_for_sampling = tee(batches)
    samples = sample_regions(
        model,
        (batch.regions for batch in batches_for_sampling),
        config,
        total_regions=design.n_backgrounds * (1 + 2 * design.n_motifs),
        compute_dtype=compute_dtype,
    )
    for batch, sample_batch in zip(batches_for_metadata, samples, strict=True):
        yield MarginalizationSampleBatch(
            condition=batch.condition,
            motif_index=batch.motif_index,
            motif_name=batch.motif_name,
            insertion_start=batch.insertion_start,
            background_indices=batch.background_indices,
            samples=sample_batch,
        )
