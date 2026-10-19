import re
import warnings
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import h5py
import numpy as np
import torch
from pyfaidx import Fasta
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SourceKind = Literal["observed_h5", "fasta", "bed"]


@dataclass(frozen=True)
class RegionSource:
    """Files from which one sampled region was derived."""

    kind: SourceKind
    path: str
    reference_fasta_path: str | None = None


AssayType = Literal["cg_gc_accessibility", "gch_accessibility", "fiberseq_m6a"]


@dataclass(frozen=True)
class AssaySpec:
    target_rule: Literal["cg_gc", "gch", "at"]
    # Default for training-time RC augmentation. RC is only valid when a
    # dataset merges both strands evenly, which depends on how it was made:
    # the GCH bait-capture data we have is strand-biased, so GCH defaults to
    # off. Fiber-seq reads both strands of every molecule, and m6A rates at A
    # and reverse-complement T contexts agree to within split-half noise, so
    # it defaults to on. SparseDataModuleConfig.rc_augmentation overrides this
    # per dataset.
    rc_augmentation: bool


ASSAYS = MappingProxyType(
    {
        "cg_gc_accessibility": AssaySpec("cg_gc", True),
        "gch_accessibility": AssaySpec("gch", False),
        "fiberseq_m6a": AssaySpec("at", True),
    }
)


@contextmanager
def single_threaded_torch() -> Iterator[None]:
    """Run torch on one thread, restoring the previous count afterwards.

    Loading and filtering loop over regions whose tensors are far too small for
    a thread pool to pay for itself. On an 8-CPU allocation the default pool
    makes loading a split about 13x slower and the assay filter about 40x
    slower.
    """

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _get_assay_spec(assay_type: AssayType) -> AssaySpec:
    try:
        return ASSAYS[assay_type]
    except KeyError:
        raise ValueError(f"Unsupported assay type: {assay_type}") from None


@dataclass
class SMFItem:
    smf: torch.Tensor
    dna: torch.Tensor

    def pin_memory(self):
        """Required for DataLoader pin_memory=True to work correctly."""
        return SMFItem(self.smf.pin_memory(), self.dna.pin_memory())

    def to(self, *args, **kwargs) -> "SMFItem":
        return SMFItem(self.smf.to(*args, **kwargs), self.dna.to(*args, **kwargs))


@dataclass(frozen=True)
class SparseSMFRegion:
    """One DNA window and all of its observed reads in sparse-column form."""

    region_id: str | None
    dna: torch.Tensor
    smf: torch.Tensor
    positions: torch.Tensor
    window_start: int
    source: RegionSource | None = None

    def crop(self, window: slice) -> "SparseSMFRegion":
        start = window.start or 0
        end = window.stop or self.dna.shape[-1]
        keep = (self.positions >= start) & (self.positions < end)
        return SparseSMFRegion(
            region_id=self.region_id,
            dna=self.dna[start:end],
            smf=self.smf[:, keep],
            positions=self.positions[keep] - start,
            window_start=self.window_start + start,
            source=self.source,
        )

    def mean(self) -> torch.Tensor:
        mean, counts = site_means(self.smf)
        return mean.masked_fill(counts == 0, -1)


def site_means(smf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean call at each site over the reads called there, and their count.

    smf: [..., reads, sites] calls, -1 where unobserved. A site no read calls
    has a mean of 0 and a count of 0.
    """

    valid = smf != -1
    counts = valid.sum(dim=-2)
    return (smf.float() * valid).sum(dim=-2) / counts.clamp_min(1), counts


DNA_MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
REVERSE_DNA_MAPPING = {v: k for k, v in DNA_MAPPING.items()}


def dna_str_to_tensor(seq: str) -> torch.Tensor:
    """
    Convert DNA sequence string to int8 tensor with ACGT=0,1,2,3 encoding.
    -1 for unknown bases.
    """
    seq_int = [DNA_MAPPING.get(base, -1) for base in seq]
    return torch.tensor(seq_int, dtype=torch.int8)


def tensor_to_dna_str(tensor: torch.Tensor) -> str:
    """Convert int8 tensor with ACGT=0,1,2,3 encoding to DNA sequence string."""
    seq = "".join([REVERSE_DNA_MAPPING.get(int(base), "N") for base in tensor])
    return seq


def dna_tensor_to_one_hot(dna: torch.Tensor) -> torch.Tensor:
    """
    dna: bsz x L  (0–3 for A/C/G/T, -1 for N/unknown)
    returns: bsz x 4 x L
    """
    bsz, L = dna.shape
    one_hot = torch.zeros((bsz, 4, L), dtype=torch.float32, device=dna.device)

    # Valid positions are A/C/G/T
    valid = (dna >= 0) & (dna < 4)

    # Clamp so we don't feed -1 into scatter;  -1 → 0 but we'll gate with `valid`
    indices = dna.clamp(min=0).unsqueeze(1).to(torch.long)

    # src is 1 where valid base, 0 where N/unknown
    src = valid.to(one_hot.dtype).unsqueeze(1)  # shape: bsz x 1 x L

    one_hot.scatter_(1, indices, src)
    return one_hot


def compute_masks_cg_gc(dna: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return boolean masks (CG_mask, GC_mask) over positions where:
      - CG_mask: C is followed by G (C at i, G at i+1)
      - GC_mask: C is preceded by G (G at i-1, C at i)

    Works for dna of shape (..., L) with ACGT encoded as 0,1,2,3.
    """
    # C at current position
    C = dna == 1  # shape (..., L)

    # Allocate left/right G masks
    G_left = torch.zeros_like(C, dtype=torch.bool, device=dna.device)  # G at i-1
    G_right = torch.zeros_like(C, dtype=torch.bool, device=dna.device)  # G at i+1

    # Fill them along the last dimension
    # G at i-1 → shift comparison to the right
    G_left[..., 1:] = dna[..., :-1] == 2

    # G at i+1 → shift comparison to the left
    G_right[..., :-1] = dna[..., 1:] == 2

    CG_mask = C & G_right
    GC_mask = C & G_left
    return CG_mask, GC_mask


def get_valid_mask(dna: torch.Tensor, assay_type: AssayType) -> torch.Tensor:
    """Return eligible sites for integer-encoded DNA of shape (..., L).

    GCH accessibility requires DGCH (D = A/G/T, H = A/C/T), excluding
    CpG overlap on either strand. Both flanking bases must be known.

    Fiber-seq m6A sites are every A and T: the methyltransferase marks
    adenines on both strands, and a bottom-strand adenine sits opposite a
    reference T.
    """
    spec = _get_assay_spec(assay_type)
    if spec.target_rule == "cg_gc":
        cg_mask, gc_mask = compute_masks_cg_gc(dna)
        return cg_mask | gc_mask
    if spec.target_rule == "gch":
        _, gc_mask = compute_masks_cg_gc(dna)
        preceded_by_non_c_base = torch.zeros_like(gc_mask)
        preceded_by_non_c_base[..., 2:] = (
            (dna[..., :-2] >= 0) & (dna[..., :-2] < 4) & (dna[..., :-2] != DNA_MAPPING["C"])
        )
        followed_by_non_g_base = torch.zeros_like(gc_mask)
        followed_by_non_g_base[..., :-1] = (
            (dna[..., 1:] >= 0) & (dna[..., 1:] < 4) & (dna[..., 1:] != DNA_MAPPING["G"])
        )
        return gc_mask & preceded_by_non_c_base & followed_by_non_g_base
    if spec.target_rule == "at":
        return (dna == DNA_MAPPING["A"]) | (dna == DNA_MAPPING["T"])
    raise ValueError(f"Unsupported target rule: {spec.target_rule}")


def _filter_smf_columns_for_assay(
    smf: torch.Tensor,
    positions: torch.Tensor,
    dna: torch.Tensor,
    assay_type: AssayType,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep eligible positions having at least one observed call."""
    if smf.ndim != 2 or positions.ndim != 1 or smf.shape[1] != len(positions):
        raise ValueError("SMF columns must match the one-dimensional positions")
    if ((positions < 0) | (positions >= dna.shape[-1])).any():
        raise ValueError("SMF positions must fall within the DNA region")

    eligible = get_valid_mask(dna, assay_type)[positions]
    observed = (smf != -1).any(dim=0)
    keep = eligible & observed
    return smf[:, keep], positions[keep]


def filter_smf_regions_for_assay(
    regions: list[str],
    smf_list: list[torch.Tensor],
    pos_list: list[torch.Tensor],
    dna: torch.Tensor,
    sources: list[RegionSource],
    assay_type: AssayType,
) -> tuple[
    list[str],
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
    list[RegionSource],
]:
    """Filter sparse columns and discard regions without eligible observations."""
    kept_regions = []
    kept_smf = []
    kept_pos = []
    kept_dna = []
    kept_sources = []
    for region, smf, positions, region_dna, source in zip(
        regions, smf_list, pos_list, dna, sources, strict=True
    ):
        smf, positions = _filter_smf_columns_for_assay(smf, positions, region_dna, assay_type)
        if smf.shape[0] == 0 or positions.numel() == 0:
            continue
        kept_regions.append(region)
        kept_smf.append(smf)
        kept_pos.append(positions)
        kept_dna.append(region_dna)
        kept_sources.append(source)

    filtered_dna = torch.stack(kept_dna) if kept_dna else dna.new_empty((0, dna.shape[-1]))
    return kept_regions, kept_smf, kept_pos, filtered_dna, kept_sources


def reverse_complement_dna(dna: torch.Tensor) -> torch.Tensor:
    """
    dna: shape (..., L), with A,C,G,T encoded as 0,1,2,3 and -1 for unknown/N.
    Returns the reverse complement with N staying as -1.
    """
    # Flip along the last dimension (handles both 1D [L] and batched [..., L])
    dna_flipped = torch.flip(dna, dims=[-1])

    # Copy so we don't modify in-place
    rc = dna_flipped.clone()

    # Complement only valid A/C/G/T: 0↔3, 1↔2
    valid = (rc >= 0) & (rc < 4)
    rc[valid] = 3 - rc[valid]

    # Invalid bases (e.g., -1) remain unchanged
    return rc


def reverse_complement_dna_and_smf(
    dna: torch.Tensor,
    smf: torch.Tensor,
    assay_type: AssayType = "cg_gc_accessibility",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse complement a region and move each call to the opposite strand.

    dna: [L] integer-encoded DNA; smf: [R, L] calls, -1 where unobserved.

    A Fiber-seq call belongs to one base pair, so RC only flips the calls:
    an m6A on a top-strand A becomes one on the bottom-strand adenine opposite
    the T at the mirrored position. The rest of this docstring is about the
    cytosine assays, where a call moves to the other cytosine of its
    dinucleotide.

    Calls sit on top-strand cytosines, and a CpG or GpC call is moved to the
    cytosine of the same dinucleotide on the other strand: a CpG call one base
    to the right and a GpC call one base to the left, before flipping. RC can
    then be read as "the same molecule, measured from the other strand".

    The h5 files already merge reads from both strands this way, so a
    forward-strand read here looks just like a reverse-strand read did to the
    pipeline that built them. Two contexts map unevenly, and both are handled
    as that pipeline handles reverse-strand reads:

    - A forward GCG has one cytosine in two dinucleotides. Its call is copied
      to both cytosines of the RC CGC, as a reverse-strand GCG call is copied
      to both cytosines of the top-strand CGC.
    - A forward CGC has two cytosines whose partners are one cytosine, the
      middle of the RC GCG. It takes the GpC call, or the CpG call for reads
      where the GpC call is missing.

    In observed data both cytosines of a CGC almost always agree (99.4% of
    reads in an ONT mESC sample), largely because merged reverse-strand reads
    give them identical calls, so little is lost by keeping one.
    """
    assert dna.dim() == 1 and smf.dim() == 2 and smf.shape[1] == dna.shape[0]
    dna_rc = reverse_complement_dna(dna)  # [L]
    smf_flip = torch.flip(smf, dims=[-1])  # [R, L]
    if _get_assay_spec(assay_type).target_rule == "at":
        return dna_rc, smf_flip
    _R, L = smf.shape
    smf_out = torch.full_like(smf, -1)

    CG_mask, GC_mask = compute_masks_cg_gc(dna_rc)  # [L], [L]

    # GC: take from i-1
    idxs = torch.nonzero(GC_mask).flatten()
    if idxs.numel():
        src = idxs - 1
        valid = src >= 0
        smf_out[:, idxs[valid]] = smf_flip[:, src[valid]]

    # CG: take from i+1, keeping GpC calls at a GCG cytosine
    idxs = torch.nonzero(CG_mask).flatten()
    if idxs.numel():
        src = idxs + 1
        valid = src < L
        idxs, src = idxs[valid], src[valid]
        current = smf_out[:, idxs]
        smf_out[:, idxs] = torch.where(current == -1, smf_flip[:, src], current)

    return dna_rc, smf_out


def densify_smf(smf: torch.Tensor, pos: torch.Tensor, region_len: int) -> torch.Tensor:
    """
    smf: (num_reads, num_smf_pos)
    pos: (num_smf_pos,)

    returns: (num_reads, region_len)
    """
    expanded = torch.full((smf.shape[0], region_len), -1.0, dtype=smf.dtype)
    expanded[:, pos] = smf
    return expanded


class SparseSMFDataset(Dataset):
    def __init__(
        self,
        smf_list: list[torch.Tensor],
        pos_list: list[torch.Tensor],
        dna: torch.Tensor,
        regions: list[str] | None = None,
        sources: list[RegionSource] | None = None,
        reads_per_region: int = 32,
        train: bool = False,
        model_input_length: int = 2048,
        assay_type: AssayType = "cg_gc_accessibility",
        rc_augmentation: bool | None = None,
    ):
        """
        DNA: N x region_len, int8 encoding
        smf_list: list of (num_reads, num_smf_pos) tensors
        pos_list: list of (num_smf_pos,) tensors with valid SMF positions

        Dataset for sparse SMF data with corresponding DNA sequences.
        Output shape of DNA is (4, model_input_length)
        Output shape of SMF is (reads_per_region, model_input_length)
        The assay determines valid positions. Training applies random
        reverse-complement augmentation if rc_augmentation is set, or by the
        assay's default when it is None.
        """
        self.assay_type = assay_type
        if rc_augmentation is None:
            rc_augmentation = _get_assay_spec(assay_type).rc_augmentation
        self.rc_augmentation = rc_augmentation
        if not (len(smf_list) == len(dna) == len(pos_list)):
            raise ValueError("Length of smf_list, dna, and pos_list must be the same")
        if regions is not None and len(regions) != len(smf_list):
            raise ValueError("regions must match the number of SMF regions")
        if sources is not None and len(sources) != len(smf_list):
            raise ValueError("sources must match the number of SMF regions")

        filtered = [
            _filter_smf_columns_for_assay(smf, positions, region_dna, assay_type)
            for smf, positions, region_dna in zip(smf_list, pos_list, dna, strict=True)
        ]
        for idx, (smf, positions) in enumerate(filtered):
            if smf.shape[0] == 0 or positions.numel() == 0:
                region = idx if regions is None else regions[idx]
                raise ValueError(
                    f"Region {region!r} has no eligible observed SMF positions "
                    f"for assay {assay_type!r}"
                )

        self.smf = [smf for smf, _ in filtered]
        self.dna = dna
        self.pos_list = [positions for _, positions in filtered]
        self.regions = regions
        self.sources = sources
        self.reads_per_region = reads_per_region
        self.train = train
        self.region_len = dna.shape[1]
        if model_input_length < 1:
            raise ValueError("model_input_length must be positive")
        if model_input_length > self.region_len:
            raise ValueError(
                "model_input_length cannot exceed the source region length, got "
                f"{model_input_length} and {self.region_len}"
            )
        self.model_input_length = model_input_length
        self.jitter = self.region_len - self.model_input_length
        self.num_reads = [smf.shape[0] for smf in self.smf]

    def __len__(self) -> int:
        return len(self.pos_list)

    def get_region(self, idx: int, start: int | None = None) -> SparseSMFRegion:
        """Return all observed reads for one cropped region without densifying."""
        if start is None:
            start = self.region_len // 2 - self.model_input_length // 2
        end = start + self.model_input_length
        positions = self.pos_list[idx]
        keep = (positions >= start) & (positions < end)
        dna = self.dna[idx, start:end]
        # Cropping can remove a flank that makes a site eligible.
        kept_indices = torch.nonzero(keep).flatten()
        cropped_positions = positions[kept_indices] - start
        cropped_valid = get_valid_mask(dna, self.assay_type)[cropped_positions]
        keep[kept_indices] = cropped_valid
        region_id = None if self.regions is None else self.regions[idx]
        return SparseSMFRegion(
            region_id=region_id,
            dna=dna,
            smf=self.smf[idx][:, keep],
            positions=positions[keep] - start,
            window_start=start,
            source=None if self.sources is None else self.sources[idx],
        )

    def iter_regions(self) -> Iterator[SparseSMFRegion]:
        for idx in range(len(self)):
            yield self.get_region(idx)

    def get_item(
        self,
        idx: int,
        read_idx: torch.Tensor,
        start: int,
        reverse_complement: bool = False,
    ) -> SMFItem:
        """
        idx: index of region
        read_idx: tensor of shape (reads_per_region,) with indices of reads to select
        start: starting position in the output region (allows for jitter)
        reverse_complement: whether to apply reverse complement to the DNA and
            SMF, regardless of rc_augmentation (e.g. for test-time averaging)
        """
        region = self.get_region(idx, start)
        dna = region.dna
        smf = densify_smf(region.smf[read_idx], region.positions, self.model_input_length)
        if reverse_complement:
            dna, smf = reverse_complement_dna_and_smf(dna, smf, self.assay_type)
        return SMFItem(smf=smf, dna=dna)

    def __getitem__(self, idx: int) -> SMFItem:
        num_reads = self.num_reads[idx]

        if not self.train:
            generator = torch.Generator()
            generator.manual_seed(idx)
            start = self.region_len // 2 - self.model_input_length // 2
            rc = False
        else:
            generator = None
            start = torch.randint(0, self.jitter + 1, (1,)).item()
            rc = self.rc_augmentation and torch.rand(1).item() < 0.5

        read_idx = self._sample_uniform_read_idx(num_reads=num_reads, generator=generator)
        return self.get_item(idx, read_idx, start, reverse_complement=rc)

    def _sample_uniform_read_idx(
        self, num_reads: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.multinomial(
            torch.ones(num_reads),
            self.reads_per_region,
            replacement=num_reads < self.reads_per_region,
            generator=generator,
        )

    @staticmethod
    def collate_fn(batch: list[SMFItem]) -> SMFItem:
        """
        batch: list of SMFItem
        returns: CNN-ready SMFItem
        SMFItem with batched smf and dna tensors
        smf is (bsz, reads_per_region, L)
        dna is bsz x 4 x L
        """
        smf_batch = torch.stack([item.smf for item in batch], dim=0)
        dna_batch = torch.stack([item.dna for item in batch], dim=0)
        return SMFItem(smf=smf_batch, dna=dna_tensor_to_one_hot(dna_batch))


def parse_region_string(region: str) -> tuple[str, int, int]:
    match = re.match(r"^(chr[\w]+)[_:](\d+)[-_](\d+)$", region)
    if not match:
        raise ValueError(f"Region string not in expected format: {region}")
    chrom, start, end = match.groups()
    return chrom, int(start), int(end)


def load_dna(fasta_path: str, regions: list[str], region_length: int) -> torch.Tensor:
    """Load the reference DNA of regions named ``chrom:start-end``.

    Starts are 1-based. Ends come in two conventions, told apart by
    region_length: exclusive (end - start == region_length, as in the ONT and
    Fiber-seq files) or inclusive (end - start + 1 == region_length, as in
    the GCH files).
    """
    fasta = Fasta(fasta_path)
    dna_list = []
    for region in tqdm(regions, desc="Loading DNA sequences"):
        chrom, start, end = parse_region_string(region)
        if end - start + 1 == region_length:
            end += 1  # inclusive end
        elif end - start != region_length:
            raise ValueError(
                f"Region {region} does not span {region_length} bases with "
                "either an inclusive or an exclusive end"
            )
        # pyfaidx slices are 0-based, so shift the 1-based start and end.
        seq = fasta[chrom][start - 1 : end - 1].seq.upper()
        dna_list.append(dna_str_to_tensor(seq))
    return torch.stack(dna_list, dim=0)


def extract_smf_and_pos(
    h5_path: str,
    region_length: int = 2114,
    max_regions_per_h5_per_split: int | None = None,
    chromosomes: Collection[str] | None = None,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[RegionSource]]:
    smf_list = []
    pos_list = []
    selected_regions = []
    sources = []
    source = RegionSource("observed_h5", str(Path(h5_path).resolve()))
    with h5py.File(h5_path, "r") as h5_file:
        smf_pos_group = h5_file["smf_pos"]
        smf_mat_group = h5_file["smf_mat"]
        regions = list(smf_mat_group.keys())
        if chromosomes is not None:
            chromosomes = set(chromosomes)
            regions = [
                region for region in regions if parse_region_string(region)[0] in chromosomes
            ]

        # Deterministically shuffle even when loading every region so early
        # validation batches are not biased by the HDF5 key order.
        num_regions = min(max_regions_per_h5_per_split or len(regions), len(regions))
        rng = np.random.default_rng(1231241)
        regions = rng.choice(regions, num_regions, replace=False)
        for region in tqdm(regions, desc=f"Loading SMF data from {h5_path}"):
            pos = torch.tensor(smf_pos_group[region][:], dtype=torch.long)
            smf = torch.tensor(smf_mat_group[region][:], dtype=torch.int8)
            if len(pos) != smf.shape[1]:
                print(
                    f"Skipping region {region} due to mismatched pos/mat: pos len = {len(pos)}, mat cols = {smf.shape[1]}"
                )
                continue
            valid_idx = (pos >= 0) & (pos < region_length)
            smf_list.append(smf[:, valid_idx])
            pos_list.append(pos[valid_idx])
            selected_regions.append(region)
            sources.append(source)
    return selected_regions, smf_list, pos_list, sources


def bernoulli_thin_reads(
    regions: list[str],
    smf_list: list[torch.Tensor],
    pos_list: list[torch.Tensor],
    sources: list[RegionSource],
    keep_probability: float,
    generator: torch.Generator,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[RegionSource]]:
    """Independently retain reads and discard regions with no retained reads."""
    if not 0 < keep_probability <= 1:
        raise ValueError("keep_probability must be in (0, 1]")
    if keep_probability == 1:
        return regions, smf_list, pos_list, sources

    retained_regions = []
    retained_smf = []
    retained_pos = []
    retained_sources = []
    for region, smf, pos, source in zip(regions, smf_list, pos_list, sources, strict=True):
        keep = torch.rand(smf.shape[0], generator=generator) < keep_probability
        if not keep.any():
            continue
        retained_regions.append(region)
        retained_smf.append(smf[keep])
        retained_pos.append(pos)
        retained_sources.append(source)

    return retained_regions, retained_smf, retained_pos, retained_sources


DEFAULT_TRAIN_CHRM = [
    "chr2",
    "chr4",
    "chr5",
    "chr7",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr17",
    "chr18",
    "chr19",
    "chrX",
    "chrY",
]
DEFAULT_VAL_CHRM = ["chr8", "chr16"]
DEFAULT_TEST_CHRM = ["chr1", "chr3", "chr6"]


@dataclass
class SparseDataModuleConfig:
    fasta_path: str
    h5_path: str | list[str]
    assay_type: AssayType = "cg_gc_accessibility"
    batch_size: int = 256
    num_workers: int = 4
    train_chrm: list[str] = field(default_factory=lambda: DEFAULT_TRAIN_CHRM.copy())
    val_chrm: list[str] = field(default_factory=lambda: DEFAULT_VAL_CHRM.copy())
    test_chrm: list[str] = field(default_factory=lambda: DEFAULT_TEST_CHRM.copy())
    reads_per_region: int = 32
    max_regions_per_h5_per_split: int | None = None
    h5_region_length: int = 2114
    model_input_length: int = 2048
    train_val_read_fraction: float = 1.0
    read_downsample_seed: int = 0
    # Random RC of training items; None uses the assay's default.
    rc_augmentation: bool | None = None

    def __post_init__(self) -> None:
        _get_assay_spec(self.assay_type)
        if self.model_input_length < 1:
            raise ValueError("model_input_length must be positive")
        if self.model_input_length > self.h5_region_length:
            raise ValueError(
                "model_input_length cannot exceed h5_region_length, got "
                f"{self.model_input_length} and {self.h5_region_length}"
            )
        if not 0 < self.train_val_read_fraction <= 1:
            raise ValueError("train_val_read_fraction must be in (0, 1]")
        if self.max_regions_per_h5_per_split is not None and self.max_regions_per_h5_per_split <= 0:
            raise ValueError(
                "max_regions_per_h5_per_split must be positive or None, got "
                f"{self.max_regions_per_h5_per_split}"
            )

    @property
    def h5_paths(self) -> list[str]:
        """The configured h5 inputs, whether given as one path or several."""
        if isinstance(self.h5_path, str):
            return [self.h5_path]
        try:
            return list(self.h5_path)
        except TypeError:
            raise ValueError(f"Bad type for {self.h5_path=}, {type(self.h5_path)=}") from None


def validate_data_files(cfg: SparseDataModuleConfig) -> None:
    """
    Check the inputs exist before loading regions, which takes minutes.

    Also warns when pyfaidx would silently (re)build the fasta index, since
    that is another multi-minute delay with no output of its own.
    """
    missing = [path for path in [cfg.fasta_path, *cfg.h5_paths] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing data files: " + ", ".join(repr(path) for path in missing))

    fasta = Path(cfg.fasta_path)
    index = fasta.with_name(fasta.name + ".fai")
    if not index.exists():
        warnings.warn(
            f"No fasta index at {str(index)!r}; pyfaidx will build one, which "
            "can take several minutes for a large genome.",
            stacklevel=2,
        )
    elif index.stat().st_mtime < fasta.stat().st_mtime:
        warnings.warn(
            f"Fasta index {str(index)!r} is older than {str(fasta)!r}; pyfaidx "
            "treats it as stale and will rebuild it.",
            stacklevel=2,
        )


class SparseSMFDataModule(LightningDataModule):
    def __init__(self, cfg: SparseDataModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset: SparseSMFDataset | None = None
        self.val_dataset: SparseSMFDataset | None = None
        self.test_dataset: SparseSMFDataset | None = None

    def load_smf_from_h5(
        self, chromosomes: Collection[str]
    ) -> tuple[
        list[str],
        list[torch.Tensor],
        list[torch.Tensor],
        list[RegionSource],
    ]:
        regions, smf_list, pos_list, sources = [], [], [], []
        for h5_path in self.cfg.h5_paths:
            r, s, p, region_sources = extract_smf_and_pos(
                h5_path,
                region_length=self.cfg.h5_region_length,
                max_regions_per_h5_per_split=self.cfg.max_regions_per_h5_per_split,
                chromosomes=chromosomes,
            )
            regions += r
            smf_list += s
            pos_list += p
            sources += region_sources

        return regions, smf_list, pos_list, sources

    def _setup_split(self, split: str, train: bool) -> None:
        print(f"Preparing split: {split}")
        dataset_attr = f"{split}_dataset"
        if getattr(self, dataset_attr) is not None:
            return

        chromosomes = getattr(self.cfg, f"{split}_chrm")
        regions, mats, pos, sources = self.load_smf_from_h5(chromosomes)

        if split in {"train", "val"} and self.cfg.train_val_read_fraction < 1:
            had_regions = bool(regions)
            generator = torch.Generator().manual_seed(
                self.cfg.read_downsample_seed + (0 if split == "train" else 1)
            )
            regions, mats, pos, sources = bernoulli_thin_reads(
                regions,
                mats,
                pos,
                sources,
                keep_probability=self.cfg.train_val_read_fraction,
                generator=generator,
            )
            if had_regions and not regions:
                raise ValueError(f"Read downsampling removed every region from the {split} split")

        if regions:
            dna = load_dna(self.cfg.fasta_path, regions, region_length=self.cfg.h5_region_length)
        else:
            dna = torch.empty((0, self.cfg.h5_region_length), dtype=torch.int8)

        if dna.shape[1] != self.cfg.h5_region_length:
            raise ValueError(
                f"Expected {self.cfg.h5_region_length} DNA bases per region, "
                f"got {dna.shape[1]} for the {split} split"
            )

        if regions:
            num_regions = len(regions)
            regions, mats, pos, dna, sources = filter_smf_regions_for_assay(
                regions,
                mats,
                pos,
                dna,
                sources,
                self.cfg.assay_type,
            )
            dropped = num_regions - len(regions)
            if dropped:
                print(
                    f"Dropped {dropped} {split} regions without eligible observed "
                    f"positions for assay {self.cfg.assay_type!r}"
                )
            if not regions:
                raise ValueError(f"Assay filtering removed every region from the {split} split")

        dataset = SparseSMFDataset(
            mats,
            pos,
            dna,
            regions=regions,
            sources=None if train else sources,
            train=train,
            reads_per_region=self.cfg.reads_per_region,
            model_input_length=self.cfg.model_input_length,
            assay_type=self.cfg.assay_type,
            rc_augmentation=self.cfg.rc_augmentation,
        )
        setattr(self, dataset_attr, dataset)

    def setup(self, stage: str | None = None) -> None:
        # Setup loops over regions one small tensor at a time, which a thread
        # pool only slows down. The previous thread count is restored for
        # training itself.
        with single_threaded_torch():
            if stage in ("fit", None):
                self._setup_split("train", train=True)
                self._setup_split("val", train=False)
            if stage == "validate":
                self._setup_split("val", train=False)
            if stage in ("test", None):
                self._setup_split("test", train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=SparseSMFDataset.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=SparseSMFDataset.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=SparseSMFDataset.collate_fn,
            pin_memory=True,
        )
