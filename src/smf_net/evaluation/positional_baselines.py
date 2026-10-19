"""Sequence-blind train-set baselines for the profile and covariation metrics.

Every region in a data split shares one length and is loosely anchored on the
features it was selected for, so where a site sits in the region already
predicts some of its accessibility and some of its coupling to other sites.
These baselines turn that into a prediction for any site layout, which is the
reference a sequence model has to beat:

* the profile baseline is the train-set mean call as a smooth function of
  position in the region;
* the pair baseline is the train-set mean pair correlation as a smooth function
  of the pair's separation and of its midpoint's distance from the region
  centre, which carries the generic nucleosome phasing.

Both average the whole training set, so unlike the model and replicate samples
the metrics draw they carry no sampling noise of their own. The covariation
metrics therefore also score the model from every one of its reads, and score
residual agreement after subtracting the pair baseline, which only
region-specific coupling can earn.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from os import PathLike

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from smf_net.data import (
    SparseDataModuleConfig,
    SparseSMFDataModule,
    filter_smf_regions_for_assay,
    load_dna,
    parse_region_string,
    single_threaded_torch,
    site_means,
)
from smf_net.evaluation.covariation import PairIndex, pairwise_correlation


@dataclass(frozen=True)
class PositionalBaselineSettings:
    """Smoothing and subsampling for the train-set baseline fit.

    The defaults come from a validation sweep. The profile fit was flat
    between 0 and 100 bp of bandwidth, so the exact value hardly matters. The
    pair fit wanted almost no smoothing along separation, because correlation
    changes fastest near zero separation and wider kernels blur the nucleosome
    phasing. A 10,000-region subset fits the pair grid as well as the full
    training set does, at a fraction of the cost.
    """

    profile_bandwidth_bp: float = 50.0
    separation_bandwidth_bp: float = 2.0
    midpoint_bandwidth_bp: float = 50.0
    max_pair_regions: int | None = 10_000
    min_joint_reads: int = 5
    seed: int = 0


@dataclass(frozen=True)
class PositionalBaselines:
    """Fitted baselines, indexed by position in the full source region.

    ``pair_grid[separation, offset]`` is the mean pair correlation for pairs
    ``separation`` bp apart whose midpoint sits ``offset`` bp from the centre,
    rounded down. Cells no training pair reached are NaN.
    """

    profile: np.ndarray
    pair_grid: np.ndarray
    region_length: int
    assay_type: str
    train_chromosomes: tuple[str, ...]
    n_profile_regions: int
    n_pair_regions: int
    settings: PositionalBaselineSettings

    def profile_at(self, positions: torch.Tensor) -> torch.Tensor:
        """Baseline mean call at full-region positions."""

        self._check_positions(positions)
        return torch.from_numpy(self.profile[positions.numpy()])

    def pairs_at(self, positions: torch.Tensor, index: PairIndex) -> torch.Tensor:
        """Baseline pair correlation for every pair in ``index``.

        ``positions`` are full-region coordinates in the order ``index`` was
        built from.
        """

        self._check_positions(positions)
        separation, offset = pair_coordinates(
            positions,
            index.rows,
            index.cols,
            self.region_length,
        )
        return torch.from_numpy(self.pair_grid[separation, offset])

    def _check_positions(self, positions: torch.Tensor) -> None:
        if positions.numel() and (
            int(positions.min()) < 0 or int(positions.max()) >= self.region_length
        ):
            raise ValueError(
                "Positions fall outside the "
                f"{self.region_length} bp region the baselines were fit on"
            )

    def save(self, path: str | PathLike[str]) -> None:
        """Write the fit and its provenance to an ``.npz`` for inspection."""

        np.savez_compressed(
            path,
            profile=self.profile,
            pair_grid=self.pair_grid,
            metadata=json.dumps(
                {
                    "region_length": self.region_length,
                    "assay_type": self.assay_type,
                    "train_chromosomes": list(self.train_chromosomes),
                    "n_profile_regions": self.n_profile_regions,
                    "n_pair_regions": self.n_pair_regions,
                    "settings": asdict(self.settings),
                }
            ),
        )


def pair_coordinates(
    positions: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    region_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Separation and midpoint distance from the region centre, both in bp."""

    first, second = positions[rows], positions[cols]
    separation = (second - first).abs()
    # Twice the midpoint's offset from the centre is an integer, so take its
    # size before halving.
    offset = torch.div(
        (first + second - (region_length - 1)).abs(),
        2,
        rounding_mode="floor",
    )
    return separation.numpy(), offset.numpy()


def _site_means(reads: torch.Tensor) -> np.ndarray:
    """Per-site means, NaN where no read calls the site."""

    means, counts = site_means(reads)
    return torch.where(counts > 0, means, torch.nan).numpy()


def _kernel_mean(
    sums: np.ndarray,
    counts: np.ndarray,
    bandwidth: float,
    axis: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if bandwidth <= 0:
        return sums, counts
    return (
        gaussian_filter1d(sums, bandwidth, axis=axis, mode=mode),
        gaussian_filter1d(counts, bandwidth, axis=axis, mode=mode),
    )


def _ratio(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    # Smoothing the sums and counts separately and dividing is the
    # Nadaraya-Watson estimate, so sparsely supported cells borrow from their
    # neighbours instead of being averaged as if every cell were equally full.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 1e-9, sums / counts, np.nan).astype(np.float32)


def fit_profile(
    reads: Sequence[torch.Tensor],
    positions: Sequence[torch.Tensor],
    region_length: int,
    bandwidth_bp: float,
) -> np.ndarray:
    """Smoothed train-set mean call per region position.

    Each region's site contributes its mean once, so a deeply read region does
    not outweigh the others, matching the per-region scoring.
    """

    sums = np.zeros(region_length)
    counts = np.zeros(region_length)
    for region_reads, region_positions in zip(reads, positions, strict=True):
        means = _site_means(region_reads)
        keep = np.isfinite(means)
        sites = region_positions.numpy()[keep]
        sums += np.bincount(sites, weights=means[keep], minlength=region_length)
        counts += np.bincount(sites, minlength=region_length)
    return _ratio(*_kernel_mean(sums, counts, bandwidth_bp, axis=0, mode="constant"))


def fit_pair_grid(
    reads: Sequence[torch.Tensor],
    positions: Sequence[torch.Tensor],
    region_length: int,
    settings: PositionalBaselineSettings,
    progress: bool = True,
) -> np.ndarray:
    """Smoothed train-set mean pair correlation on a separation x offset grid."""

    shape = (region_length, region_length // 2 + 1)
    sums = np.zeros(shape[0] * shape[1])
    counts = np.zeros_like(sums)
    for region_reads, region_positions in tqdm(
        list(zip(reads, positions, strict=True)),
        desc="fitting pair baseline",
        unit="region",
        disable=not progress,
    ):
        correlation, joint_reads = pairwise_correlation(region_reads)
        rows, cols = torch.triu_indices(
            region_positions.numel(),
            region_positions.numel(),
            offset=1,
        )
        values = correlation[rows, cols]
        keep = torch.isfinite(values) & (joint_reads[rows, cols] >= settings.min_joint_reads)
        separation, offset = pair_coordinates(
            region_positions,
            rows[keep],
            cols[keep],
            region_length,
        )
        cells = separation * shape[1] + offset
        sums += np.bincount(cells, weights=values[keep].numpy(), minlength=sums.size)
        counts += np.bincount(cells, minlength=counts.size)

    sums = sums.reshape(shape)
    counts = counts.reshape(shape)
    # The offset axis is a distance from the centre, so it reflects at zero.
    sums, counts = _kernel_mean(
        sums, counts, settings.midpoint_bandwidth_bp, axis=1, mode="reflect"
    )
    sums, counts = _kernel_mean(
        sums, counts, settings.separation_bandwidth_bp, axis=0, mode="constant"
    )
    return _ratio(sums, counts)


def fit_positional_baselines(
    reads: Sequence[torch.Tensor],
    positions: Sequence[torch.Tensor],
    region_length: int,
    assay_type: str,
    train_chromosomes: Collection[str],
    settings: PositionalBaselineSettings = PositionalBaselineSettings(),
    progress: bool = True,
) -> PositionalBaselines:
    """Fit both baselines from train regions in full-region coordinates."""

    if not reads:
        raise ValueError("Positional baselines need at least one training region")

    # Pair correlations cost far more per region than site means, so only the
    # pair fit is subsampled.
    pair_indices = np.arange(len(reads))
    if settings.max_pair_regions is not None and len(reads) > settings.max_pair_regions:
        rng = np.random.default_rng(settings.seed)
        pair_indices = np.sort(rng.choice(len(reads), settings.max_pair_regions, replace=False))

    with single_threaded_torch():
        profile = fit_profile(reads, positions, region_length, settings.profile_bandwidth_bp)
        pair_grid = fit_pair_grid(
            [reads[i] for i in pair_indices],
            [positions[i] for i in pair_indices],
            region_length,
            settings,
            progress=progress,
        )

    return PositionalBaselines(
        profile=profile,
        pair_grid=pair_grid,
        region_length=region_length,
        assay_type=assay_type,
        train_chromosomes=tuple(train_chromosomes),
        n_profile_regions=len(reads),
        n_pair_regions=len(pair_indices),
        settings=settings,
    )


# --------------------------------------------------------------------------
# Training data behind a samples file


def data_config_from_samples(path: str | PathLike[str]) -> SparseDataModuleConfig:
    """Rebuild the data configuration a split-sampled samples file came from.

    Only the recorded ``data_config`` is trusted. Files without one are
    refused rather than guessing their chromosome split, which risks fitting
    on the regions being scored.
    """

    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)

    if attrs.get("source") != "split":
        raise ValueError(
            "Positional baselines need samples drawn from a data split, so the "
            f"training data is known; {path!s} has source {attrs.get('source')!r}"
        )
    if "data_config" not in attrs:
        raise ValueError(
            f"{path!s} does not record the data configuration it was sampled "
            "with, which the positional baselines need to rebuild its training "
            "split. Resample it with scripts/sample.py, or turn off "
            "include_positional_baselines and include_flank_baselines."
        )

    # The fit reads whole regions and never builds a cropped dataset, so the
    # model crop length is irrelevant; matching the region length keeps the
    # config valid for any region size.
    recorded = json.loads(attrs["data_config"])
    return SparseDataModuleConfig(
        fasta_path=recorded["fasta_path"],
        h5_path=recorded["h5_paths"],
        assay_type=recorded["assay_type"],
        h5_region_length=recorded["h5_region_length"],
        model_input_length=recorded["h5_region_length"],
        train_chrm=recorded["train_chrm"],
        val_chrm=recorded["val_chrm"],
        test_chrm=recorded["test_chrm"],
    )


def check_regions_outside_training(
    region_ids: Collection[str],
    train_chromosomes: Collection[str],
) -> None:
    """Refuse to score regions the baselines were fit on."""

    train = set(train_chromosomes)
    leaked = sorted(
        {
            chromosome
            for chromosome in (parse_region_string(region)[0] for region in region_ids)
            if chromosome in train
        }
    )
    if leaked:
        raise ValueError(
            "Sampled regions come from chromosomes the positional baselines are "
            f"fit on: {', '.join(leaked)}"
        )


@dataclass(frozen=True)
class TrainingSplit:
    """Assay-filtered training regions behind a samples file, in full-region
    coordinates, which every train-set baseline is fit from."""

    config: SparseDataModuleConfig
    reads: list[torch.Tensor]
    positions: list[torch.Tensor]
    dna: torch.Tensor


def load_training_split(path: str | PathLike[str]) -> TrainingSplit:
    """Load the training chromosomes behind a samples file.

    Reads are loaded at full depth whatever read fraction the model trained
    on, because the baselines should be as close to noise-free as possible.
    """

    config = data_config_from_samples(path)
    with h5py.File(path, "r") as handle:
        region_ids = handle["region_ids"].asstr()[:].tolist()
    check_regions_outside_training(region_ids, config.train_chrm)

    module = SparseSMFDataModule(config)
    with single_threaded_torch():
        regions, reads, positions, sources = module.load_smf_from_h5(config.train_chrm)
        dna = load_dna(config.fasta_path, regions, region_length=config.h5_region_length)
        _, reads, positions, dna, _ = filter_smf_regions_for_assay(
            regions,
            reads,
            positions,
            dna,
            sources,
            config.assay_type,
        )
    return TrainingSplit(config, reads, positions, dna)


def fit_baselines_for_samples(
    path: str | PathLike[str],
    settings: PositionalBaselineSettings = PositionalBaselineSettings(),
    progress: bool = True,
    training: TrainingSplit | None = None,
) -> PositionalBaselines:
    """Fit baselines on the training chromosomes behind a samples file.

    ``training`` reuses an already loaded split instead of reading it again.
    """

    if training is None:
        training = load_training_split(path)
    return fit_positional_baselines(
        training.reads,
        training.positions,
        training.config.h5_region_length,
        training.config.assay_type,
        training.config.train_chrm,
        settings,
        progress=progress,
    )
