"""Run-length distributions for observed and model-generated SMF reads."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

RUN_VALUES = {"zeros": 0, "ones": 1}
HISTOGRAM_PREFIX = "run_length_hist_"


@dataclass(frozen=True)
class RunLengthShuffles:
    """Permuted observed reads scored against the matched observed sample.

    They are baselines of the data, shared by every model.
    """

    observed_within_read: torch.Tensor | None = None
    observed_read_axis: torch.Tensor | None = None


# Field and column suffix of each shuffle.
_SHUFFLE_COLUMNS = (
    ("observed_within_read", "shuffle"),
    ("observed_read_axis", "read_axis_shuffle"),
)


@dataclass(frozen=True)
class RunLengthBins:
    """Fixed base-pair bins for pooled run-length histograms."""

    max_bp: int = 400
    bin_width_bp: int = 10

    def __post_init__(self) -> None:
        if self.bin_width_bp < 1:
            raise ValueError("bin_width_bp must be positive")
        if self.max_bp < self.bin_width_bp:
            raise ValueError("max_bp cannot be smaller than bin_width_bp")
        if self.max_bp % self.bin_width_bp:
            raise ValueError("max_bp must be a multiple of bin_width_bp")

    @property
    def edges(self) -> np.ndarray:
        return np.arange(0, self.max_bp + self.bin_width_bp, self.bin_width_bp)

    @property
    def num_bins(self) -> int:
        return self.max_bp // self.bin_width_bp


def donate_missingness(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Mask predicted reads with per-read missingness drawn from observed reads.

    Model samples are missing only at positions that are invalid for the whole
    region, while observed reads are additionally missing per read. Run lengths
    computed without matching that per-read missingness would credit the model
    for denser coverage rather than for its run structure. Donors are drawn with
    replacement, so low-coverage regions reuse patterns and their predicted run
    lengths are correlated.
    """

    if observed.shape[0] == 0:
        return predicted.clone()
    donors = torch.randint(
        observed.shape[0],
        (predicted.shape[0],),
        generator=generator,
    )
    return predicted.masked_fill(observed[donors] == -1, -1)


def run_length_bp(
    reads: torch.Tensor,
    positions: torch.Tensor,
    value: int,
) -> torch.Tensor:
    """Base-pair spans of maximal runs of ``value`` within each read.

    A run ends at the first position holding any other value, including a
    missing call, so runs never bridge missing positions. The span is the
    distance between the first and last position of the run, so a run covering a
    single position spans 0 bp. Runs touching either end of the measured window
    are dropped because the window truncates them.
    """

    if reads.numel() == 0:
        return torch.empty(0, dtype=positions.dtype)

    num_positions = reads.shape[1]
    padded = torch.zeros(
        (reads.shape[0], num_positions + 2),
        dtype=torch.int8,
    )
    padded[:, 1:-1] = reads == value
    change = padded[:, 1:].int() - padded[:, :-1].int()

    # Padding guarantees one start per end within a read, and ``nonzero`` returns
    # row-major order, so the nth start pairs with the nth end.
    starts = torch.nonzero(change == 1, as_tuple=True)[1]
    ends = torch.nonzero(change == -1, as_tuple=True)[1] - 1
    interior = (starts > 0) & (ends < num_positions - 1)
    return positions[ends[interior]] - positions[starts[interior]]


def run_length_histogram(lengths: torch.Tensor, bins: RunLengthBins) -> np.ndarray:
    """Bin run lengths in bp, collecting spans at or above ``max_bp`` in the last bin."""

    clipped = np.clip(lengths.numpy(), 0, bins.max_bp)
    counts, _ = np.histogram(clipped, bins=bins.edges)
    return counts.astype(np.int64)


def _wasserstein(x: torch.Tensor, y: torch.Tensor, min_runs: int = 1) -> float:
    if x.numel() < max(min_runs, 1) or y.numel() < max(min_runs, 1):
        return float("nan")
    return float(wasserstein_distance(x.numpy(), y.numpy()))


def _mean(lengths: torch.Tensor) -> float:
    if lengths.numel() == 0:
        return float("nan")
    return float(lengths.float().mean())


def run_length_metrics(
    positions: torch.Tensor,
    predicted: torch.Tensor,
    observed: torch.Tensor,
    matched_predicted: torch.Tensor,
    matched_observed: torch.Tensor,
    matched_replicate: torch.Tensor,
    shuffles: RunLengthShuffles,
    bins: RunLengthBins,
    min_runs: int = 1,
) -> dict[str, float | int | np.ndarray]:
    """Compare predicted and observed run lengths for one region.

    Every distance is NaN unless both of its samples hold at least
    ``min_runs`` runs, since a distance between a few runs says little about
    either distribution.

    Distances use the equally sized ``matched_*`` read samples so that the
    observed pseudo-replicate distance is a like-for-like floor. Means, the
    model and observed histograms and the ``*_wasserstein_all_reads`` distance
    use every read, which estimates each distribution more precisely but
    leaves the two sides with different read counts, so that distance has no
    replicate floor to be read against.

    The observed baselines in ``shuffles`` come from the matched replicate, so
    they are the noisiest histograms here. The two permutations bound different
    kinds of structure. Within-read shuffling preserves each read's rate and
    missingness while flattening the accessibility profile, so its distance to
    ``matched_observed`` is the run structure a rate-matched sample misses.
    Read-axis shuffling permutes calls across reads at each position instead,
    keeping the observed profile exactly and destroying only the coupling
    between positions within a read, so its distance isolates the run structure
    that comes from read-level coordination rather than from the profile. A
    model beating the first but not the second is reproducing the profile
    without the coordination.
    """

    record: dict[str, float | int | np.ndarray] = {}
    for name, value in RUN_VALUES.items():
        predicted_runs = run_length_bp(predicted, positions, value)
        observed_runs = run_length_bp(observed, positions, value)
        record[f"n_run_length_{name}_model"] = int(predicted_runs.numel())
        record[f"n_run_length_{name}_observed"] = int(observed_runs.numel())
        record[f"run_length_{name}_mean_model"] = _mean(predicted_runs)
        record[f"run_length_{name}_mean_observed"] = _mean(observed_runs)
        record[f"{HISTOGRAM_PREFIX}{name}_model"] = run_length_histogram(
            predicted_runs,
            bins,
        )
        record[f"{HISTOGRAM_PREFIX}{name}_observed"] = run_length_histogram(
            observed_runs,
            bins,
        )
        record[f"run_length_{name}_wasserstein_all_reads"] = _wasserstein(
            predicted_runs,
            observed_runs,
            min_runs,
        )

        matched_predicted_runs = run_length_bp(matched_predicted, positions, value)
        matched_observed_runs = run_length_bp(matched_observed, positions, value)
        matched_replicate_runs = run_length_bp(matched_replicate, positions, value)
        for sample, runs in (
            ("model", matched_predicted_runs),
            ("observed", matched_observed_runs),
            ("replicate", matched_replicate_runs),
        ):
            record[f"n_run_length_{name}_matched_{sample}"] = int(runs.numel())
        record[f"run_length_{name}_wasserstein"] = _wasserstein(
            matched_predicted_runs,
            matched_observed_runs,
            min_runs,
        )
        record[f"run_length_{name}_wasserstein_pseudo_replicate"] = _wasserstein(
            matched_observed_runs,
            matched_replicate_runs,
            min_runs,
        )

        for field, suffix in _SHUFFLE_COLUMNS:
            reads = getattr(shuffles, field)
            if reads is None:
                continue
            runs = run_length_bp(reads, positions, value)
            record[f"run_length_{name}_wasserstein_{suffix}"] = _wasserstein(
                runs,
                matched_observed_runs,
                min_runs,
            )
            record[f"{HISTOGRAM_PREFIX}{name}_{suffix}"] = run_length_histogram(
                runs,
                bins,
            )
    return record


def histogram_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column.startswith(HISTOGRAM_PREFIX)]


def pooled_histogram(
    table: pd.DataFrame,
    column: str,
    weight_by_coverage: bool = True,
) -> np.ndarray:
    """Pool one region-level histogram column into a single distribution.

    Coverage weighting sums raw counts, so each region contributes in proportion
    to its reads and its site density. The alternative normalizes each region
    first, giving every region equal weight.
    """

    counts = np.stack([np.asarray(value) for value in table[column]])
    if weight_by_coverage:
        return counts.sum(axis=0).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    keep = totals[:, 0] > 0
    if not keep.any():
        return np.zeros(counts.shape[1])
    fractions = counts[keep] / totals[keep]
    return fractions.mean(axis=0)
