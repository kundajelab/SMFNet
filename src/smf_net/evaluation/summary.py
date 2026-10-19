"""Aggregate summaries of a per-region metrics table.

A single region's distance metric is close to its own sampling noise: with
around a hundred matched reads the observed pseudo-replicate floor scatters by
roughly the size of a good model's discrepancy, so no per-region ratio of two
such numbers is stable. Pooling regions first is what makes a normalized score
usable, which is why every number here is a median over regions rather than an
average of per-region ratios.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

FLOOR_SUFFIX = "_pseudo_replicate"
NULL_SUFFIX = "_shuffle"

# Most metrics here are discrepancies, where the replicate floor is the smallest
# reachable value. Agreement metrics run the other way: their floor is the
# largest reachable value and their null sits near zero.
HIGHER_IS_BETTER_PREFIXES = (
    "pearson",
    "spearman",
    "covariation_corr_of_corr",
    "covariation_residual_corr_of_corr",
)


def prefers_higher(column: str) -> bool:
    """Whether a larger value of this metric means a closer match."""

    return column.startswith(HIGHER_IS_BETTER_PREFIXES)


def metric_unit(column: str) -> str:
    """The unit a metric is stored in: bp for run-length distances, else none."""

    return "bp" if "wasserstein" in column else ""


@dataclass(frozen=True)
class GapClosed:
    """Where the model sits between the shuffle baseline and the noise floor.

    ``fraction`` is 1 when the model matches the observed sample as closely as
    an observed replicate does and 0 when it is no better than the structure
    free baseline. Negative values mean the model is worse than the baseline.
    ``model``, ``floor`` and ``null`` are the medians over regions it is built
    from, in ``unit``.
    """

    metric: str
    n_regions: int
    unit: str
    model: float
    floor: float
    null: float
    fraction: float
    standard_error: float


def gap_closed_columns(columns: pd.Index | list[str]) -> list[str]:
    """Find metrics carrying both a shuffle baseline and a replicate floor."""

    available = set(columns)
    return [
        column
        for column in columns
        if column + FLOOR_SUFFIX in available and column + NULL_SUFFIX in available
    ]


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan")
    return float(np.median(finite))


def _fraction(
    model: np.ndarray,
    floor: np.ndarray,
    null: np.ndarray,
    higher_is_better: bool = False,
) -> float:
    null_median = _median(null)
    gap = null_median - _median(floor)
    # The ratio itself is direction free; only the sign the gap must have to be
    # a real gap flips. An agreement metric has its floor above its null, so a
    # positive gap there is the degenerate case rather than the usual one.
    expected_sign = -1.0 if higher_is_better else 1.0
    if not np.isfinite(gap) or gap * expected_sign <= 0:
        # Without a baseline worse than the floor there is no gap to close, so
        # the region set holds no structure this metric can resolve.
        return float("nan")
    return (null_median - _median(model)) / gap


def _triplet(table: pd.DataFrame, column: str) -> tuple[np.ndarray, ...]:
    return tuple(
        table[name].to_numpy(dtype=float)
        for name in (column, column + FLOOR_SUFFIX, column + NULL_SUFFIX)
    )


def gap_closed_fraction(table: pd.DataFrame, column: str) -> float:
    """Fraction of the baseline-to-floor gap the model closes, or NaN."""

    return _fraction(*_triplet(table, column), prefers_higher(column))


def gap_closed(
    table: pd.DataFrame,
    column: str,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> GapClosed:
    """Summarize one metric, resampling regions for a standard error."""

    model, floor, null = _triplet(table, column)
    higher_is_better = prefers_higher(column)
    generator = np.random.default_rng(seed)
    resampled = [
        _fraction(model[draw], floor[draw], null[draw], higher_is_better)
        for draw in (
            generator.integers(0, model.size, model.size)
            for _ in range(bootstrap_samples if model.size else 0)
        )
    ]
    finite = [value for value in resampled if np.isfinite(value)]
    fraction = _fraction(model, floor, null, higher_is_better)
    return GapClosed(
        metric=column,
        n_regions=int(model.size),
        unit=metric_unit(column),
        model=_median(model),
        floor=_median(floor),
        null=_median(null),
        fraction=fraction,
        # A few resamples can define a fraction the whole table does not, and
        # an error bar on an undefined estimate would only mislead.
        standard_error=(
            float(np.std(finite)) if finite and np.isfinite(fraction) else float("nan")
        ),
    )


def gap_closed_summary(
    table: pd.DataFrame,
    *,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per metric that has both a shuffle baseline and a floor."""

    return pd.DataFrame.from_records(
        [
            asdict(
                gap_closed(
                    table,
                    column,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                )
            )
            for column in gap_closed_columns(table.columns)
        ]
    )
