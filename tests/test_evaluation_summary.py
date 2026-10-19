import numpy as np
import pandas as pd
import pytest

from smf_net.evaluation.summary import (
    gap_closed,
    gap_closed_columns,
    gap_closed_fraction,
    gap_closed_summary,
)


def _table(model: list[float], floor: list[float], null: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance_x": model,
            "distance_x_pseudo_replicate": floor,
            "distance_x_shuffle": null,
        }
    )


def test_gap_closed_columns_needs_both_a_floor_and_a_baseline():
    columns = [
        "distance_x",
        "distance_x_pseudo_replicate",
        "distance_x_shuffle",
        "pearson",
        "pearson_shuffle",
        "distance_y",
        "distance_y_pseudo_replicate",
    ]

    assert gap_closed_columns(columns) == ["distance_x"]


def test_gap_closed_fraction_measures_position_between_baseline_and_floor():
    at_floor = _table([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    at_baseline = _table([1.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    halfway = _table([0.5, 0.5], [0.0, 0.0], [1.0, 1.0])

    assert gap_closed_fraction(at_floor, "distance_x") == pytest.approx(1.0)
    assert gap_closed_fraction(at_baseline, "distance_x") == pytest.approx(0.0)
    assert gap_closed_fraction(halfway, "distance_x") == pytest.approx(0.5)


def test_gap_closed_fraction_is_negative_when_worse_than_the_baseline():
    worse = _table([2.0, 2.0], [0.0, 0.0], [1.0, 1.0])

    assert gap_closed_fraction(worse, "distance_x") == pytest.approx(-1.0)


def test_gap_closed_fraction_uses_medians_not_means():
    # One region with an enormous baseline would dominate any mean.
    skewed = _table([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1000.0])

    assert gap_closed_fraction(skewed, "distance_x") == pytest.approx(0.5)


def test_gap_closed_fraction_is_nan_without_a_resolvable_gap():
    no_structure = _table([0.01, 0.02], [0.01, 0.02], [0.01, 0.02])
    empty = _table([], [], [])

    assert np.isnan(gap_closed_fraction(no_structure, "distance_x"))
    assert np.isnan(gap_closed_fraction(empty, "distance_x"))


def test_gap_closed_ignores_non_finite_regions():
    with_nan = _table([0.5, np.nan, 0.5], [0.0, 0.0, np.nan], [1.0, 1.0, 1.0])

    assert gap_closed_fraction(with_nan, "distance_x") == pytest.approx(0.5)


def test_gap_closed_reports_a_bootstrap_standard_error():
    generator = np.random.default_rng(0)
    table = _table(
        list(0.5 + 0.1 * generator.standard_normal(400)),
        list(0.1 * generator.standard_normal(400)),
        list(1.0 + 0.1 * generator.standard_normal(400)),
    )

    result = gap_closed(table, "distance_x", bootstrap_samples=50, seed=1)

    assert result.metric == "distance_x"
    assert result.n_regions == 400
    assert result.fraction == pytest.approx(0.5, abs=0.05)
    assert 0.0 < result.standard_error < 0.05
    assert gap_closed(table, "distance_x", bootstrap_samples=50, seed=1) == result


def test_gap_closed_reports_run_lengths_in_bp():
    run_length = pd.DataFrame(
        {
            "run_length_ones_wasserstein": [10.0],
            "run_length_ones_wasserstein_pseudo_replicate": [5.0],
            "run_length_ones_wasserstein_shuffle": [25.0],
        }
    )
    bp = gap_closed(run_length, "run_length_ones_wasserstein", bootstrap_samples=5)

    assert bp.unit == "bp"
    assert (bp.model, bp.floor, bp.null) == pytest.approx((10.0, 5.0, 25.0))

    assert bp.fraction == pytest.approx(0.75)


def test_write_gap_closed_tsv_round_trips(tmp_path):
    from smf_net.evaluation.reporting import write_gap_closed_tsv

    path = tmp_path / "metrics_summary.tsv"
    summary = write_gap_closed_tsv(_table([0.5, 0.5], [0.0, 0.0], [1.0, 1.0]), path)

    written = pd.read_csv(path, sep="\t")
    assert written["metric"].tolist() == ["distance_x"]
    assert written["fraction"].tolist() == pytest.approx([0.5])
    assert written.columns.tolist() == summary.columns.tolist()


def _agreement_table(
    model: list[float],
    floor: list[float],
    null: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pearson_matched": model,
            "pearson_matched_pseudo_replicate": floor,
            "pearson_matched_shuffle": null,
        }
    )


def test_gap_closed_handles_metrics_where_higher_is_better():
    column = "pearson_matched"
    # An agreement metric has its floor above its null, so the model reaching
    # the floor is what closes the gap.
    at_floor = _agreement_table([0.6, 0.6], [0.6, 0.6], [0.0, 0.0])
    halfway = _agreement_table([0.3, 0.3], [0.6, 0.6], [0.0, 0.0])
    at_null = _agreement_table([0.0, 0.0], [0.6, 0.6], [0.0, 0.0])

    assert gap_closed_fraction(at_floor, column) == pytest.approx(1.0)
    assert gap_closed_fraction(halfway, column) == pytest.approx(0.5)
    assert gap_closed_fraction(at_null, column) == pytest.approx(0.0)


def test_gap_closed_rejects_an_agreement_gap_pointing_the_wrong_way():
    column = "pearson_matched"
    # A null that agrees with the observed sample better than a replicate does
    # leaves no gap an agreement metric can resolve.
    inverted = _agreement_table([0.3, 0.3], [0.1, 0.1], [0.8, 0.8])

    assert np.isnan(gap_closed_fraction(inverted, column))


def test_gap_closed_direction_does_not_leak_into_discrepancy_metrics():
    # The same medians read as a resolvable gap for a discrepancy metric and as
    # an inverted one for an agreement metric.
    discrepancy = _table([0.5, 0.5], [0.0, 0.0], [1.0, 1.0])

    assert gap_closed_fraction(discrepancy, "distance_x") == pytest.approx(0.5)


def test_gap_closed_summary_covers_every_eligible_metric():
    table = _table([0.5], [0.0], [1.0])
    table["run_length_ones_wasserstein"] = [10.0]
    table["run_length_ones_wasserstein_pseudo_replicate"] = [5.0]
    table["run_length_ones_wasserstein_shuffle"] = [25.0]

    summary = gap_closed_summary(table, bootstrap_samples=10)

    assert list(summary["metric"]) == ["distance_x", "run_length_ones_wasserstein"]
    assert summary.set_index("metric").loc[
        "run_length_ones_wasserstein", "fraction"
    ] == pytest.approx(0.75)
