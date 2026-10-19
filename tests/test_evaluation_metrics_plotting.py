import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from smf_net.evaluation.covariation import DECAY_PREFIX, SeparationBins
from smf_net.evaluation.metrics_plotting import (
    noise_free_ceiling,
    plot_covariation,
    plot_metrics,
    plot_run_lengths,
    plot_site_correlations,
)
from smf_net.evaluation.run_lengths import RunLengthBins
from smf_net.evaluation.site_correlations import site_correlation_summary


def _metrics_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_valid_positions": [2, 10, 20],
            "n_observed_reads": [20, 20, 20],
            "n_matched_reads": [10, 10, 10],
            "pearson": [0.1, 0.7, 0.9],
            "spearman": [0.2, 0.6, 0.8],
            "pearson_matched": [0.1, 0.5, 0.5],
            "pearson_matched_pseudo_replicate": [0.2, 0.5, 0.8],
            "pearson_matched_shuffle": [0.0, 0.2, 0.2],
            "pearson_positional_baseline": [0.1, 0.3, 0.3],
            "run_length_zeros_wasserstein": [4.0, 2.0, 1.0],
            "run_length_zeros_wasserstein_shuffle": [5.0, 4.0, 3.0],
            "run_length_zeros_wasserstein_read_axis_shuffle": [3.0, 2.5, 2.0],
            "run_length_zeros_wasserstein_pseudo_replicate": [0.5, 0.1, 0.2],
        }
    )


def _run_length_table() -> pd.DataFrame:
    columns = {}
    for name in ("zeros", "ones"):
        columns[f"run_length_{name}_wasserstein"] = [12.0, 8.0, 5.0]
        columns[f"run_length_{name}_wasserstein_shuffle"] = [40.0, 38.0, 35.0]
        columns[f"run_length_{name}_wasserstein_read_axis_shuffle"] = [30.0, 28.0, 25.0]
        columns[f"run_length_{name}_wasserstein_pseudo_replicate"] = [4.0, 3.0, 2.0]
        for suffix in ("model", "observed", "shuffle", "read_axis_shuffle"):
            columns[f"run_length_hist_{name}_{suffix}"] = [
                np.array([10, 5, 1, 0]),
                np.array([0, 20, 4, 2]),
                np.array([1, 1, 1, 1]),
            ]
    return pd.DataFrame(columns)


def _covariation_table() -> pd.DataFrame:
    columns = {}
    for method in ("pearson", "spearman"):
        for prefix in ("covariation_corr_of_corr", "covariation_residual_corr_of_corr"):
            column = f"{prefix}_{method}"
            columns[column] = [0.5, 0.6, 0.7]
            columns[f"{column}_all_reads"] = [0.7, 0.75, 0.8]
            columns[f"{column}_pseudo_replicate"] = [0.6, 0.65, 0.7]
        columns[f"covariation_corr_of_corr_{method}_positional_baseline"] = [0.7] * 3
    for suffix in ("model", "observed", "shuffle", "read_axis_shuffle"):
        columns[f"{DECAY_PREFIX}sum_{suffix}"] = [
            np.array([2.0, 1.0, 0.0]),
            np.array([1.0, 0.5, 0.0]),
            np.array([3.0, 1.5, 0.0]),
        ]
        columns[f"{DECAY_PREFIX}count_{suffix}"] = [
            np.array([4, 2, 0]),
            np.array([2, 1, 0]),
            np.array([6, 3, 0]),
        ]
    return pd.DataFrame(columns)


def test_noise_free_ceiling_applies_spearman_brown_to_the_replicate():
    table = pd.DataFrame(
        {
            "replicate": [0.5, 0.5, -0.2, 1.0],
            "n_observed_reads": [20, 40, 20, 20],
            "n_matched_reads": [10, 10, 10, 10],
        }
    )

    ceiling = noise_free_ceiling(table, "replicate")

    assert ceiling == pytest.approx([np.sqrt(2 / 3), np.sqrt(0.8), 0.0, 1.0])


def test_every_metrics_plot_renders_without_changing_global_settings(tmp_path):
    table = _metrics_table()
    for column, values in {**_run_length_table(), **_covariation_table()}.items():
        table[column] = values
    sites = pd.DataFrame(
        [
            {"comparison": name, "pearson": 0.6, "spearman": 0.7, "n_sites": 42}
            for name in (
                "model_all_reads",
                "flank_baseline_all_reads",
                "model_matched",
                "flank_baseline_matched",
                "pseudo_replicate_matched",
            )
        ]
    )
    before = dict(plt.rcParams)

    figures = [
        plot_metrics(table, min_correlation_sites=5, bins=5),
        plot_site_correlations(sites),
        # A summary with no eligible sites draws placeholder text instead of bars.
        plot_site_correlations(site_correlation_summary(pd.DataFrame())),
        plot_run_lengths(table, RunLengthBins(max_bp=40, bin_width_bp=10)),
        plot_covariation(table, SeparationBins(max_bp=30, bin_width_bp=10)),
    ]

    for index, figure in enumerate(figures):
        figure.savefig(tmp_path / f"figure_{index}.pdf")
        plt.close(figure)
    assert dict(plt.rcParams) == before
