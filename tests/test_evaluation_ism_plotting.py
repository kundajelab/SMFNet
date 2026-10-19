import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch

pytest.importorskip("matplotlib")
pytest.importorskip("tangermeme")

import matplotlib.pyplot as plt

from smf_net.evaluation.ism import ProfileISM
from smf_net.evaluation.ism_plotting import plot_profile_ism
from smf_net.evaluation.prism import PRISMLogRatioResult, PRISMResult
from smf_net.evaluation.prism_plotting import plot_prism, plot_prism_log_ratio


def _dna() -> torch.Tensor:
    return torch.tensor([0, 1, 2, 3], dtype=torch.int8)


def test_ism_and_prism_plotting_smoke(tmp_path):
    profile_result = ProfileISM(
        influence=torch.arange(16).reshape(4, 4).float() - 8,
        scaled_influence=torch.arange(16).reshape(4, 4).float() / 16,
        base_deltas=torch.zeros(4, 4, 4),
        reference_logits=torch.zeros(4),
        dna=_dna(),
        mutation_start=10,
        mutation_end=14,
        output_start=2,
        output_end=6,
    )
    single_read_result = PRISMResult(
        scores=torch.zeros(4, 4),
        deltas=torch.tensor(
            [[0, 1, -1, 2], [1, 0, 2, -1], [-1, 2, 0, 1], [2, -1, 1, 0]],
            dtype=torch.float32,
        ),
        reference_score=torch.tensor(0.0),
        per_order_scores=torch.zeros(2, 4, 4),
        per_order_reference_scores=torch.zeros(2),
        dna=_dna(),
        read=torch.tensor([0, 1, -1, 1], dtype=torch.int8),
        mutation_start=10,
        mutation_end=14,
        output_start=2,
        output_end=6,
    )
    log_ratio_result = PRISMLogRatioResult(
        scores=torch.zeros(4, 4),
        deltas=torch.tensor(
            [[0, 1, -1, 2], [1, 0, 2, -1], [-1, 2, 0, 1], [2, -1, 1, 0]],
            dtype=torch.float32,
        ),
        reference_score=torch.tensor(0.0),
        per_order_scores=torch.zeros(2, 4, 4),
        per_order_reference_scores=torch.zeros(2),
        dna=_dna(),
        read_a=torch.tensor([0, 1, -1, 1], dtype=torch.int8),
        read_b=torch.tensor([1, 0, 1, -1], dtype=torch.int8),
        shared_valid_mask=torch.tensor([True, True, False, False]),
        mutation_start=10,
        mutation_end=14,
        output_start=2,
        output_end=6,
    )

    before = dict(plt.rcParams)
    figures = [
        plot_profile_ism(profile_result, "region_a"),
        plot_prism(single_read_result, "region_a"),
        plot_prism_log_ratio(log_ratio_result, "region_a"),
    ]

    for index, figure in enumerate(figures):
        figure.savefig(tmp_path / f"figure_{index}.pdf")
        plt.close(figure)
    assert dict(plt.rcParams) == before
