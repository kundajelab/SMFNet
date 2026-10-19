import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from smf_net.evaluation.marginalization import (
    MarginalizationConfig,
    Motif,
    generate_marginalization_design,
    sample_marginalization,
)
from smf_net.evaluation.marginalization_io import write_marginalizations_h5
from smf_net.evaluation.marginalization_plotting import (
    MotifProfiles,
    effect,
    marginalization_profiles,
    marginalization_profiles_h5,
    plot_marginalizations,
    plot_marginalizations_h5,
)
from smf_net.evaluation.sampling import SamplingConfig


class FakeModel:
    device = torch.device("cpu")

    def sample(self, dna, reads_per_region, generator=None, **kwargs):
        # Calls follow the DNA, so motif and control profiles differ.
        return dna[:, None, :].remainder(2).expand(-1, reads_per_region, -1).clone()


MOTIFS = [Motif("motif_a", "AC", "TG"), Motif("motif_b", "GG", "CC")]
SAMPLING = SamplingConfig(reads_per_region=2, regions_per_batch=2, progress=False)


def _design():
    return generate_marginalization_design(
        MOTIFS, sequence_length=16, config=MarginalizationConfig(n_backgrounds=3)
    )


def test_file_and_streamed_profiles_agree_and_pool_orientations(tmp_path):
    design = _design()
    path = tmp_path / "marginalizations.h5"
    write_marginalizations_h5(
        path,
        design,
        sample_marginalization(FakeModel(), design, SAMPLING),
        show_progress=False,
    )

    from_file = marginalization_profiles_h5(path)
    streamed = marginalization_profiles(
        design, sample_marginalization(FakeModel(), design, SAMPLING), show_progress=False
    )

    # Both orientations of a motif are pooled: 3 backgrounds each.
    assert [profiles.name for profiles in from_file] == ["motif_a", "motif_b"]
    assert from_file[0].motif.shape == (6, 16)
    for a, b in zip(from_file, streamed, strict=True):
        assert a.name == b.name and a.length == b.length
        assert torch.equal(a.x, b.x)
        assert torch.allclose(a.motif, b.motif, equal_nan=True)
        assert torch.allclose(a.control, b.control, equal_nan=True)
    assert [p.name for p in marginalization_profiles_h5(path, combine_orientations=False)] == [
        "motif_a(+)",
        "motif_a(-)",
        "motif_b(+)",
        "motif_b(-)",
    ]


def test_effect_is_motif_minus_control_within_the_window():
    x = torch.arange(-4, 4).float()
    motif = torch.where(x.abs() <= 1, 1.0, 0.0).expand(2, -1)
    profiles = MotifProfiles("m", 2, x, motif, torch.zeros(2, 8))

    assert effect(profiles, window_bp=1) == pytest.approx(1.0)
    assert effect(profiles, window_bp=3) == pytest.approx(3 / 7)


def test_marginalization_plots_render_without_changing_global_settings(tmp_path):
    design = _design()
    path = tmp_path / "marginalizations.h5"
    write_marginalizations_h5(
        path,
        design,
        sample_marginalization(FakeModel(), design, SAMPLING),
        show_progress=False,
    )
    before = dict(plt.rcParams)

    figures = [
        plot_marginalizations_h5(path),
        plot_marginalizations_h5(path, combine_orientations=False, interval=(0.25, 0.75)),
        plot_marginalizations(
            marginalization_profiles_h5(path), effect_window_bp=None, xlim_bp=None
        ),
    ]

    for index, figure in enumerate(figures):
        figure.savefig(tmp_path / f"figure_{index}.pdf")
        plt.close(figure)
    assert dict(plt.rcParams) == before
