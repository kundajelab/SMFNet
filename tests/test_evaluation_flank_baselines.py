import numpy as np
import pytest
import torch

from smf_net.data import dna_str_to_tensor
from smf_net.evaluation.flank_baselines import (
    FlankBaselineSettings,
    fit_flank_baseline,
    flank_tokens,
    site_types,
)

SETTINGS = FlankBaselineSettings(flank_bp=2, max_sites=None)


def _simulate(assay_type, true_rate, n_regions=60, length=300, n_reads=200, seed=0):
    """Random regions whose reads are drawn at ``true_rate(dna, sites)``."""

    generator = torch.Generator().manual_seed(seed)
    reads, positions, dna = [], [], []
    for _ in range(n_regions):
        region_dna = torch.randint(0, 4, (length,), generator=generator).to(torch.int8)
        sites = torch.nonzero(site_types(region_dna, assay_type) >= 0).flatten()
        rate = true_rate(region_dna, sites)
        reads.append(
            (torch.rand((n_reads, sites.numel()), generator=generator) < rate).to(torch.int8)
        )
        positions.append(sites)
        dna.append(region_dna)
    return reads, positions, torch.stack(dna)


def _additive_gch_rate(dna, sites):
    # Columns are flanks -2, -1, +1, +2. T two bases left of the G raises the
    # rate and A right after the C lowers it, so a context and its reverse
    # complement differ.
    tokens = flank_tokens(dna, sites, 2)
    logit = -0.2 + 1.0 * (tokens[:, 0] == 3) - 0.8 * (tokens[:, 2] == 0)
    return torch.sigmoid(logit)


def test_fit_recovers_additive_flank_effects_without_tying_reverse_complements():
    reads, positions, dna = _simulate("gch_accessibility", _additive_gch_rate)

    baseline = fit_flank_baseline(reads, positions, dna, "gch_accessibility", ["chr2"], SETTINGS)

    for region_dna, sites in zip(dna[:5], positions[:5], strict=True):
        np.testing.assert_allclose(
            baseline.predict(region_dna, sites),
            _additive_gch_rate(region_dna, sites),
            atol=0.03,
        )
    # Reverse complements of each other, with true rates 0.27 and 0.69.
    site = torch.tensor([3])
    for sequence in ("TAGCAA", "TTGCTA"):
        context = dna_str_to_tensor(sequence)
        assert float(baseline.predict(context, site)) == pytest.approx(
            float(_additive_gch_rate(context, site)), abs=0.03
        )


def test_missing_flanks_fall_back_and_off_site_positions_are_nan():
    reads, positions, dna = _simulate("gch_accessibility", _additive_gch_rate)
    baseline = fit_flank_baseline(reads, positions, dna, "gch_accessibility", ["chr2"], SETTINGS)

    # The +2 flank of the site at 3 lies past the end of the window.
    truncated = baseline.predict(dna_str_to_tensor("TAGCA"), torch.tensor([3, 4]))
    complete = torch.cat(
        [baseline.predict(dna_str_to_tensor(f"TAGCA{base}"), torch.tensor([3])) for base in "ACGT"]
    )

    assert complete.min() <= truncated[0] <= complete.max()
    assert torch.isnan(truncated[1])


def test_cg_gc_site_types_get_separate_models():
    rates = torch.tensor([0.8, 0.2, 0.5])  # CpG, GpC, GCG
    reads, positions, dna = _simulate(
        "cg_gc_accessibility",
        lambda dna, sites: rates[site_types(dna, "cg_gc_accessibility")[sites]],
    )

    baseline = fit_flank_baseline(reads, positions, dna, "cg_gc_accessibility", ["chr2"], SETTINGS)

    assert baseline.site_type_names == ("CpG", "GpC", "GCG")
    sites = torch.tensor([1, 7, 12])  # one of each type
    predicted = baseline.predict(dna_str_to_tensor("ACGTAAGCTTAGCGT"), sites)
    np.testing.assert_allclose(predicted, rates, atol=0.03)


def test_fiberseq_a_and_t_sites_get_separate_models():
    rates = torch.tensor([0.3, 0.1])  # A, T
    reads, positions, dna = _simulate(
        "fiberseq_m6a",
        lambda dna, sites: rates[site_types(dna, "fiberseq_m6a")[sites]],
    )

    baseline = fit_flank_baseline(reads, positions, dna, "fiberseq_m6a", ["chr2"], SETTINGS)

    assert baseline.site_type_names == ("A", "T")
    sites = torch.tensor([0, 3])  # an A and a T
    predicted = baseline.predict(dna_str_to_tensor("ACGTCC"), sites)
    np.testing.assert_allclose(predicted, rates, atol=0.03)
