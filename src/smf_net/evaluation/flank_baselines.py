"""Sequence-context train-set baseline for the profile correlations.

How methylated a site is depends partly on its immediate sequence context,
through the methyltransferase's flanking preference and any bias in calling.
This baseline predicts each site's mean call from its flanking bases alone, so
it is the reference a model has to beat to show it learned more than local
context:

* the logit of a site's mean call is a bias plus one weight per flanking
  position and base, a logistic regression on one-hot flanks. Being additive,
  every weight is learned from every site with that base at that position, so
  rare contexts need no special handling;
* flanks are read on the plus strand and a context is never tied to its
  reverse complement, so any strand-specific bias in the data is part of the
  baseline rather than credit for the model;
* for ``cg_gc`` assays CpG, GpC and GCG cytosines each get their own model,
  because they differ far more than any flank does. Fiber-seq A and T sites
  get one each: read on the plus strand, a T site is a bottom-strand adenine
  whose context runs the other way.

A flank outside the DNA or an unknown base contributes its position's average
effect over the training base frequencies, which is what the centred one-hot
encoding of a missing base reduces to. Sites at the edge of a sampled window
therefore fall back to their remaining context.

The baseline only gives a probability per site, which suits the profile
correlations. Sampling reads from it would make every site independent, and
the read-axis shuffle of the observed reads already gives that reference with
the true per-site rates.
"""

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from os import PathLike

import numpy as np
import torch

from smf_net.data import (
    ASSAYS,
    DNA_MAPPING,
    compute_masks_cg_gc,
    get_valid_mask,
    site_means,
)

UNKNOWN_BASE = 4
SITE_TYPES = {"cg_gc": ("CpG", "GpC", "GCG"), "gch": ("GpC",), "at": ("A", "T")}


@dataclass(frozen=True)
class FlankBaselineSettings:
    """Context width, regularization and subsampling for the fit.

    The L2 penalty mostly picks one solution among the equivalent ones the
    four bases of each position allow. With millions of sites it barely moves
    the fit. ``max_sites`` caps how many training sites each model is fit on,
    which keeps the fit to seconds; a few dozen weights need far fewer.
    """

    flank_bp: int = 5
    l2: float = 1e-4
    max_sites: int | None = 2_000_000
    max_iter: int = 100
    seed: int = 0


def site_types(dna: torch.Tensor, assay_type: str) -> torch.Tensor:
    """Index into the assay's ``SITE_TYPES`` per position, -1 off-site."""

    target_rule = ASSAYS[assay_type].target_rule
    types = torch.full(dna.shape, -1, dtype=torch.long)
    if target_rule == "gch":
        types[get_valid_mask(dna, assay_type)] = 0
        return types
    if target_rule == "at":
        types[dna == DNA_MAPPING["A"]] = 0
        types[dna == DNA_MAPPING["T"]] = 1
        return types
    if target_rule != "cg_gc":
        raise ValueError(f"No flank site types for target rule {target_rule!r}")
    cg, gc = compute_masks_cg_gc(dna)
    types[cg & ~gc] = 0
    types[gc & ~cg] = 1
    types[cg & gc] = 2
    return types


def flank_tokens(dna: torch.Tensor, sites: torch.Tensor, flank_bp: int) -> torch.Tensor:
    """Bases at the ``flank_bp`` positions either side of each site.

    Columns run from the farthest left flank to the farthest right one; bases
    outside ``dna`` or unknown are ``UNKNOWN_BASE``.
    """

    offsets = torch.cat([torch.arange(-flank_bp, 0), torch.arange(1, flank_bp + 1)])
    index = sites[:, None] + offsets
    inside = (index >= 0) & (index < dna.shape[-1])
    bases = dna[index.clamp(0, dna.shape[-1] - 1)].long()
    known = inside & (bases >= 0) & (bases < UNKNOWN_BASE)
    return torch.where(known, bases, UNKNOWN_BASE)


def _logits(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    base_frequencies: torch.Tensor,
) -> torch.Tensor:
    # An unknown base takes the frequency-weighted mean of its position's
    # weights, which is what the centred encoding gives a missing base.
    unknown = (base_frequencies * weights).sum(dim=-1, keepdim=True)
    table = torch.cat([weights, unknown], dim=-1)
    columns = torch.arange(tokens.shape[1])
    return bias + table[columns, tokens].sum(dim=-1)


def _base_frequencies(tokens: torch.Tensor) -> torch.Tensor:
    counts = torch.stack(
        [(tokens == base).sum(dim=0) for base in range(UNKNOWN_BASE)], dim=-1
    ).double()
    totals = counts.sum(dim=-1, keepdim=True)
    return torch.where(totals > 0, counts / totals.clamp_min(1), 0.25)


def _fit_logistic(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    settings: FlankBaselineSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_frequencies = _base_frequencies(tokens)
    mean = targets.mean().clamp(1e-4, 1 - 1e-4)
    weights = torch.zeros((tokens.shape[1], UNKNOWN_BASE), dtype=torch.float64, requires_grad=True)
    bias = torch.logit(mean).detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        max_iter=settings.max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                _logits(tokens, weights, bias, base_frequencies), targets
            )
            + settings.l2 * weights.square().sum()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach(), bias.detach(), base_frequencies


@dataclass(frozen=True)
class FlankBaseline:
    """Fitted flank models, one per site type of the assay.

    ``weights[type]`` is ``[2 * flank_bp, 4]`` over flank positions, from the
    farthest left to the farthest right, and bases A/C/G/T. A site type no
    training site had predicts NaN.
    """

    weights: np.ndarray
    bias: np.ndarray
    base_frequencies: np.ndarray
    n_sites: np.ndarray
    assay_type: str
    train_chromosomes: tuple[str, ...]
    n_regions: int
    settings: FlankBaselineSettings

    @property
    def site_type_names(self) -> tuple[str, ...]:
        return SITE_TYPES[ASSAYS[self.assay_type].target_rule]

    def predict(self, dna: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Baseline mean call at ``positions`` of ``dna``; NaN off-site."""

        types = site_types(dna, self.assay_type)[positions]
        tokens = flank_tokens(dna, positions, self.settings.flank_bp)
        predicted = torch.full(positions.shape, torch.nan, dtype=torch.float32)
        for site_type in range(len(self.site_type_names)):
            selected = types == site_type
            if not selected.any() or self.n_sites[site_type] == 0:
                continue
            logits = _logits(
                tokens[selected],
                torch.from_numpy(self.weights[site_type]),
                torch.tensor(self.bias[site_type]),
                torch.from_numpy(self.base_frequencies[site_type]),
            )
            predicted[selected] = torch.sigmoid(logits).float()
        return predicted

    def save(self, path: str | PathLike[str]) -> None:
        """Write the fit and its provenance to an ``.npz`` for inspection."""

        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            base_frequencies=self.base_frequencies,
            n_sites=self.n_sites,
            metadata=json.dumps(
                {
                    "assay_type": self.assay_type,
                    "site_types": list(self.site_type_names),
                    "bases": "ACGT",
                    "train_chromosomes": list(self.train_chromosomes),
                    "n_regions": self.n_regions,
                    "settings": asdict(self.settings),
                }
            ),
        )


def fit_flank_baseline(
    reads: Sequence[torch.Tensor],
    positions: Sequence[torch.Tensor],
    dna: torch.Tensor,
    assay_type: str,
    train_chromosomes: Collection[str],
    settings: FlankBaselineSettings = FlankBaselineSettings(),
) -> FlankBaseline:
    """Fit the flank models from training regions in full-region coordinates.

    Each region's site contributes its mean call once, as a soft label, so a
    deeply read region does not outweigh the others, matching the per-region
    scoring.
    """

    if not reads:
        raise ValueError("The flank baseline needs at least one training region")

    tokens_by_type: list[list[torch.Tensor]] = [
        [] for _ in SITE_TYPES[ASSAYS[assay_type].target_rule]
    ]
    targets_by_type: list[list[torch.Tensor]] = [[] for _ in tokens_by_type]
    for region_reads, region_positions, region_dna in zip(reads, positions, dna, strict=True):
        means, counts = site_means(region_reads)
        types = site_types(region_dna, assay_type)[region_positions]
        keep = counts > 0
        tokens = flank_tokens(region_dna, region_positions, settings.flank_bp)
        for site_type in range(len(tokens_by_type)):
            selected = keep & (types == site_type)
            tokens_by_type[site_type].append(tokens[selected])
            targets_by_type[site_type].append(means[selected])

    width = 2 * settings.flank_bp
    weights = np.zeros((len(tokens_by_type), width, UNKNOWN_BASE))
    bias = np.full(len(tokens_by_type), np.nan)
    base_frequencies = np.full((len(tokens_by_type), width, UNKNOWN_BASE), 0.25)
    n_sites = np.zeros(len(tokens_by_type), dtype=np.int64)
    generator = torch.Generator().manual_seed(settings.seed)
    for site_type, (tokens, targets) in enumerate(
        zip(tokens_by_type, targets_by_type, strict=True)
    ):
        tokens = torch.cat(tokens)
        targets = torch.cat(targets).double()
        n_sites[site_type] = targets.numel()
        if not targets.numel():
            continue
        if settings.max_sites is not None and targets.numel() > settings.max_sites:
            subset = torch.randperm(targets.numel(), generator=generator)
            subset = subset[: settings.max_sites]
            tokens, targets = tokens[subset], targets[subset]
        fit_weights, fit_bias, frequencies = _fit_logistic(tokens, targets, settings)
        weights[site_type] = fit_weights.numpy()
        bias[site_type] = float(fit_bias)
        base_frequencies[site_type] = frequencies.numpy()

    return FlankBaseline(
        weights=weights,
        bias=bias,
        base_frequencies=base_frequencies,
        n_sites=n_sites,
        assay_type=assay_type,
        train_chromosomes=tuple(train_chromosomes),
        n_regions=len(reads),
        settings=settings,
    )
