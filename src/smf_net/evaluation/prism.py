"""Per-read in-silico mutagenesis (PRISM).

PRISM scores how each DNA substitution changes the likelihood of one observed
read, along seeded unmasking trajectories of the diffusion sampler.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from smf_net.data import dna_tensor_to_one_hot
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.ism import (
    MutationWindow,
    _mutated_dna,
    _resolve_window,
    _score_dna_batches,
)
from smf_net.model import SMFNet


@dataclass(frozen=True)
class PRISMResult:
    """Read likelihood scores and deltas with shape ``[mutation, 4]``."""

    scores: torch.Tensor
    deltas: torch.Tensor
    reference_score: torch.Tensor
    per_order_scores: torch.Tensor
    per_order_reference_scores: torch.Tensor
    dna: torch.Tensor
    read: torch.Tensor
    mutation_start: int
    mutation_end: int
    output_start: int
    output_end: int
    likelihood_start: int = 0
    likelihood_end: int | None = None
    context_mode: Literal["masked", "full_read"] = "full_read"


@dataclass(frozen=True)
class PRISMLogRatioResult:
    """Read log-ratio scores and deltas with shape ``[mutation, 4]``."""

    scores: torch.Tensor
    deltas: torch.Tensor
    reference_score: torch.Tensor
    per_order_scores: torch.Tensor
    per_order_reference_scores: torch.Tensor
    dna: torch.Tensor
    read_a: torch.Tensor
    read_b: torch.Tensor
    shared_valid_mask: torch.Tensor
    mutation_start: int
    mutation_end: int
    output_start: int
    output_end: int


@dataclass(frozen=True)
class _ReadTrajectory:
    inputs: torch.Tensor
    labels: torch.Tensor
    score_mask: torch.Tensor
    read: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class _PerOrderReadScores:
    mutation_scores: torch.Tensor
    reference_scores: torch.Tensor


def _prepare_read_trajectory(
    model: SMFNet,
    read: torch.Tensor,
    dna_length: int,
    sample_steps: int,
    unmask_order: torch.Tensor,
    sampling_mode: Literal["cosine", "one_pos_at_a_time"],
    valid_positions: torch.Tensor | None = None,
    score_positions: torch.Tensor | None = None,
    context_mode: Literal["masked", "full_read"] = "full_read",
) -> _ReadTrajectory:
    x_0 = model._center_crop_smf(read, dna_length)
    valid = x_0 != -1
    if valid_positions is not None:
        valid &= valid_positions.to(valid.device)
    scored = valid.clone()
    if score_positions is not None:
        scored &= score_positions.to(valid.device)
    if context_mode == "masked":
        valid = scored
        x_0 = x_0.masked_fill(~valid, -1)

    if sampling_mode == "cosine":
        time = torch.linspace(1, 0, sample_steps + 1, device=x_0.device)
        if context_mode == "masked":
            active_order = unmask_order.to(x_0.device)
            active_order = active_order[valid[active_order]]
            position_rank = torch.full_like(x_0, len(active_order), dtype=torch.long)
            position_rank[active_order] = torch.arange(len(active_order), device=x_0.device)
            trajectory_length = len(active_order)
        else:
            position_rank = torch.argsort(unmask_order.to(x_0.device))
            trajectory_length = len(x_0)
        num_unmasked = (model.schedule.alpha_t(time) * trajectory_length).long()
        keep = position_rank.unsqueeze(0) < num_unmasked.unsqueeze(1)
    elif sampling_mode == "one_pos_at_a_time":
        valid_order = unmask_order.to(x_0.device)
        valid_order = valid_order[valid[valid_order]]
        position_rank = torch.full(
            (len(x_0),),
            len(valid_order),
            dtype=torch.long,
            device=x_0.device,
        )
        position_rank[valid_order] = torch.arange(len(valid_order), device=x_0.device)
        keep = position_rank.unsqueeze(0) < torch.arange(
            len(valid_order) + 1, device=x_0.device
        ).unsqueeze(1)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    states = torch.where(keep, x_0, torch.full_like(x_0, MASK_TOKEN))
    states = states.masked_fill(states == -1, MASK_TOKEN)
    score_mask = (states[:-1] == MASK_TOKEN) & (states[1:] != MASK_TOKEN) & scored
    labels = x_0.masked_fill(~valid, 0).expand(len(score_mask), -1)
    inputs = model._encode_smf_tokens(states[:-1].unsqueeze(0))
    return _ReadTrajectory(
        inputs=inputs,
        labels=labels.unsqueeze(0).to(model.device, model.dtype),
        score_mask=score_mask.cpu(),
        read=x_0,
        valid_mask=valid,
    )


def _dna_embedding(model: SMFNet, dna: torch.Tensor) -> torch.Tensor:
    one_hot = dna_tensor_to_one_hot(dna).to(model.device, model.dtype)
    return model.model.dna_encoder(one_hot)


def _embedded_trajectory_scores(
    model: SMFNet,
    dna_embed: torch.Tensor,
    trajectory: _ReadTrajectory,
) -> torch.Tensor:
    if trajectory.inputs.shape[1] == 0:
        return torch.zeros(len(dna_embed))
    inputs = trajectory.inputs.expand(len(dna_embed), -1, -1, -1).contiguous()
    logits = model.model(inputs, dna_embed=dna_embed).squeeze(2)
    labels = trajectory.labels.expand(len(dna_embed), -1, -1)
    log_probs = -F.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    ).cpu()
    return log_probs.masked_fill(~trajectory.score_mask, 0).sum(dim=(1, 2))


def _score_against_trajectories(
    model: SMFNet,
    dna: torch.Tensor,
    mutations: torch.Tensor,
    trajectories: Sequence[_ReadTrajectory],
    batch_size: int,
    verbose: bool,
    description: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score sequences on every trajectory, encoding each sequence's DNA once.

    The DNA encoder costs about as much as scoring 32 partially masked reads,
    so it runs once per batch and is shared by all trajectories rather than
    rerun for each one. Returns [trajectories, mutations] scores and the
    reference sequence's score on each trajectory.
    """

    reference_embed = _dna_embedding(model, dna.unsqueeze(0))
    references = torch.stack(
        [
            _embedded_trajectory_scores(model, reference_embed, trajectory)[0]
            for trajectory in trajectories
        ]
    )

    def score_batch(batch: torch.Tensor) -> torch.Tensor:
        embed = _dna_embedding(model, batch)
        return torch.stack(
            [_embedded_trajectory_scores(model, embed, trajectory) for trajectory in trajectories],
            dim=1,
        )

    if len(mutations) == 0:
        scores = torch.zeros((0, len(trajectories)))
    else:
        scores = _score_dna_batches(mutations, batch_size, score_batch, verbose, description)
    return scores.T.float(), references.float()


def _make_unmasking_orders(
    read_length: int,
    n_unmasking_orders: int,
    seed: int,
) -> torch.Tensor:
    if n_unmasking_orders < 1:
        raise ValueError("n_unmasking_orders must be positive")
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(read_length, generator=generator) for _ in range(n_unmasking_orders)]
    )


def _score_positions(
    output_length: int, likelihood_window: tuple[int, int] | None
) -> tuple[torch.Tensor, int, int]:
    """The output positions a read is scored on, and the window's bounds."""

    if likelihood_window is None:
        start, end = 0, output_length
    else:
        if len(likelihood_window) != 2:
            raise ValueError("likelihood_window must contain (start, end)")
        start, end = likelihood_window
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= output_length
        ):
            raise ValueError("likelihood_window must be a nonempty interval inside the SMF output")
    positions = torch.zeros(output_length, dtype=torch.bool)
    positions[start:end] = True
    return positions, start, end


@torch.no_grad()
def score_read_mutations(
    model: SMFNet,
    read: torch.Tensor,
    dna: torch.Tensor,
    mutations: torch.Tensor,
    *,
    sample_steps: int | None = None,
    batch_size: int = 128,
    n_unmasking_orders: int = 8,
    seed: int = 13,
    sampling_mode: Literal["cosine", "one_pos_at_a_time"] = "cosine",
    likelihood_window: tuple[int, int] | None = None,
    context_mode: Literal["masked", "full_read"] = "full_read",
    valid_positions: torch.Tensor | None = None,
    verbose: bool = False,
    description: str = "PRISM",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one observed read under each of a set of mutated DNA sequences.

    ``read`` and ``dna`` are in input coordinates and ``mutations`` holds full
    ``[N, input_length]`` sequences. The read is scored by its log-likelihood
    along each of ``n_unmasking_orders`` seeded unmasking trajectories, the
    same trajectories for every sequence. Returns ``[orders, N]`` mutation
    scores and the ``[orders]`` scores of ``dna`` itself.

    ``likelihood_window`` and ``context_mode`` are as in :func:`prism`.
    ``valid_positions`` (output coordinates) further restricts the read's
    calls, as :func:`prism_log_ratio` does to compare two reads on the same
    sites.
    """

    model.eval()
    dna_length = dna.shape[-1]
    model._validate_input_length(dna_length)
    if context_mode not in {"masked", "full_read"}:
        raise ValueError("context_mode must be 'masked' or 'full_read'")
    score_positions, _, _ = _score_positions(dna_length // 2, likelihood_window)
    sample_steps = sample_steps or model.cfg.sample_steps
    trajectories = [
        _prepare_read_trajectory(
            model,
            read,
            dna_length,
            sample_steps,
            order,
            sampling_mode,
            valid_positions,
            score_positions,
            context_mode,
        )
        for order in _make_unmasking_orders(dna_length // 2, n_unmasking_orders, seed)
    ]
    return _score_against_trajectories(
        model, dna, mutations, trajectories, batch_size, verbose, description
    )


def _score_window_substitutions(
    model: SMFNet,
    read: torch.Tensor,
    dna: torch.Tensor,
    window: MutationWindow,
    **options,
) -> _PerOrderReadScores:
    """Score every base at every window position, reusing the reference score.

    The reference base is the unmutated sequence, so it takes the reference
    score instead of being scored again. Unknown reference bases (N) never
    match, so all four bases are scored there.
    """

    mutations = _mutated_dna(dna, window)
    wildtype = dna[window.mutation_start : window.mutation_end].to(mutations.device)
    bases = torch.arange(4, device=mutations.device).repeat(window.length)
    alternative = bases != wildtype.repeat_interleave(4)
    scores, references = score_read_mutations(model, read, dna, mutations[alternative], **options)
    mutation_scores = references[:, None].repeat(1, len(mutations))
    mutation_scores[:, alternative] = scores
    return _PerOrderReadScores(
        mutation_scores=mutation_scores.view(len(references), window.length, 4),
        reference_scores=references,
    )


def _mean_read_scores(
    per_order: _PerOrderReadScores,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = per_order.mutation_scores.mean(dim=0)
    reference = per_order.reference_scores.mean()
    deltas = (per_order.mutation_scores - per_order.reference_scores[:, None, None]).mean(dim=0)
    return scores, deltas, reference


@torch.no_grad()
def prism(
    model: SMFNet,
    read: torch.Tensor,
    dna: torch.Tensor,
    *,
    length: int | None = None,
    mutation_start: int | None = None,
    sample_steps: int | None = None,
    batch_size: int = 128,
    n_unmasking_orders: int = 8,
    seed: int = 13,
    sampling_mode: Literal["cosine", "one_pos_at_a_time"] = "cosine",
    likelihood_window: tuple[int, int] | None = None,
    context_mode: Literal["masked", "full_read"] = "full_read",
    verbose: bool = False,
) -> PRISMResult:
    """Score one observed read under every DNA substitution in a window.

    ``read`` and ``dna`` use the same input coordinates; the model's usual SMF
    center crop is applied before trajectory scoring.

    ``likelihood_window=(start, end)`` selects a half-open interval in the
    center-cropped SMF output coordinates, independently of the DNA mutation
    window. ``None`` scores the full output. With ``context_mode="masked"``,
    positions outside this interval remain masked throughout every trajectory;
    only observed positions inside it are unmasked. ``"full_read"`` retains
    the full read as trajectory context but scores only the selected interval.
    Full DNA context is retained in both modes. These are trajectory scores,
    not exact marginal likelihoods under the normal full-read sampler.
    """

    window = _resolve_window(dna.shape[-1], length, mutation_start)
    _, likelihood_start, likelihood_end = _score_positions(dna.shape[-1] // 2, likelihood_window)
    per_order = _score_window_substitutions(
        model,
        read,
        dna,
        window,
        sample_steps=sample_steps,
        batch_size=batch_size,
        n_unmasking_orders=n_unmasking_orders,
        seed=seed,
        sampling_mode=sampling_mode,
        likelihood_window=likelihood_window,
        context_mode=context_mode,
        verbose=verbose,
        description="PRISM",
    )
    scores, deltas, reference = _mean_read_scores(per_order)
    cropped_read = model._center_crop_smf(read, dna.shape[-1])
    output_slice = slice(window.output_start, window.output_end)
    return PRISMResult(
        scores=scores,
        deltas=deltas,
        reference_score=reference,
        per_order_scores=per_order.mutation_scores,
        per_order_reference_scores=per_order.reference_scores,
        dna=dna[window.mutation_start : window.mutation_end].cpu(),
        read=cropped_read[output_slice].cpu(),
        mutation_start=window.mutation_start,
        mutation_end=window.mutation_end,
        output_start=window.output_start,
        output_end=window.output_end,
        likelihood_start=likelihood_start,
        likelihood_end=likelihood_end,
        context_mode=context_mode,
    )


@torch.no_grad()
def prism_log_ratio(
    model: SMFNet,
    read_a: torch.Tensor,
    read_b: torch.Tensor,
    dna: torch.Tensor,
    *,
    length: int | None = None,
    mutation_start: int | None = None,
    sample_steps: int | None = None,
    batch_size: int = 128,
    n_unmasking_orders: int = 8,
    seed: int = 13,
    sampling_mode: Literal["cosine", "one_pos_at_a_time"] = "cosine",
    verbose: bool = False,
) -> PRISMLogRatioResult:
    """Attribute the relative likelihood of two reads to DNA substitutions.

    The reads share each random unmasking order and are scored only where both
    have observed calls.
    """

    dna_length = dna.shape[-1]
    model._validate_input_length(dna_length)
    window = _resolve_window(dna_length, length, mutation_start)
    cropped_a = model._center_crop_smf(read_a, dna_length)
    cropped_b = model._center_crop_smf(read_b, dna_length)
    if cropped_a.shape != cropped_b.shape:
        raise ValueError("Read shapes must match")
    shared_valid = (cropped_a != -1) & (cropped_b != -1)
    if not shared_valid.any():
        raise ValueError("Reads have no shared valid positions")
    # Both reads use the same seeded unmasking orders.
    options = {
        "sample_steps": sample_steps,
        "batch_size": batch_size,
        "n_unmasking_orders": n_unmasking_orders,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "valid_positions": shared_valid,
        "verbose": verbose,
    }
    scores_a = _score_window_substitutions(
        model, read_a, dna, window, description="PRISM log ratio: read A", **options
    )
    scores_b = _score_window_substitutions(
        model, read_b, dna, window, description="PRISM log ratio: read B", **options
    )
    per_order = _PerOrderReadScores(
        mutation_scores=scores_a.mutation_scores - scores_b.mutation_scores,
        reference_scores=scores_a.reference_scores - scores_b.reference_scores,
    )
    scores, deltas, reference = _mean_read_scores(per_order)
    output_slice = slice(window.output_start, window.output_end)
    return PRISMLogRatioResult(
        scores=scores,
        deltas=deltas,
        reference_score=reference,
        per_order_scores=per_order.mutation_scores,
        per_order_reference_scores=per_order.reference_scores,
        dna=dna[window.mutation_start : window.mutation_end].cpu(),
        read_a=cropped_a[output_slice].cpu(),
        read_b=cropped_b[output_slice].cpu(),
        shared_valid_mask=shared_valid[output_slice].cpu(),
        mutation_start=window.mutation_start,
        mutation_end=window.mutation_end,
        output_start=window.output_start,
        output_end=window.output_end,
    )
