"""In-silico mutagenesis (ISM) of predicted SMF profiles.

Per-read scoring of substitutions (PRISM) is in :mod:`smf_net.evaluation.prism`.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from tqdm import tqdm

from smf_net.data import dna_tensor_to_one_hot
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.regions import SequenceRegion
from smf_net.model import SMFNet


@dataclass(frozen=True)
class ProfileISM:
    """Profile effects by mutation position, substituted base, and output."""

    influence: torch.Tensor
    scaled_influence: torch.Tensor
    base_deltas: torch.Tensor
    reference_logits: torch.Tensor
    dna: torch.Tensor
    mutation_start: int
    mutation_end: int
    output_start: int
    output_end: int


@dataclass(frozen=True)
class ProfileISMConfig:
    """Configuration for profile ISM over one or more regions."""

    length: int | None = None
    mutation_start: int | None = None
    batch_size: int = 128
    importance_window: int = 51
    softmax_temperature: float = 0.3
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.length is not None and self.length < 1:
            raise ValueError("length must be positive or None")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.importance_window < 1 or self.importance_window % 2 == 0:
            raise ValueError("importance_window must be a positive odd number")
        if self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")


@dataclass(frozen=True)
class RegionProfileISM:
    region_id: str
    result: ProfileISM


@dataclass(frozen=True)
class MutationWindow:
    mutation_start: int
    mutation_end: int
    output_start: int
    output_end: int

    @property
    def length(self) -> int:
        return self.mutation_end - self.mutation_start


def _resolve_window(
    dna_length: int,
    length: int | None,
    mutation_start: int | None,
) -> MutationWindow:
    # The model predicts the middle half of its DNA input.
    output_length = dna_length // 2
    length = output_length if length is None else length
    if length < 1:
        raise ValueError("ISM window length must be positive")
    output_offset = (dna_length - output_length) // 2

    if mutation_start is None:
        output_start = (output_length - length) // 2
        mutation_start = output_offset + output_start
    else:
        output_start = mutation_start - output_offset

    window = MutationWindow(
        mutation_start=mutation_start,
        mutation_end=mutation_start + length,
        output_start=output_start,
        output_end=output_start + length,
    )
    if window.mutation_start < 0 or window.mutation_end > dna_length:
        raise ValueError("Mutation window falls outside the DNA sequence")
    if window.output_start < 0 or window.output_end > output_length:
        raise ValueError("Mutation window falls outside the model output")
    return window


def _mutated_dna(dna: torch.Tensor, window: MutationWindow) -> torch.Tensor:
    mutations = dna.repeat(window.length * 4, 1).long()
    positions = torch.arange(
        window.mutation_start,
        window.mutation_end,
        device=dna.device,
    ).repeat_interleave(4)
    bases = torch.arange(4, device=dna.device).repeat(window.length)
    mutations[torch.arange(len(mutations), device=dna.device), positions] = bases
    return mutations


def _score_dna_batches(
    dna: torch.Tensor,
    batch_size: int,
    score_batch: Callable[[torch.Tensor], torch.Tensor],
    verbose: bool,
    description: str,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = []
    for start in tqdm(
        range(0, len(dna), batch_size),
        desc=description,
        disable=not verbose,
    ):
        scores.append(score_batch(dna[start : start + batch_size]).cpu())
    return torch.cat(scores)


def _mean_alternative_delta(
    base_deltas: torch.Tensor,
    wildtype: torch.Tensor,
) -> torch.Tensor:
    known = (wildtype >= 0) & (wildtype < 4)
    include = torch.ones(
        (len(wildtype), 4),
        dtype=torch.bool,
        device=base_deltas.device,
    )
    include[known, wildtype[known].long()] = False
    weights = include.to(base_deltas.dtype)
    weights /= weights.sum(dim=1, keepdim=True)
    return (base_deltas * weights.unsqueeze(-1)).sum(dim=1)


def _profile_logits(model: SMFNet, dna: torch.Tensor) -> torch.Tensor:
    one_hot = dna_tensor_to_one_hot(dna).to(model.device, model.dtype)
    masked = torch.full(
        (len(dna), 1, dna.shape[-1] // 2),
        MASK_TOKEN,
        dtype=torch.long,
        device=model.device,
    )
    logits = model.model(model._encode_smf_tokens(masked), one_hot)
    return logits[:, 0, 0]


@torch.no_grad()
def profile_ism(
    model: SMFNet,
    dna: torch.Tensor,
    *,
    length: int | None = None,
    mutation_start: int | None = None,
    batch_size: int = 128,
    importance_window: int = 51,
    softmax_temperature: float = 0.3,
    verbose: bool = False,
) -> ProfileISM:
    """Measure profile-logit changes after every base substitution in a window."""

    model.eval()
    window = _resolve_window(dna.shape[-1], length, mutation_start)
    mutations = _mutated_dna(dna, window)
    model._validate_input_length(dna.shape[-1])
    reference_logits = _profile_logits(model, dna.unsqueeze(0))[0].float().cpu()
    mutation_logits = _score_dna_batches(
        mutations,
        batch_size,
        lambda batch: _profile_logits(model, batch),
        verbose,
        "Profile ISM",
    ).view(window.length, 4, -1)

    output_slice = slice(window.output_start, window.output_end)
    base_deltas = (mutation_logits[..., output_slice] - reference_logits[output_slice]).float()
    influence = _mean_alternative_delta(
        base_deltas,
        dna[window.mutation_start : window.mutation_end].to(base_deltas.device),
    )

    centered = reference_logits - reference_logits.mean()
    kernel = torch.ones(1, 1, importance_window) / importance_window
    importance = F.conv1d(
        -centered.view(1, 1, -1),
        kernel,
        padding=importance_window // 2,
    )[0, 0]
    importance = torch.softmax(importance / softmax_temperature, dim=0)
    scaled_influence = influence * importance[output_slice]

    return ProfileISM(
        influence=influence,
        scaled_influence=scaled_influence,
        base_deltas=base_deltas,
        reference_logits=reference_logits[output_slice],
        dna=dna[window.mutation_start : window.mutation_end].cpu(),
        mutation_start=window.mutation_start,
        mutation_end=window.mutation_end,
        output_start=window.output_start,
        output_end=window.output_end,
    )


def profile_ism_regions(
    model: SMFNet,
    regions: Iterable[SequenceRegion],
    config: ProfileISMConfig = ProfileISMConfig(),
) -> Iterator[RegionProfileISM]:
    """Run profile ISM over a stream of named DNA regions."""

    for region in regions:
        yield RegionProfileISM(
            region_id=region.region_id,
            result=profile_ism(
                model,
                region.dna,
                length=config.length,
                mutation_start=config.mutation_start,
                batch_size=config.batch_size,
                importance_window=config.importance_window,
                softmax_temperature=config.softmax_temperature,
                verbose=config.verbose,
            ),
        )
