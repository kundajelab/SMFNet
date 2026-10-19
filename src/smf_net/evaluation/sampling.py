"""Batched, streaming sampling for genomic regions."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import batched, tee
from typing import Literal

import torch
from tqdm import tqdm

from smf_net.data import RegionSource, SparseSMFDataModule, SparseSMFRegion
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.regions import RegionBatch
from smf_net.model import SMFNet, validate_data_model_compatibility


@dataclass(frozen=True)
class SampleBatch:
    """Sampled SMF reads for one :class:`RegionBatch`.

    ``reads`` is a dense CPU tensor with shape ``[regions, reads, length]`` and
    values -1/0/1. ``output_offset`` locates its first position relative to the
    corresponding input DNA sequence.
    """

    region_ids: tuple[str, ...]
    dna: torch.Tensor
    reads: torch.Tensor
    input_length: int
    output_offset: int
    sources: tuple[RegionSource | None, ...]

    @property
    def batch_size(self) -> int:
        return len(self.region_ids)

    @property
    def reads_per_region(self) -> int:
        return self.reads.shape[1]

    @property
    def output_length(self) -> int:
        return self.reads.shape[2]

    @property
    def output_slice(self) -> slice:
        return slice(self.output_offset, self.output_offset + self.output_length)


@dataclass(frozen=True)
class PairedSampleBatch:
    """Model samples paired with all observed reads in the output window."""

    samples: SampleBatch
    observed_regions: tuple[SparseSMFRegion, ...]


@dataclass(frozen=True)
class SamplingConfig:
    """Controls model sampling and peak memory use.

    ``sampling_window`` restricts generation to a half-open interval of the
    SMF output window, in output coordinates; positions outside it stay
    missing (-1). The model still sees the full DNA context. Use it to sample
    reads no longer than the training reads.
    """

    reads_per_region: int = 1024
    reads_per_chunk: int = 128
    regions_per_batch: int = 128
    sample_steps: int | None = None
    seed: int | None = 0
    progress: bool = True
    verbose: bool = False
    sampling_window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.sampling_window is not None:
            # Hydra passes lists; store a tuple so the config stays hashable.
            window = tuple(self.sampling_window)
            if (
                len(window) != 2
                or not all(isinstance(v, int) for v in window)
                or not 0 <= window[0] < window[1]
            ):
                raise ValueError("sampling_window must be (start, end) with 0 <= start < end")
            object.__setattr__(self, "sampling_window", window)
        if self.reads_per_region < 1:
            raise ValueError("reads_per_region must be positive")
        if self.reads_per_chunk < 1:
            raise ValueError("reads_per_chunk must be positive")
        if self.regions_per_batch < 1:
            raise ValueError("regions_per_batch must be positive")
        if self.sample_steps is not None and self.sample_steps < 1:
            raise ValueError("sample_steps must be positive or None")


def _make_generator(model: SMFNet, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None

    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    return generator


def _sample_region_batch(
    model: SMFNet,
    regions: RegionBatch,
    config: SamplingConfig,
    generator: torch.Generator | None,
    compute_dtype: torch.dtype,
) -> SampleBatch:
    chunks = []

    for read_start in range(0, config.reads_per_region, config.reads_per_chunk):
        chunk_size = min(
            config.reads_per_chunk,
            config.reads_per_region - read_start,
        )
        sampled = model.sample(
            regions.dna,
            reads_per_region=chunk_size,
            verbose=config.verbose,
            sample_steps=config.sample_steps,
            generator=generator,
            compute_dtype=compute_dtype,
            sampling_window=config.sampling_window,
        )

        # Sites that stay masked are missing calls, which evaluation marks -1.
        sampled = torch.where(sampled == MASK_TOKEN, -1, sampled)
        chunks.append(sampled.detach().to(device="cpu", dtype=torch.int8))

    reads = torch.cat(chunks, dim=1)
    output_offset = (regions.sequence_length - reads.shape[-1]) // 2
    return SampleBatch(
        region_ids=regions.region_ids,
        dna=regions.dna.detach().cpu(),
        reads=reads,
        input_length=regions.sequence_length,
        output_offset=output_offset,
        sources=regions.sources,
    )


@torch.no_grad()
def sample_regions(
    model: SMFNet,
    region_batches: Iterable[RegionBatch],
    config: SamplingConfig,
    total_regions: int | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Iterator[SampleBatch]:
    """Sample named region batches without retaining the full result in memory.

    Region batching belongs to the input source; ``reads_per_chunk`` independently
    limits the number of reads materialized on the model device at one time.
    Results are yielded on CPU and can be sent directly to metrics or an output
    writer.
    """

    generator = _make_generator(model, config.seed)
    with tqdm(
        total=total_regions,
        desc="sampling regions",
        unit="region",
        disable=not config.progress,
    ) as progress:
        for regions in region_batches:
            samples = _sample_region_batch(model, regions, config, generator, compute_dtype)
            progress.update(regions.batch_size)
            yield samples


def sample_observed_regions(
    model: SMFNet,
    regions: Iterable[SparseSMFRegion],
    config: SamplingConfig,
    total_regions: int | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Iterator[PairedSampleBatch]:
    """Batch region DNA for sampling while retaining all sparse observed reads."""

    region_groups = batched(regions, config.regions_per_batch)
    groups_for_sampling, groups_for_output = tee(region_groups)

    def region_batches() -> Iterator[RegionBatch]:
        for group in groups_for_sampling:
            region_ids = tuple(region.region_id for region in group)
            if any(region_id is None for region_id in region_ids):
                raise ValueError("Observed regions must include region IDs")
            yield RegionBatch(
                region_ids=region_ids,
                dna=torch.stack([region.dna for region in group]),
                sources=[region.source for region in group],
            )

    for group, samples in zip(
        groups_for_output,
        sample_regions(
            model,
            region_batches(),
            config,
            total_regions=total_regions,
            compute_dtype=compute_dtype,
        ),
        strict=True,
    ):
        yield PairedSampleBatch(
            samples=samples,
            observed_regions=tuple(region.crop(samples.output_slice) for region in group),
        )


def sample_split(
    model: SMFNet,
    data: SparseSMFDataModule,
    split: Literal["val", "test"],
    config: SamplingConfig,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Iterator[PairedSampleBatch]:
    """Sample a validation or test split by directly iterating sparse regions."""

    validate_data_model_compatibility(data.cfg, model.cfg)
    data.setup("validate" if split == "val" else "test")
    dataset = getattr(data, f"{split}_dataset")
    yield from sample_observed_regions(
        model,
        dataset.iter_regions(),
        config,
        total_regions=len(dataset),
        compute_dtype=compute_dtype,
    )
