from types import SimpleNamespace

import pytest
import torch

from smf_net.data import SparseSMFRegion
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.regions import RegionBatch
from smf_net.evaluation.sampling import SamplingConfig, sample_regions, sample_split


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, assay_type: str = "cg_gc_accessibility") -> None:
        self.calls: list[tuple[int, int | None, torch.dtype]] = []
        self.cfg = SimpleNamespace(input_length=8, assay_type=assay_type)

    def sample(
        self,
        dna: torch.Tensor,
        reads_per_region: int,
        verbose: bool = False,
        sample_steps: int | None = None,
        generator: torch.Generator | None = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        sampling_window: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        del verbose
        self.calls.append((reads_per_region, sample_steps, compute_dtype))
        self.windows = [*getattr(self, "windows", []), sampling_window]
        samples = torch.randint(
            0,
            2,
            (dna.shape[0], reads_per_region, dna.shape[-1] // 2),
            generator=generator,
        )
        samples[..., -1] = MASK_TOKEN
        return samples


def test_sampling_window_is_forwarded_to_the_model():
    assert SamplingConfig(sampling_window=[1, 3]).sampling_window == (1, 3)
    model = FakeModel()
    batch = RegionBatch(region_ids=("a",), dna=torch.zeros((1, 8), dtype=torch.long))
    config = SamplingConfig(reads_per_region=2, progress=False, sampling_window=(1, 3))
    list(sample_regions(model, iter([batch]), config, total_regions=1))
    assert model.windows == [(1, 3)]


def test_sample_regions_streams_chunked_cpu_batches():
    model = FakeModel()
    regions = RegionBatch(
        ["region_a", "region_b"],
        torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [3, 2, 1, 0, 3, 2, 1, 0],
            ],
            dtype=torch.int8,
        ),
    )
    config = SamplingConfig(
        reads_per_region=5,
        reads_per_chunk=2,
        sample_steps=7,
        seed=11,
        progress=False,
    )

    result_iterator = sample_regions(model, [regions], config)
    assert model.calls == []

    result = next(result_iterator)

    assert model.calls == [
        (2, 7, torch.bfloat16),
        (2, 7, torch.bfloat16),
        (1, 7, torch.bfloat16),
    ]
    assert result.region_ids == ("region_a", "region_b")
    assert result.reads.shape == (2, 5, 4)
    assert result.reads.device.type == "cpu"
    assert result.reads.dtype == torch.int8
    assert torch.all(result.reads[..., -1] == -1)
    assert set(result.reads.unique().tolist()) <= {-1, 0, 1}
    assert result.input_length == 8
    assert result.output_offset == 2
    assert result.output_slice == slice(2, 6)

    with pytest.raises(StopIteration):
        next(result_iterator)


def test_sample_regions_is_reproducible_with_a_local_seed():
    regions = RegionBatch(
        ["region_a"],
        torch.tensor([[0, 1, 2, 3] * 4], dtype=torch.int8),
    )
    config = SamplingConfig(
        reads_per_region=16,
        reads_per_chunk=3,
        seed=42,
        progress=False,
    )

    first = next(sample_regions(FakeModel(), [regions], config)).reads
    second = next(sample_regions(FakeModel(), [regions], config)).reads

    assert torch.equal(first, second)


def test_sample_regions_forwards_compute_dtype():
    model = FakeModel()
    regions = RegionBatch(
        ["region_a"],
        torch.tensor([[0, 1, 2, 3] * 4], dtype=torch.int8),
    )

    next(
        sample_regions(
            model,
            [regions],
            SamplingConfig(reads_per_region=1, progress=False),
            compute_dtype=torch.float32,
        )
    )

    assert model.calls == [(1, None, torch.float32)]


class FakeDataset:
    def __init__(self) -> None:
        self.regions = [
            SparseSMFRegion(
                region_id="region_a",
                dna=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int8),
                smf=torch.tensor([[0, 1], [1, 0]], dtype=torch.int8),
                positions=torch.tensor([2, 4]),
                window_start=0,
            )
        ]

    def __len__(self) -> int:
        return len(self.regions)

    def iter_regions(self):
        return iter(self.regions)


class FakeDataModule:
    def __init__(self, assay_type: str) -> None:
        self.cfg = SimpleNamespace(model_input_length=8, assay_type=assay_type)
        self.val_dataset = FakeDataset()
        self.test_dataset = FakeDataset()
        self.setup_calls = []

    def setup(self, stage: str) -> None:
        self.setup_calls.append(stage)


def test_sample_split_accepts_matching_assay_configuration():
    model = FakeModel("gch_accessibility")
    data = FakeDataModule("gch_accessibility")

    batches = list(
        sample_split(
            model,
            data,
            "val",
            SamplingConfig(reads_per_region=2, progress=False),
        )
    )

    assert data.setup_calls == ["validate"]
    assert len(batches) == 1
    assert batches[0].samples.region_ids == ("region_a",)
    assert batches[0].observed_regions[0].region_id == "region_a"
