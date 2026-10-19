from types import SimpleNamespace

import h5py
import numpy as np
import torch

from smf_net.data import RegionSource, SparseSMFRegion
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.evaluation.regions import RegionBatch
from smf_net.evaluation.sample_io import read_samples_h5, write_samples_h5
from smf_net.evaluation.sampling import (
    SamplingConfig,
    sample_observed_regions,
    sample_regions,
)


class FakeModel:
    device = torch.device("cpu")
    cfg = SimpleNamespace(
        input_length=8,
        assay_type="cg_gc_accessibility",
    )

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
        del verbose, sample_steps, generator, compute_dtype, sampling_window
        samples = torch.full(
            (dna.shape[0], reads_per_region, dna.shape[-1] // 2),
            MASK_TOKEN,
            dtype=torch.int8,
        )
        samples[..., 1] = 0
        samples[..., 3] = 1
        return samples


def test_sample_observed_regions_and_write_sparse_h5(tmp_path):
    source_path = str((tmp_path / "observed.h5").resolve())
    source = RegionSource("observed_h5", source_path)
    regions = [
        SparseSMFRegion(
            region_id="region_a",
            dna=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int8),
            smf=torch.tensor(
                [[0, 1, -1, 1], [1, 0, 1, -1], [0, 0, 1, 1]],
                dtype=torch.int8,
            ),
            positions=torch.tensor([1, 3, 5, 7]),
            window_start=10,
            source=source,
        ),
        SparseSMFRegion(
            region_id="region_b",
            dna=torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.int8),
            smf=torch.tensor([[1, 0], [0, 1]], dtype=torch.int8),
            positions=torch.tensor([3, 6]),
            window_start=20,
            source=source,
        ),
    ]
    config = SamplingConfig(
        reads_per_region=5,
        reads_per_chunk=2,
        regions_per_batch=2,
        progress=False,
    )
    output_path = tmp_path / "samples.h5"

    count = write_samples_h5(
        output_path,
        sample_observed_regions(FakeModel(), regions, config),
        metadata={
            "split": "test",
            "assay_type": "gch_accessibility",
        },
    )

    assert count == 2
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "smfnet.samples"
        assert handle.attrs["schema_version"] == 2
        assert handle.attrs["split"] == "test"
        assert handle.attrs["assay_type"] == "gch_accessibility"
        assert handle.attrs["num_regions"] == 2
        assert handle["region_ids"].asstr()[:].tolist() == [
            "region_a",
            "region_b",
        ]
        assert handle["sources"]["paths"].asstr()[:].tolist() == [source_path]
        assert handle["sources"]["kinds"].asstr()[:].tolist() == ["observed_h5"]

        first = handle["regions"]["00000000"]
        assert first.attrs["region_id"] == "region_a"
        assert first.attrs["window_start"] == 12
        assert first.attrs["source_index"] == 0
        assert first["dna"].shape == (4,)
        assert first["predicted"]["smf"].shape == (5, 2)
        assert first["predicted"]["positions"][:].tolist() == [1, 3]
        assert first["observed"]["smf"].shape == (3, 2)
        assert first["observed"]["positions"][:].tolist() == [1, 3]

        second = handle["regions"]["00000001"]
        assert second.attrs["region_id"] == "region_b"
        assert second["observed"]["smf"].shape == (2, 1)
        assert second["observed"]["positions"][:].tolist() == [1]

    stored = list(read_samples_h5(output_path))
    assert [region.region_id for region in stored] == ["region_a", "region_b"]
    assert stored[0].predicted_smf.shape == (5, 2)
    assert stored[0].observed_smf.shape == (3, 2)
    assert stored[0].observed_mean.shape == (2,)
    assert stored[0].source == source
    assert stored[0].assay_type == "gch_accessibility"


def test_write_and_read_samples_without_observed_reads(tmp_path):
    bed_path = str((tmp_path / "regions.bed").resolve())
    reference_path = str((tmp_path / "reference.fa").resolve())
    source = RegionSource("bed", bed_path, reference_path)
    batch = RegionBatch(
        [np.str_("sequence_a")],
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int8),
        sources=[source],
    )
    output_path = tmp_path / "predictions.h5"

    count = write_samples_h5(
        output_path,
        sample_regions(
            FakeModel(),
            [batch],
            SamplingConfig(reads_per_region=3, progress=False),
        ),
        metadata={"source": "fasta"},
    )

    assert count == 1
    with h5py.File(output_path, "r") as handle:
        assert handle["region_ids"].asstr()[:].tolist() == ["sequence_a"]
        region = handle["regions"]["00000000"]
        assert region.attrs["region_id"] == "sequence_a"
        assert region.attrs["window_start"] == 2
        assert region["dna"][:].tolist() == [2, 3, 0, 1]
        assert "predicted" in region
        assert "observed" not in region
        assert region.attrs["source_index"] == 0
        assert region.attrs["reference_source_index"] == 1
        assert handle["sources"]["paths"].asstr()[:].tolist() == [
            bed_path,
            reference_path,
        ]
        assert handle["sources"]["kinds"].asstr()[:].tolist() == [
            "bed",
            "reference_fasta",
        ]

    stored = next(read_samples_h5(output_path))
    assert stored.observed_smf is None
    assert stored.observed_positions is None
    assert stored.observed_mean is None
    assert stored.source == source
