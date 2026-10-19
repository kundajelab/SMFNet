import h5py
import torch

from smf_net.evaluation.marginalization import (
    MarginalizationConfig,
    Motif,
    generate_marginalization_design,
    sample_marginalization,
)
from smf_net.evaluation.marginalization_io import write_marginalizations_h5
from smf_net.evaluation.sampling import SamplingConfig


class FakeModel:
    device = torch.device("cpu")

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
        return dna[:, None, :].remainder(2).expand(-1, reads_per_region, -1)


def test_write_marginalizations_h5_stores_full_design_and_samples(tmp_path):
    design = generate_marginalization_design(
        [Motif("motif_a", "AC", "TG"), Motif("motif_b", "GG", "CC")],
        sequence_length=8,
        config=MarginalizationConfig(n_backgrounds=3, seed=4, orientations="forward"),
    )
    batches = sample_marginalization(
        FakeModel(),
        design,
        SamplingConfig(
            reads_per_region=2,
            reads_per_chunk=1,
            regions_per_batch=2,
            progress=False,
        ),
    )
    output_path = tmp_path / "marginalizations.h5"

    write_marginalizations_h5(
        output_path,
        design,
        batches,
        metadata={"checkpoint": "/models/test.ckpt"},
        show_progress=False,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "smfnet.marginalizations"
        assert handle.attrs["schema_version"] == 1
        assert handle.attrs["checkpoint"] == "/models/test.ckpt"
        assert handle.attrs["output_offset"] == 0
        assert handle["motif_names"].asstr()[:].tolist() == [
            "motif_a",
            "motif_b",
        ]
        assert handle["motif_orientations"].asstr()[:].tolist() == ["+", "+"]
        assert handle["dna/background"].shape == (3, 8)
        assert handle["dna/motif"].shape == (2, 3, 8)
        assert handle["samples/background"].shape == (3, 2, 8)
        assert handle["samples/motif"].shape == (2, 3, 2, 8)
        assert handle["samples/control"].shape == (2, 3, 2, 8)
        assert torch.equal(
            torch.from_numpy(handle["dna/motif"][:]),
            design.motif_dna,
        )
        assert torch.equal(
            torch.from_numpy(handle["samples/motif"][0]),
            design.motif_dna[0, :, None, :].remainder(2).expand(-1, 2, -1),
        )
