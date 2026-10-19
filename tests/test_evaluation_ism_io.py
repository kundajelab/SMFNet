import h5py
import numpy as np
import torch

from smf_net.evaluation.ism import ProfileISM, RegionProfileISM
from smf_net.evaluation.ism_io import (
    read_profile_ism_h5,
    write_profile_ism_h5,
)


def _region(region_id: str) -> RegionProfileISM:
    return RegionProfileISM(
        region_id=region_id,
        result=ProfileISM(
            influence=torch.arange(16).reshape(4, 4).float(),
            scaled_influence=torch.arange(16).reshape(4, 4).float() / 2,
            base_deltas=torch.arange(64).reshape(4, 4, 4).float(),
            reference_logits=torch.arange(4).float(),
            dna=torch.tensor([0, 1, 2, 3], dtype=torch.int8),
            mutation_start=14,
            mutation_end=18,
            output_start=6,
            output_end=10,
        ),
    )


def test_write_and_read_profile_ism_h5(tmp_path):
    output_path = tmp_path / "profile_ism.h5"

    count = write_profile_ism_h5(
        output_path,
        [_region(np.str_("region_a")), _region("region_b")],
        metadata={"checkpoint": np.str_("model.ckpt"), "length": 4},
    )

    assert count == 2
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["schema"] == "smfnet.profile_ism"
        assert handle.attrs["schema_version"] == 1
        assert handle.attrs["num_regions"] == 2
        assert handle.attrs["checkpoint"] == "model.ckpt"
        assert handle["region_ids"].asstr()[:].tolist() == [
            "region_a",
            "region_b",
        ]
        assert handle["mutation_start"][:].tolist() == [14, 14]
        assert handle["dna"].shape == (2, 4)
        assert handle["influence"].shape == (2, 4, 4)
        assert handle["scaled_influence"].shape == (2, 4, 4)
        assert handle["base_deltas"].shape == (2, 4, 4, 4)
        assert handle["reference_logits"].shape == (2, 4)
        assert handle["base_deltas"].chunks == (1, 4, 4, 4)

    stored = list(read_profile_ism_h5(output_path))
    assert [region.region_id for region in stored] == ["region_a", "region_b"]
    assert torch.equal(stored[0].result.dna, _region("region_a").result.dna)
    assert torch.equal(
        stored[0].result.base_deltas,
        _region("region_a").result.base_deltas,
    )
