import pytest
import torch

from smf_net.cnn import CNNConfig
from smf_net.data import dna_str_to_tensor
from smf_net.evaluation.ism import ProfileISMConfig, profile_ism, profile_ism_regions
from smf_net.evaluation.regions import SequenceRegion
from smf_net.model import SMFNet, SMFNetConfig


@pytest.fixture
def model() -> SMFNet:
    config = CNNConfig(
        n_filters=4,
        num_dilation_blocks=1,
        kernel_size=3,
        final_kernel_size=3,
        dna_encoder_filters=4,
        dna_encoder_dilation_blocks=1,
        dna_filter_size=3,
        input_kernel_size=3,
    )
    torch.manual_seed(0)
    return SMFNet(SMFNetConfig(cnn_config=config, input_length=32, sample_steps=2)).eval()


def test_profile_ism_returns_profile_and_per_base_deltas(model):
    dna = dna_str_to_tensor("CG" * 16)

    result = profile_ism(
        model,
        dna,
        length=4,
        batch_size=5,
        importance_window=3,
    )

    assert result.influence.shape == (4, 4)
    assert result.scaled_influence.shape == (4, 4)
    assert result.base_deltas.shape == (4, 4, 4)
    assert result.reference_logits.shape == (4,)
    assert (result.mutation_start, result.mutation_end) == (14, 18)
    assert (result.output_start, result.output_end) == (6, 10)

    # Substituting a base with itself changes nothing, up to float32 rounding
    # between the reference (scored alone) and the batched mutations.
    positions = torch.arange(4)
    assert torch.allclose(
        result.base_deltas[positions, result.dna.long()],
        torch.zeros((4, 4)),
        atol=1e-5,
    )


def test_profile_ism_regions_preserves_region_names(model):
    regions = [
        SequenceRegion("a", dna_str_to_tensor("CG" * 16)),
        SequenceRegion("b", dna_str_to_tensor("GC" * 16)),
    ]

    results = list(
        profile_ism_regions(
            model,
            regions,
            ProfileISMConfig(length=2, batch_size=8, importance_window=3),
        )
    )

    assert [region.region_id for region in results] == ["a", "b"]
    assert all(region.result.influence.shape == (2, 2) for region in results)
