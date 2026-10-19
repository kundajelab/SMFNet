import pytest
import torch

from smf_net.cnn import CNNConfig
from smf_net.data import dna_str_to_tensor
from smf_net.evaluation.prism import _prepare_read_trajectory, prism, prism_log_ratio
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


def test_prism_scores_each_substitution_against_reference(model):
    dna = dna_str_to_tensor("CG" * 16)
    read = torch.arange(32).remainder(2).to(torch.int8)
    read[15] = -1

    result = prism(
        model,
        read,
        dna,
        length=4,
        sample_steps=2,
        batch_size=7,
        n_unmasking_orders=3,
        seed=5,
    )

    assert result.scores.shape == (4, 4)
    assert result.deltas.shape == (4, 4)
    assert result.reference_score.ndim == 0
    assert result.per_order_scores.shape == (3, 4, 4)
    assert result.per_order_reference_scores.shape == (3,)
    assert torch.equal(result.read, read[14:18])

    # Reference bases take the reference score, so their deltas are exactly 0.
    positions = torch.arange(4)
    assert torch.equal(result.deltas[positions, result.dna.long()], torch.zeros(4))


def test_prism_reference_bases_take_the_reference_score(model):
    dna = dna_str_to_tensor("ACGT" * 8)
    dna[16] = -1
    read = torch.arange(32).remainder(2).to(torch.int8)

    result = prism(
        model,
        read,
        dna,
        length=4,
        mutation_start=14,
        batch_size=3,
        n_unmasking_orders=2,
    )

    wildtype = dna[14:18]
    for offset, base in enumerate(wildtype.tolist()):
        if base >= 0:
            assert torch.equal(
                result.per_order_scores[:, offset, base],
                result.per_order_reference_scores,
            )
            assert torch.all(result.deltas[offset, base] == 0)


def test_prism_encodes_each_sequence_once_across_orders(model):
    dna = dna_str_to_tensor("ACGT" * 8)
    dna[16] = -1
    read = torch.arange(32).remainder(2).to(torch.int8)
    encoded = []
    handle = model.model.dna_encoder.register_forward_hook(
        lambda module, inputs, output: encoded.append(len(inputs[0]))
    )
    try:
        prism(
            model,
            read,
            dna,
            length=4,
            mutation_start=14,
            batch_size=3,
            n_unmasking_orders=3,
        )
    finally:
        handle.remove()

    # Reference once, then 3 alternatives at each known base and all 4 at the N.
    assert sum(encoded) == 1 + 3 * 3 + 4


def test_prism_log_ratio_is_zero_for_identical_reads(model):
    dna = dna_str_to_tensor("CG" * 16)
    read = torch.arange(32).remainder(2).to(torch.int8)

    result = prism_log_ratio(
        model,
        read,
        read,
        dna,
        length=4,
        sample_steps=2,
        batch_size=8,
        n_unmasking_orders=2,
        seed=7,
    )

    assert result.scores.shape == (4, 4)
    assert result.per_order_scores.shape == (2, 4, 4)
    assert result.per_order_reference_scores.shape == (2,)
    assert torch.equal(result.shared_valid_mask, torch.ones(4, dtype=torch.bool))
    assert torch.allclose(result.reference_score, torch.tensor(0.0))
    assert torch.allclose(result.scores, torch.zeros((4, 4)))
    assert torch.allclose(result.deltas, torch.zeros((4, 4)))


def test_log_ratio_uses_the_same_orders_as_prism(model):
    dna = dna_str_to_tensor("CG" * 16)
    read_a = torch.arange(32).remainder(2).to(torch.int8)
    read_b = 1 - read_a
    kwargs = {
        "length": 2,
        "sample_steps": 2,
        "batch_size": 8,
        "n_unmasking_orders": 2,
        "seed": 11,
    }

    result_a = prism(model, read_a, dna, **kwargs)
    result_b = prism(model, read_b, dna, **kwargs)
    ratio = prism_log_ratio(model, read_a, read_b, dna, **kwargs)

    assert torch.allclose(
        ratio.per_order_scores,
        result_a.per_order_scores - result_b.per_order_scores,
    )
    assert torch.allclose(
        ratio.per_order_reference_scores,
        result_a.per_order_reference_scores - result_b.per_order_reference_scores,
    )
    assert torch.allclose(
        ratio.scores,
        result_a.scores - result_b.scores,
        atol=1e-6,
    )
    assert torch.allclose(
        ratio.deltas,
        result_a.deltas - result_b.deltas,
        atol=1e-6,
    )


@pytest.mark.parametrize("sampling_mode", ["cosine", "one_pos_at_a_time"])
def test_likelihood_window_controls_scoring_and_context(model, sampling_mode):
    read = torch.arange(32).remainder(2)
    read[15] = -1
    selected = torch.zeros(16, dtype=torch.bool)
    selected[6:10] = True
    order = torch.arange(16)
    for context_mode in ("masked", "full_read"):
        trajectory = _prepare_read_trajectory(
            model,
            read,
            32,
            2,
            order,
            sampling_mode,
            score_positions=selected,
            context_mode=context_mode,
        )
        # Output positions 6:10 correspond to input positions 14:18;
        # the missing call at input position 15 must never contribute.
        expected = selected.clone()
        expected[7] = False
        assert torch.equal(trajectory.score_mask.sum(dim=0), expected.long())
        tokens = trajectory.inputs.argmax(dim=2)[0]
        if context_mode == "masked":
            assert (tokens[:, ~selected] == 2).all()
            assert (tokens[:, 7] == 2).all()
        else:
            assert (tokens[:, ~selected] != 2).any()


@pytest.mark.parametrize("sampling_mode", ["cosine", "one_pos_at_a_time"])
def test_masked_likelihood_is_independent_of_outside_read(model, sampling_mode):
    dna = dna_str_to_tensor("CG" * 16)
    read_a = torch.arange(32).remainder(2)
    read_b = 1 - read_a
    read_b[14:18] = read_a[14:18]
    kwargs = {
        "length": 2,
        "batch_size": 8,
        "n_unmasking_orders": 2,
        "likelihood_window": (6, 10),
        "context_mode": "masked",
        "sampling_mode": sampling_mode,
    }
    result_a = prism(model, read_a, dna, **kwargs)
    result_b = prism(model, read_b, dna, **kwargs)
    assert torch.equal(result_a.per_order_scores, result_b.per_order_scores)
    assert torch.equal(result_a.per_order_reference_scores, result_b.per_order_reference_scores)
    assert (result_a.likelihood_start, result_a.likelihood_end) == (6, 10)
    assert result_a.context_mode == "masked"


@pytest.mark.parametrize("context_mode", ["masked", "full_read"])
def test_full_likelihood_window_preserves_default_scores(model, context_mode):
    dna = dna_str_to_tensor("CG" * 16)
    read = torch.arange(32).remainder(2)
    kwargs = {"length": 2, "batch_size": 8, "n_unmasking_orders": 1}
    default = prism(model, read, dna, **kwargs)
    explicit = prism(
        model,
        read,
        dna,
        likelihood_window=(0, 16),
        context_mode=context_mode,
        **kwargs,
    )
    assert torch.equal(default.per_order_scores, explicit.per_order_scores)
    assert torch.equal(default.reference_score, explicit.reference_score)


@pytest.mark.parametrize("sampling_mode", ["cosine", "one_pos_at_a_time"])
def test_masked_window_with_no_observed_calls_has_zero_score(model, sampling_mode):
    read = torch.zeros(32, dtype=torch.long)
    read[14:18] = -1
    result = prism(
        model,
        read,
        dna_str_to_tensor("CG" * 16),
        length=2,
        likelihood_window=(6, 10),
        context_mode="masked",
        sampling_mode=sampling_mode,
        n_unmasking_orders=1,
        batch_size=8,
    )
    assert result.reference_score == 0
    assert (result.scores == 0).all()
    assert (result.deltas == 0).all()
