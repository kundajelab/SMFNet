import pytest
import torch
import torch.nn.functional as F

from smf_net.cnn import CNNConfig
from smf_net.data import (
    SMFItem,
    SparseSMFDataset,
    dna_str_to_tensor,
    get_valid_mask,
)
from smf_net.diffusion_schedule import MASK_TOKEN
from smf_net.model import SMFNet, SMFNetConfig, load_model


def _build_model(assay_type="cg_gc_accessibility") -> SMFNet:
    cnn_cfg = CNNConfig(
        n_filters=8,
        num_dilation_blocks=1,
        kernel_size=3,
        final_kernel_size=3,
        dna_encoder_filters=6,
        dna_encoder_dilation_blocks=1,
        dna_filter_size=3,
        input_kernel_size=3,
    )
    model_cfg = SMFNetConfig(
        cnn_config=cnn_cfg,
        assay_type=assay_type,
        input_length=32,
        sample_steps=3,
    )
    model = SMFNet(model_cfg)
    model.eval()
    return model


def _random_one_hot_dna(batch_size: int, length: int) -> torch.Tensor:
    dna_tokens = torch.randint(0, 4, (batch_size, length))
    return F.one_hot(dna_tokens, num_classes=4).permute(0, 2, 1).float()


def test_forward_loss_and_sampling_shapes():
    model = _build_model()
    batch_size, reads_per_region, length = 2, 3, 32
    dna = _random_one_hot_dna(batch_size, length)

    logits = model(torch.randint(0, MASK_TOKEN + 1, (batch_size, reads_per_region, length)), dna)
    assert logits.shape == (
        batch_size,
        reads_per_region,
        model.cfg.cnn_config.output_channels,
        length // 2,
    )

    smf = torch.randint(0, 2, (batch_size, reads_per_region, length))
    smf = smf.masked_fill(torch.rand(smf.shape) < 0.1, -1)
    loss = model.get_loss(SMFItem(smf=smf, dna=dna), prefix="val")
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    samples = model.sample(
        dna_str_to_tensor("CG" * (length // 2)).repeat(batch_size, 1),
        reads_per_region=reads_per_region,
    )
    assert samples.shape == (batch_size, reads_per_region, length // 2)


@pytest.mark.parametrize(
    ("assay_type", "sequence", "positions"),
    [
        pytest.param(
            "cg_gc_accessibility",
            "CG" * 16,
            torch.arange(0, 32, 2),
            id="cg-gc-accessibility",
        ),
        pytest.param(
            "gch_accessibility",
            "GCA" * 10 + "AA",
            torch.arange(1, 30, 3),
            id="gch-accessibility",
        ),
        pytest.param(
            "fiberseq_m6a",
            "ATGC" * 8,
            torch.tensor([i for i in range(32) if i % 4 in (0, 1)]),
            id="fiberseq-m6a",
        ),
    ],
)
def test_assay_filtered_dataset_batch_has_finite_training_loss(assay_type, sequence, positions):
    model = _build_model(assay_type=assay_type)
    smf = torch.randint(0, 2, (4, len(positions)), dtype=torch.int8)
    dataset = SparseSMFDataset(
        [smf],
        [positions],
        dna_str_to_tensor(sequence).unsqueeze(0),
        reads_per_region=2,
        model_input_length=32,
        assay_type=assay_type,
    )
    batch = SparseSMFDataset.collate_fn([dataset[0]])

    loss = model.get_loss(batch, prefix="train")
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    ("assay_type", "dna_seq"),
    [
        ("cg_gc_accessibility", 8 * "A" + "CGCGATATCGCGATAT" + 8 * "A"),
        ("gch_accessibility", 8 * "A" + "GCACGCAGCGAGCNGC" + 8 * "A"),
    ],
)
def test_sampling_only_unmasks_eligible_sites(assay_type, dna_seq):
    torch.manual_seed(0)
    model = _build_model(assay_type=assay_type)
    dna = dna_str_to_tensor(dna_seq).unsqueeze(0)

    samples = model.sample(dna, reads_per_region=3).squeeze(0)

    # Match the sampler: identify sites first, then crop.
    valid_mask = get_valid_mask(dna, assay_type)[0, 8:-8]
    assert valid_mask.any() and (~valid_mask).any()
    assert torch.all((samples[:, valid_mask] == 0) | (samples[:, valid_mask] == 1))
    assert torch.all(samples[:, ~valid_mask] == MASK_TOKEN)


def test_sampling_accepts_a_local_random_generator():
    model = _build_model()
    dna = dna_str_to_tensor("CG" * 16).unsqueeze(0)

    first = model.sample(dna, reads_per_region=4, generator=torch.Generator().manual_seed(17))
    second = model.sample(dna, reads_per_region=4, generator=torch.Generator().manual_seed(17))

    assert torch.equal(first, second)


def test_sampling_window_keeps_outside_positions_masked():
    model = _build_model()
    dna = dna_str_to_tensor("CG" * 16).unsqueeze(0)
    seen_inputs = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args: seen_inputs.append(args[0].argmax(dim=2).clone())
    )
    try:
        samples = model.sample(dna, reads_per_region=3, sampling_window=(6, 10))
    finally:
        hook.remove()
    outside = torch.ones(16, dtype=torch.bool)
    outside[6:10] = False
    assert (samples[:, :, outside] == MASK_TOKEN).all()
    informative = get_valid_mask(dna, model.cfg.assay_type)[0, 8:24]
    inside = informative & ~outside
    assert ((samples[:, :, inside] == 0) | (samples[:, :, inside] == 1)).all()
    assert (samples[:, :, ~informative] == MASK_TOKEN).all()
    assert all((inputs[:, :, outside] == MASK_TOKEN).all() for inputs in seen_inputs)


@pytest.mark.parametrize(
    "compute_dtype",
    [torch.bfloat16, torch.float16, torch.float32],
)
def test_sampling_compute_dtype_does_not_change_parameter_dtype(compute_dtype):
    model = _build_model().float()
    dna = dna_str_to_tensor("CG" * 16).unsqueeze(0)

    model.sample(
        dna,
        reads_per_region=1,
        compute_dtype=compute_dtype,
    )

    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "compute_dtype",
    [torch.bfloat16, torch.float16, torch.float32],
)
def test_sampling_uses_requested_cuda_compute_dtype(compute_dtype):
    model = _build_model().cuda().float()
    dna = dna_str_to_tensor("CG" * 16).unsqueeze(0).cuda()
    output_dtypes = []
    hook = model.model.input_conv.register_forward_hook(
        lambda _module, _inputs, output: output_dtypes.append(output.dtype)
    )

    try:
        model.sample(
            dna,
            reads_per_region=1,
            compute_dtype=compute_dtype,
        )
    finally:
        hook.remove()

    assert output_dtypes
    assert set(output_dtypes) == {compute_dtype}
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def _seeded_sample(model):
    dna = dna_str_to_tensor("CG" * 16).unsqueeze(0)
    return model.sample(dna, reads_per_region=2, generator=torch.Generator().manual_seed(0))


def test_checkpoints_and_saved_models_load_the_same_model(tmp_path):
    import pytorch_lightning as pl

    model = _build_model()
    trainer = pl.Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.strategy.connect(model)
    trainer.save_checkpoint(tmp_path / "model.ckpt")
    model.save_pretrained(tmp_path / "saved")

    assert "library_name: smfnet" in (tmp_path / "saved" / "README.md").read_text()
    for source in (tmp_path / "model.ckpt", tmp_path / "saved"):
        loaded = load_model(str(source))
        assert not loaded.training
        assert loaded.cfg == model.cfg
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value), (source, key)
        assert torch.equal(_seeded_sample(loaded), _seeded_sample(model))


def test_saved_models_missing_a_weight_are_refused(tmp_path):
    from safetensors.torch import load_file, save_file

    _build_model().save_pretrained(tmp_path)
    weights = load_file(tmp_path / "model.safetensors")
    del weights[next(iter(weights))]
    save_file(weights, tmp_path / "model.safetensors")

    with pytest.raises(RuntimeError, match="Missing key"):
        SMFNet.from_pretrained(tmp_path)
