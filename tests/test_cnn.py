import torch

from smf_net.cnn import CNN, CNNConfig


def _build_cnn_config() -> CNNConfig:
    return CNNConfig(
        n_filters=8,
        num_dilation_blocks=1,
        kernel_size=3,
        final_kernel_size=3,
        dropout=0.0,
        dna_encoder_filters=6,
        dna_encoder_dilation_blocks=1,
        dna_filter_size=3,
        input_kernel_size=3,
    )


def test_cnn_forward_with_dna_or_precomputed_embedding_matches():
    cfg = _build_cnn_config()
    model = CNN(cfg)
    batch_size, reads_per_region, length = 2, 3, 16
    smf = torch.randn(batch_size, reads_per_region, cfg.input_channels, length // 2)
    dna = torch.randn(batch_size, cfg.dna_dim, length)

    out_with_dna = model(smf, dna)
    out_with_embed = model(smf, dna_embed=model.dna_encoder(dna))

    assert out_with_dna.shape == (
        batch_size,
        reads_per_region,
        cfg.output_channels,
        length // 2,
    )
    assert torch.allclose(out_with_dna, out_with_embed, atol=1e-6, rtol=1e-5)


def test_cnn_repeated_reads_produce_identical_outputs():
    cfg = _build_cnn_config()
    model = CNN(cfg)
    batch_size, reads_per_region, length = 5, 3, 24
    base_smfs = torch.randn(batch_size, 1, cfg.input_channels, length // 2)
    # repeat each region's SMF across its reads
    smf = base_smfs.repeat(1, reads_per_region, 1, 1)
    dna = torch.randn(batch_size, cfg.dna_dim, length)

    output = model(smf, dna)  # shape (B, R, Cout, L)

    # Reads of one region match; each region gets its own DNA conditioning.
    first_reads = output[:, :1]
    assert torch.allclose(output, first_reads.expand_as(output), atol=1e-6, rtol=1e-5)
    differences = (first_reads[:, None, 0] - first_reads[None, :, 0]).abs()
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool)
    assert (differences.flatten(2).amax(dim=2)[off_diagonal] > 1e-5).all()
