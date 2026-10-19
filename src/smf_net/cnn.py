from dataclasses import dataclass

import torch
from torch import nn


class FiLM(nn.Module):
    """Memory-efficient FiLM with implicit broadcasting.

    Expects:
        x: (Batch * Reads, Channels, Length)
        z: (Batch, 2 * Channels, Length)
    """

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B_total, C, L = x.shape
        B_cond = z.shape[0]

        if B_total % B_cond != 0:
            raise ValueError(f"SMF batch {B_total} is not divisible by DNA batch {B_cond}")

        reads_per_region = B_total // B_cond

        # View x as (B, R, C, L)
        x_view = x.view(B_cond, reads_per_region, C, L)

        # View z as (B, 1, 2C, L)
        z_view = z.view(B_cond, 1, 2 * C, L)

        # Chunk z along the channel dimension (dim 2)
        s, b = z_view.chunk(2, dim=2)

        # Apply FiLM: y = x + (x ⊙ s) + b, broadcasting (B, R, C, L) * (B, 1, C, L)
        out = x_view + x_view * s + b

        # Return to (N, C, L)
        return out.view(B_total, C, L)


class CNNLayerNorm1D(nn.Module):
    """
    ConvNeXt-style LayerNorm for 1D convs:
    - input: (N, C, L)
    - normalize over C for each (N, L)
    """

    def __init__(self, n_filters: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # shape (1, C, 1) so it broadcasts over N and L
        self.weight = nn.Parameter(torch.ones(1, n_filters, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_filters, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        mean = x.mean(dim=1, keepdim=True)  # (N, 1, L)
        var = x.var(dim=1, unbiased=False, keepdim=True)  # (N, 1, L)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # (N, C, L)
        return x_hat * self.weight + self.bias


class CNNBlock(nn.Module):
    """Dilated residual block with optional FiLM conditioning.

    Args:
        x: Tensor of shape (B * R, n_filters, L).
        dna: Tensor of shape (B, dna_emb_dim, L) with per-region conditioning,
            required when the block is conditional.

    Returns:
        Tensor of shape (B * R, n_filters, L).
    """

    def __init__(
        self,
        n_filters: int,
        dilation: int,
        dna_emb_dim: int | None = None,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            CNNLayerNorm1D(n_filters),
            nn.Conv1d(
                n_filters,
                n_filters,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.is_conditional = dna_emb_dim is not None
        if self.is_conditional:
            # Map DNA dim -> FiLM scale and shift
            self.cond_proj = nn.Sequential(
                CNNLayerNorm1D(dna_emb_dim),
                nn.Conv1d(dna_emb_dim, 2 * n_filters, kernel_size=1, padding="same"),
            )
            self.cond_op = FiLM()

    def forward(self, x: torch.Tensor, dna: torch.Tensor | None = None) -> torch.Tensor:
        h = self.cond_op(x, self.cond_proj(dna)) if self.is_conditional else x
        return self.conv(h) + x  # residual


class CNNBody(nn.Module):
    """Stack of dilated CNNBlocks, conditioned on DNA if dna_emb_dim is set."""

    def __init__(
        self,
        n_filters: int,
        num_dilation_blocks: int,
        dna_emb_dim: int | None = None,
        dropout: float = 0,
        kernel_size: int = 9,
    ):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.blocks = nn.ModuleList(
            CNNBlock(
                n_filters,
                dilation=dilation,
                dna_emb_dim=dna_emb_dim,
                dropout=dropout,
                kernel_size=kernel_size,
            )
            for _ in range(num_dilation_blocks)
            for dilation in dilations
        )

    def forward(self, x: torch.Tensor, dna: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, dna)
        return x


class TakeMiddle(nn.Module):
    """Extracts the middle half of the DNA embedding, which is twice the SMF length."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        assert L % 4 == 0
        return x[..., L // 4 : -L // 4].contiguous()


@dataclass
class CNNConfig:
    input_channels: int = 3
    output_channels: int = 1
    n_filters: int = 128
    num_dilation_blocks: int = 1
    dna_dim: int = 4
    dropout: float = 0
    input_kernel_size: int = 21
    kernel_size: int = 9
    final_kernel_size: int = 75

    # dna encoder
    dna_encoder_filters: int = 512
    dna_encoder_dilation_blocks: int = 1
    dna_filter_size: int = 21  # same as BPNet


class CNN(nn.Module):
    """1D CNN backbone that conditions on DNA to denoise SMF reads.

    The DNA input is twice the SMF length; the encoder keeps the middle half
    of its embedding, so the SMF window sees flanking sequence context.

    Args:
        smf (Tensor): Shape (B, R, input_channels, L).
        dna (Tensor | None): Shape (B, dna_dim, 2L) one-hot DNA.
        dna_embed (Tensor | None): Shape (B, dna_encoder_filters, L), the
            output of dna_encoder. Pass exactly one of dna and dna_embed.

    Returns:
        Tensor of shape (B, R, output_channels, L).
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg

        # DNA encoder
        self.dna_encoder = nn.Sequential(
            nn.Conv1d(
                cfg.dna_dim,
                cfg.dna_encoder_filters,
                cfg.dna_filter_size,
                padding="same",
            ),
            CNNBody(
                n_filters=cfg.dna_encoder_filters,
                num_dilation_blocks=cfg.dna_encoder_dilation_blocks,
                dna_emb_dim=None,
                dropout=cfg.dropout,
                kernel_size=cfg.kernel_size,
            ),
        )
        self.dna_encoder.add_module("take_middle", TakeMiddle())

        # SMF predictor
        self.input_conv = nn.Conv1d(
            cfg.input_channels,
            cfg.n_filters,
            kernel_size=cfg.input_kernel_size,  # Same as BPNet
            padding="same",
        )
        self.body = CNNBody(
            n_filters=cfg.n_filters,
            num_dilation_blocks=cfg.num_dilation_blocks,
            dna_emb_dim=cfg.dna_encoder_filters,
            dropout=cfg.dropout,
            kernel_size=cfg.kernel_size,
        )
        self.output_conv = nn.Conv1d(
            cfg.n_filters,
            cfg.output_channels,
            kernel_size=cfg.final_kernel_size,
            padding="same",
        )

    def forward(
        self,
        x: torch.Tensor,  # SMF
        dna: torch.Tensor | None = None,
        dna_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (dna is None) == (dna_embed is None):
            raise ValueError("Pass exactly one of dna and dna_embed")
        if x.dim() != 4:
            raise ValueError(f"Expected SMF tensor with shape (B, R, C, L), got {x.shape}")
        batch_size, reads_per_region, channels, length = x.shape
        x = x.view(batch_size * reads_per_region, channels, length)
        if x.dtype != self.input_conv.weight.dtype:
            x = x.to(dtype=self.input_conv.weight.dtype)
        x = self.input_conv(x)
        if dna is not None:
            dna_embed = self.dna_encoder(dna)
        if dna_embed.dtype != x.dtype:
            dna_embed = dna_embed.to(x.dtype)
        if dna_embed.shape[-1] != x.shape[-1]:
            raise ValueError("DNA embedding length does not match SMF input length.")
        x = self.body(x, dna_embed)
        out = self.output_conv(x)
        out = out.view(batch_size, reads_per_region, self.cfg.output_channels, length)
        return out
