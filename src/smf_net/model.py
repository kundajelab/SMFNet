import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from smf_net.cnn import CNN, CNNConfig
from smf_net.data import (
    AssayType,
    SMFItem,
    SparseDataModuleConfig,
    dna_tensor_to_one_hot,
    get_valid_mask,
)
from smf_net.diffusion_schedule import MASK_TOKEN, CosineDiffusionSchedule
from smf_net.lr_scheduler import LinearWarmupCosineAnnealingLR


@dataclass
class SMFNetConfig:
    cnn_config: CNNConfig
    assay_type: AssayType = "cg_gc_accessibility"
    input_length: int = 2048
    lr: float = 1e-3
    wd: float = 1e-2
    sample_steps: int = 32

    def __post_init__(self) -> None:
        # A config.json from the Hugging Face Hub holds the CNN config as a dict.
        if isinstance(self.cnn_config, dict):
            self.cnn_config = CNNConfig(**self.cnn_config)
        # The SMF window is the middle half of the DNA input.
        if self.input_length % 4:
            raise ValueError("input_length must be divisible by 4")


def validate_data_model_compatibility(
    data_cfg: SparseDataModuleConfig,
    model_cfg: SMFNetConfig,
) -> None:
    """Require observed data to match the model's input and assay contract."""
    if data_cfg.model_input_length != model_cfg.input_length:
        raise ValueError(
            "Data model_input_length must match model input_length, got "
            f"{data_cfg.model_input_length} and {model_cfg.input_length}"
        )
    if data_cfg.assay_type != model_cfg.assay_type:
        raise ValueError(
            "Data assay_type must match model assay_type, got "
            f"{data_cfg.assay_type!r} and {model_cfg.assay_type!r}"
        )


# The mixin adds from_pretrained, save_pretrained and push_to_hub, storing the
# weights as safetensors and the config as JSON. Its keyword arguments are the
# metadata of the model card that save_pretrained and push_to_hub write, which
# paper_analysis/prep/export_hf.py also uses for the released models.
class SMFNet(
    LightningModule,
    PyTorchModelHubMixin,
    library_name="smfnet",
    license="mit",
    repo_url="https://github.com/kundajelab/SMFNet",
    tags=["single-molecule-footprinting"],
):
    # Validation loss by masking level, enabled by setting ``log_mask_rate_bins``
    # on the module (scripts/train.py does this). Bins are on each read's mask
    # rate, 1 - alpha_t(t), the probability that a call is hidden.
    log_mask_rate_bins = False
    MASK_RATE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    @classmethod
    def _from_pretrained(cls, *, strict: bool = True, **kwargs) -> "SMFNet":
        # The mixin loads non-strictly by default, which would leave any weight
        # missing from the file at its random initialization without a warning.
        return super()._from_pretrained(strict=strict, **kwargs)

    def __init__(self, cfg: SMFNetConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        # Not compiled here; callers that want speed call self.model.compile(),
        # which compiles in place and leaves parameter names unchanged.
        self.model = CNN(cfg.cnn_config)
        self.schedule = CosineDiffusionSchedule()
        self._mask_rate_sums = None

    def forward(self, smf: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        _, _, dna_len = dna.shape
        self._validate_input_length(dna_len)
        smf_len = smf.shape[-1]
        if smf_len != dna_len:
            raise ValueError(f"SMF length must match DNA length, got {smf_len} and {dna_len}")
        smf = self._center_crop_smf(smf.long(), dna_len)
        smf = self._encode_smf_tokens(smf)
        return self.model(smf, dna)

    def training_step(self, batch: SMFItem) -> torch.Tensor:
        return self.get_loss(batch, "train")

    def validation_step(self, batch: SMFItem, *args) -> torch.Tensor:
        return self.get_loss(batch, "val")

    def get_loss(self, batch: SMFItem, prefix: str) -> torch.Tensor:
        batch = batch.to(self.device, self.dtype)
        dna = batch.dna  # bsz x 4 x L
        bsz, _, dna_len = dna.shape
        self._validate_input_length(dna_len)
        x_0 = batch.smf.long()  # bsz x reads_per_region x L
        x_0 = self._center_crop_smf(x_0, dna_len)

        valid_mask = x_0 != -1
        reads_per_region = x_0.shape[1]

        # low discrepancy sampling
        base = torch.linspace(
            0, 1 - 1 / reads_per_region, steps=reads_per_region, device=x_0.device
        )
        t_offset = torch.rand(bsz, 1, device=x_0.device)
        t = (base + t_offset) % 1
        x_t = self.schedule.p_t(x_0, t).sample()
        loss_mask = valid_mask & (x_t == MASK_TOKEN)  # bsz x reads_per_region x L
        x_t_onehot = self._encode_smf_tokens(x_t)
        x_0 = x_0.float().unsqueeze(2)  # bsz x reads x 1 x L

        # loss function
        logits = self.model(x_t_onehot, dna)
        bce_loss = F.binary_cross_entropy_with_logits(logits, x_0, reduction="none").squeeze(
            2
        )  # bsz x reads x L
        loss_weight = self.schedule.cross_entropy_loss_weight(t)
        weighted = bce_loss * loss_weight * loss_mask.float()
        if prefix == "val" and self.log_mask_rate_bins:
            self._accumulate_mask_rate_bins(t, bce_loss, weighted, loss_mask, valid_mask)
        loss = weighted.sum() / valid_mask.float().sum()
        self.smart_log("loss", loss, prefix)
        return loss

    def smart_log(self, name: str, value, prefix: str):
        if isinstance(value, torch.Tensor):
            value = value.detach()
        self.log(
            f"{prefix}/{name}",
            value,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _accumulate_mask_rate_bins(self, t, bce_loss, weighted, loss_mask, valid_mask) -> None:
        mask_rate = 1 - self.schedule.alpha_t(t)  # bsz x reads
        edges = torch.tensor(self.MASK_RATE_EDGES[1:-1], device=t.device)
        bins = torch.bucketize(mask_rate, edges).flatten()
        masked = loss_mask.float()
        # Per read: masked cross-entropy, masked calls, weighted loss, valid calls.
        per_read = (
            torch.stack(
                [
                    (bce_loss.detach() * masked).sum(-1),
                    masked.sum(-1),
                    weighted.detach().sum(-1),
                    valid_mask.float().sum(-1),
                ],
                dim=-1,
            )
            .reshape(-1, 4)
            .double()
        )
        sums = torch.zeros(len(self.MASK_RATE_EDGES) - 1, 4, device=t.device, dtype=torch.double)
        sums.index_add_(0, bins, per_read)
        self._mask_rate_sums = sums if self._mask_rate_sums is None else self._mask_rate_sums + sums

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("train/epoch_time", epoch_time, prog_bar=False, on_epoch=True, on_step=False)

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()
        self._mask_rate_sums = None

    def on_validation_epoch_end(self):
        epoch_time = time.time() - self.val_epoch_start_time
        self.log("val/epoch_time", epoch_time, prog_bar=False, on_epoch=True, on_step=False)
        sums = self._mask_rate_sums
        if sums is None:
            return
        total_valid = sums[:, 3].sum().clamp_min(1)
        for i, (lo, hi) in enumerate(zip(self.MASK_RATE_EDGES[:-1], self.MASK_RATE_EDGES[1:])):
            name = f"m{lo:.1f}-{hi:.1f}"
            # Cross-entropy per hidden call, and this bin's share of the total
            # weighted loss (the shares sum to roughly val/loss).
            self.log(
                f"val_mask/xent_{name}",
                float(sums[i, 0] / sums[i, 1].clamp_min(1)),
                on_epoch=True,
                on_step=False,
            )
            self.log(
                f"val_mask/loss_share_{name}",
                float(sums[i, 2] / total_valid),
                on_epoch=True,
                on_step=False,
            )
        self._mask_rate_sums = None

    def configure_optimizers(self):
        """AdamW with linear warmup then cosine annealing, stepped every batch.

        lr and wd come from the model config; biases get no weight decay.
        """
        lr = self.hparams.cfg.lr
        wd = self.hparams.cfg.wd

        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        # Use the Trainer's max_epochs to define the scheduler's max_epochs
        if self.trainer is None:
            raise RuntimeError("Trainer must be attached before calling configure_optimizers")

        max_epochs = self.trainer.max_epochs
        warmup_epochs = min(max(1, max_epochs // 20), 10)
        total_batches = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_batches // max_epochs
        print(f"Estimated steps per epoch: {steps_per_epoch}")

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs * steps_per_epoch,
            max_epochs=max_epochs * steps_per_epoch,
            warmup_start_lr=lr / 100,
            eta_min=lr / 100,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _center_crop_smf(self, smf: torch.Tensor, dna_len: int) -> torch.Tensor:
        """Keep the middle half of full-length SMF, the window the model predicts."""
        return smf[..., dna_len // 4 : -dna_len // 4].contiguous()

    def _validate_input_length(self, input_length: int) -> None:
        if input_length != self.cfg.input_length:
            raise ValueError(
                f"Expected DNA input length {self.cfg.input_length}, got {input_length}"
            )

    def _autocast_context(self, compute_dtype: torch.dtype):
        if compute_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError(f"Unsupported sampling compute dtype: {compute_dtype}")
        if (
            compute_dtype != torch.float32
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        ):
            return torch.autocast(device_type="cuda", dtype=compute_dtype)
        return nullcontext()

    def _encode_smf_tokens(self, smf: torch.Tensor) -> torch.Tensor:
        """Convert integer SMF tokens to one-hot tensors expected by the CNN."""
        return (
            F.one_hot(smf.long(), num_classes=MASK_TOKEN + 1)
            .permute(0, 1, 3, 2)
            .to(dtype=self.model.input_conv.weight.dtype, device=self.device)
            .contiguous()
        )

    @torch.no_grad()
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
        """Generate reads, optionally keeping SMF outside an output window masked.

        ``sampling_window`` is a half-open interval in center-cropped SMF
        output coordinates. Full DNA context is retained.
        """
        sample_steps = sample_steps or self.cfg.sample_steps
        # prepare fully masked SMF input
        bsz, L = dna.shape
        self._validate_input_length(L)
        smf_L = L // 2
        x_t = torch.full(
            (bsz, reads_per_region, smf_L), MASK_TOKEN, device=self.device, dtype=torch.long
        )

        # prepare DNA embedding
        one_hot_dna = dna_tensor_to_one_hot(dna).to(device=self.device, dtype=self.dtype)
        valid_mask = self._center_crop_smf(get_valid_mask(dna, self.cfg.assay_type), L)
        if sampling_window is not None:
            if len(sampling_window) != 2:
                raise ValueError("sampling_window must contain (start, end)")
            start, end = sampling_window
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= smf_L
            ):
                raise ValueError(
                    "sampling_window must be a nonempty interval inside the SMF output"
                )
            window_mask = torch.zeros_like(valid_mask)
            window_mask[:, start:end] = True
            valid_mask &= window_mask
        valid_mask = valid_mask.unsqueeze(1).expand(-1, reads_per_region, -1).to(self.device)

        # only compute DNA embedding once
        with self._autocast_context(compute_dtype):
            dna_embed = self.model.dna_encoder(one_hot_dna)

        # prepare sampling loop
        total_valid = int(valid_mask.sum().item())
        if total_valid == 0:
            raise ValueError("No valid positions in DNA sequence for sampling.")

        # sample loop
        for step_idx in tqdm(range(sample_steps), desc="sampling", disable=not verbose):
            # get SMF predictions at each position
            x_t_onehot = self._encode_smf_tokens(x_t)
            with self._autocast_context(compute_dtype):
                model_logits = self.model(x_t_onehot, dna_embed=dna_embed)
            x_t = self.tau_leap_sample_step(
                step_idx,
                valid_mask=valid_mask,
                x_t=x_t,
                model_logits=model_logits,
                sample_steps=sample_steps,
                generator=generator,
            )

        return x_t

    def tau_leap_sample_step(
        self,
        step_idx: int,
        *,
        valid_mask: torch.Tensor,
        x_t: torch.Tensor,
        model_logits: torch.Tensor,
        sample_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Perform a single tau-leap sampling step for the diffusion process.
        """
        delta_t = 1 / sample_steps
        t = torch.tensor(1 - step_idx * delta_t, device=x_t.device)
        t = torch.clamp(t, min=delta_t)
        s = torch.clamp(t - delta_t, min=0.0)
        alpha_s = self.schedule.alpha_t(s)
        alpha_t = self.schedule.alpha_t(t)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        unmask_prob = unmask_prob.clamp_min(0.01).item()
        candidate_mask = (x_t == MASK_TOKEN) & valid_mask

        # Unmask each still-masked site with probability unmask_prob, and every
        # remaining site on the last step.
        if step_idx == sample_steps - 1:
            unmask = candidate_mask
        else:
            random_values = torch.rand(
                x_t.shape, dtype=torch.float32, device=x_t.device, generator=generator
            )
            unmask = (random_values < unmask_prob) & candidate_mask

        # Ensure at least one site is unmasked per (batch, read) if any are available.
        needs_fallback = (unmask.sum(dim=-1) == 0) & candidate_mask.any(dim=-1)

        fallback_scores = torch.rand(
            x_t.shape, dtype=torch.float32, device=x_t.device, generator=generator
        )
        fallback_scores = fallback_scores.masked_fill(~candidate_mask, -1)
        fallback_idx = fallback_scores.argmax(dim=-1, keepdim=True)

        fallback = torch.zeros_like(candidate_mask)
        fallback.scatter_(-1, fallback_idx, True)
        fallback &= candidate_mask

        unmask |= fallback & needs_fallback.unsqueeze(-1)

        # sample from the model's predicted probabilities at the unmasked sites
        model_probs = torch.sigmoid(model_logits.squeeze(2))
        x_t_sample = torch.bernoulli(model_probs, generator=generator).long()

        return torch.where(unmask, x_t_sample, x_t)


def load_model(source: str, map_location: str | torch.device = "cpu") -> SMFNet:
    """Load a trained SMFNet for inference, in eval mode.

    ``source`` is a Lightning checkpoint file (``.ckpt``), or else a directory
    written by ``save_pretrained`` or a Hugging Face Hub repository ID.
    """

    if Path(source).suffix == ".ckpt":
        model = SMFNet.load_from_checkpoint(source, map_location=map_location)
    else:
        model = SMFNet.from_pretrained(source, map_location=str(map_location))
    return model.eval()
