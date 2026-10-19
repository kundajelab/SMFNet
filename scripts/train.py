import time
from itertools import islice
from pathlib import Path
from types import MappingProxyType

from hydra.core.hydra_config import HydraConfig
from hydra_zen import (
    ZenStore,
    make_custom_builds_fn,
    zen,
)
from hydra_zen.typing import Partial
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    EMAWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from smf_net.cli import store_help
from smf_net.cnn import CNNConfig
from smf_net.data import (
    AssayType,
    SparseDataModuleConfig,
    SparseSMFDataModule,
    validate_data_files,
)
from smf_net.model import (
    SMFNet,
    SMFNetConfig,
    validate_data_model_compatibility,
)

builds_full = make_custom_builds_fn(populate_full_signature=True)
builds_partial = make_custom_builds_fn(zen_partial=True)


def seed_before_instantiation(seed: int) -> None:
    seed_everything(seed)


def build_checkpoint() -> ModelCheckpoint:
    """Save checkpoints alongside Hydra's run configuration."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=HydraConfig.get().runtime.output_dir,
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
        save_on_train_epoch_end=False,
        every_n_train_steps=None,
        every_n_epochs=None,
        save_last=True,
    )
    return checkpoint_callback


class SampleEvalMonitor(Callback):
    """Sample a fixed set of validation regions every ``every_n_epochs`` epochs
    and log the median per-region metrics.py scores as ``sample/*``.

    The regions are the first ``n_regions`` of the validation split (a
    deterministic order), sampled with a fixed seed, so successive evaluations
    differ only by the model. Thresholds are the metrics.py defaults. Runs in
    ``on_validation_epoch_end``, so it scores the same weights as the
    validation loss (the EMA weights when EMA is on). The first evaluation is
    repeated with another seed to log the sampling noise
    (``sample/*_seed_difference``). Samples are kept in ``sample_eval/``.
    """

    COLUMNS = MappingProxyType(
        {
            "profile_pearson": "pearson",
            "profile_pearson_matched": "pearson_matched",
            "profile_pearson_replicate": "pearson_matched_pseudo_replicate",
            "coc_pearson": "covariation_corr_of_corr_pearson",
            "coc_pearson_replicate": "covariation_corr_of_corr_pearson_pseudo_replicate",
        }
    )

    def __init__(self, every_n_epochs: int, n_regions: int, reads: int, seed: int = 0) -> None:
        self.every_n_epochs = every_n_epochs
        self.n_regions = n_regions
        self.reads = reads
        self.seed = seed
        self.regions = None
        self.evaluations = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        datamodule = trainer.datamodule
        datamodule.setup("validate")
        self.regions = list(islice(datamodule.val_dataset.iter_regions(), self.n_regions))
        self.directory = Path(HydraConfig.get().runtime.output_dir) / "sample_eval"
        self.directory.mkdir(exist_ok=True)

    def _evaluate(self, pl_module, seed: int, path: Path) -> dict[str, float]:
        # Imported here, so training without this callback doesn't need the
        # evaluation code's dependencies.
        from smf_net.evaluation.metrics import MetricsConfig, compute_metrics_table
        from smf_net.evaluation.sample_io import read_samples_h5, write_samples_h5
        from smf_net.evaluation.sampling import SamplingConfig, sample_observed_regions

        config = SamplingConfig(
            reads_per_region=self.reads,
            reads_per_chunk=128,
            regions_per_batch=32,
            seed=seed,
            progress=False,
        )
        batches = sample_observed_regions(
            pl_module, iter(self.regions), config, total_regions=len(self.regions)
        )
        write_samples_h5(path, batches, metadata={"seed": seed, "reads_per_region": self.reads})
        metrics = MetricsConfig(
            include_shuffle=False,
            include_run_lengths=False,
            include_positional_baselines=False,
            include_flank_baselines=False,
            progress=False,
        )
        table = compute_metrics_table(read_samples_h5(path), metrics)
        scores = {
            name: float(table[column].median())
            for name, column in self.COLUMNS.items()
            if column in table
        }
        scores["n_regions"] = float(len(table))
        return scores

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        epoch = trainer.current_epoch
        start = time.time()
        scores = self._evaluate(pl_module, self.seed, self.directory / f"epoch_{epoch:04d}.h5")
        if self.evaluations == 0:
            repeat = self._evaluate(
                pl_module,
                self.seed + 1,
                self.directory / f"epoch_{epoch:04d}_seed{self.seed + 1}.h5",
            )
            for name in self.COLUMNS:
                if name in scores and name in repeat:
                    scores[f"{name}_seed_difference"] = abs(scores[name] - repeat[name])
        self.evaluations += 1
        scores["seconds"] = time.time() - start
        pl_module.log_dict(
            {f"sample/{k}": v for k, v in scores.items()}, on_step=False, on_epoch=True
        )


def main(
    seed: int,
    assay_type: AssayType,
    data: SparseSMFDataModule,
    model: SMFNet,
    trainer: Partial[Trainer],
    wandb_logger: WandbLogger,
    log_mask_rate_bins: bool = False,
    checkpoint_every_n_epochs: int = 0,
    sample_eval_every_n_epochs: int = 0,
    sample_eval_regions: int = 256,
    sample_eval_reads: int = 256,
    ema_decay: float = 0.0,
) -> None:
    """Train SMFNet.

    Opt-in extras (all off by default):

    - ``log_mask_rate_bins``: validation cross-entropy by masking level.
    - ``checkpoint_every_n_epochs``: also keep a checkpoint every N epochs.
    - ``sample_eval_every_n_epochs``: sample ``sample_eval_regions`` validation
      regions with ``sample_eval_reads`` reads every N epochs, log metrics.py
      scores, and keep the best checkpoint by sampled profile Pearson.
    - ``ema_decay``: if > 0, keep an exponential moving average of the weights,
      used for validation, sampling and saved checkpoints (the raw weights are
      kept in each checkpoint under ``current_model_state``).
    """
    validate_data_files(data.cfg)
    validate_data_model_compatibility(data.cfg, model.cfg)

    checkpoint_callback = build_checkpoint()
    callback_list = [
        LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=100,
            mode="min",
            check_on_train_epoch_end=False,
            verbose=True,
            check_finite=True,
            min_delta=0.0,
        ),
        checkpoint_callback,
    ]
    if log_mask_rate_bins:
        model.log_mask_rate_bins = True
    output_dir = HydraConfig.get().runtime.output_dir
    if checkpoint_every_n_epochs > 0:
        callback_list.append(
            ModelCheckpoint(
                dirpath=f"{output_dir}/periodic",
                filename="{epoch}",
                save_top_k=-1,
                every_n_epochs=checkpoint_every_n_epochs,
                save_on_train_epoch_end=False,
            )
        )
    if sample_eval_every_n_epochs > 0:
        # Must come before the EMA callback, so it runs before the EMA weights
        # are swapped back out at the end of validation.
        callback_list.append(
            SampleEvalMonitor(
                sample_eval_every_n_epochs,
                sample_eval_regions,
                sample_eval_reads,
                seed=seed,
            )
        )
        callback_list.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="best-sample-{epoch}",
                monitor="sample/profile_pearson",
                mode="max",
                save_top_k=1,
                every_n_epochs=sample_eval_every_n_epochs,
                save_on_train_epoch_end=False,
            )
        )
    if ema_decay > 0:
        callback_list.append(EMAWeightAveraging(decay=ema_decay))
    # A local copy of every logged metric, readable without wandb.
    csv_logger = CSVLogger(save_dir=HydraConfig.get().runtime.output_dir, name="csv", version="")

    trainer = trainer(
        callbacks=callback_list,
        logger=[wandb_logger, csv_logger],
        enable_progress_bar=True,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
    )

    model.model.compile()
    trainer.fit(model, data)

    best = checkpoint_callback.best_model_path
    print("Best ckpt:", best)


TrainConfig = builds_full(
    main,
    seed=0,
    assay_type="cg_gc_accessibility",
    model=builds_full(
        SMFNet,
        cfg=builds_full(
            SMFNetConfig,
            assay_type="${assay_type}",
            cnn_config=builds_full(
                CNNConfig,
            ),
        ),
    ),
    data=builds_full(
        SparseSMFDataModule,
        cfg=builds_full(
            SparseDataModuleConfig,
            assay_type="${assay_type}",
        ),
    ),
    trainer=builds_partial(
        Trainer,
        accelerator="gpu",
        max_epochs=400,
        precision="bf16-mixed",
    ),
    wandb_logger=builds_full(
        WandbLogger,
        project="SMFNet",
        log_model=False,
    ),
)

store = ZenStore()
store(TrainConfig, name="base")
store_help(
    store,
    "train",
    """Train an SMFNet model on H5 files of observed reads.

Example:
  pixi run python scripts/train.py \\
      "data.cfg.h5_path=[ONT_mESC_acRegions.h5,ONT_mESC_bgRegions.h5]" \\
      data.cfg.fasta_path=mm10.fa assay_type=cg_gc_accessibility""",
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main, pre_call=zen(seed_before_instantiation)).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )
