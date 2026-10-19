import json
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import ZenStore, make_custom_builds_fn, zen

from smf_net.cli import load_model_for_inference, resolve_dtype, resolve_model_source, store_help
from smf_net.data import SparseDataModuleConfig, SparseSMFDataModule
from smf_net.evaluation.regions import (
    RegionInputConfig,
    batch_regions,
    regions_from_bed,
    regions_from_fasta,
)
from smf_net.evaluation.sample_io import write_samples_h5
from smf_net.evaluation.sampling import SamplingConfig, sample_regions, sample_split
from smf_net.model import validate_data_model_compatibility

builds_full = make_custom_builds_fn(populate_full_signature=True)


def main(
    checkpoint: str,
    data: SparseSMFDataModule,
    sampling: SamplingConfig,
    regions: RegionInputConfig,
    device: str,
    dtype: str,
) -> None:
    # A local checkpoint or save_pretrained directory, or a Hub repository ID.
    checkpoint = resolve_model_source(checkpoint)
    compute_dtype = resolve_dtype(dtype)
    model = load_model_for_inference(checkpoint, device)

    output_path = Path(HydraConfig.get().runtime.output_dir) / "samples.h5"

    sample_steps = sampling.sample_steps or model.cfg.sample_steps
    metadata: dict[str, str | int | bool] = {
        "checkpoint": checkpoint,
        "device": device,
        "dtype": dtype,
        "parameter_dtype": "float32",
        "source": regions.source,
        "reads_per_region": sampling.reads_per_region,
        "reads_per_chunk": sampling.reads_per_chunk,
        "regions_per_batch": sampling.regions_per_batch,
        "sample_steps": sample_steps,
        "model_input_length": model.cfg.input_length,
        "assay_type": model.cfg.assay_type,
        "progress": sampling.progress,
        "verbose": sampling.verbose,
    }
    if sampling.seed is not None:
        metadata["seed"] = sampling.seed
    if sampling.sampling_window is not None:
        metadata["sampling_window_start"], metadata["sampling_window_end"] = (
            sampling.sampling_window
        )
    if regions.source == "split":
        validate_data_model_compatibility(data.cfg, model.cfg)
        metadata["split"] = regions.split
        # Metrics refit train-set baselines from exactly this split, so record
        # everything that decides which regions and sites it holds.
        metadata["data_config"] = json.dumps(
            {
                "fasta_path": to_absolute_path(data.cfg.fasta_path),
                "h5_paths": [to_absolute_path(path) for path in data.cfg.h5_paths],
                "assay_type": data.cfg.assay_type,
                "h5_region_length": data.cfg.h5_region_length,
                "train_chrm": list(data.cfg.train_chrm),
                "val_chrm": list(data.cfg.val_chrm),
                "test_chrm": list(data.cfg.test_chrm),
            }
        )
        batches = sample_split(
            model,
            data,
            regions.split,
            sampling,
            compute_dtype=compute_dtype,
        )
    else:
        if regions.source == "fasta":
            if regions.fasta_path is None:
                raise ValueError("regions.fasta_path is required for FASTA input")
            fasta_path = to_absolute_path(regions.fasta_path)
            metadata["fasta_path"] = fasta_path
            sequence_regions = regions_from_fasta(
                fasta_path,
                sequence_length=model.cfg.input_length,
            )
        else:
            if regions.bed_path is None or regions.reference_fasta is None:
                raise ValueError(
                    "regions.bed_path and regions.reference_fasta are required for BED input"
                )
            bed_path = to_absolute_path(regions.bed_path)
            reference_fasta = to_absolute_path(regions.reference_fasta)
            metadata["bed_path"] = bed_path
            metadata["reference_fasta"] = reference_fasta
            sequence_regions = regions_from_bed(
                bed_path,
                reference_fasta,
                sequence_length=model.cfg.input_length,
            )
        batches = sample_regions(
            model,
            batch_regions(sequence_regions, sampling.regions_per_batch),
            sampling,
            total_regions=sequence_regions.total_regions,
            compute_dtype=compute_dtype,
        )

    num_regions = write_samples_h5(
        str(output_path),
        batches,
        metadata=metadata,
    )
    print(f"Wrote {num_regions} regions to {output_path}")


SampleAppConfig = builds_full(
    main,
    data=builds_full(
        SparseSMFDataModule,
        cfg=builds_full(
            SparseDataModuleConfig,
            fasta_path="",
            h5_path="",
        ),
    ),
    sampling=builds_full(SamplingConfig),
    regions=builds_full(RegionInputConfig),
    device="cuda",
    dtype="bfloat16",
)

store = ZenStore()
store(SampleAppConfig, name="base")
store_help(
    store,
    "sample",
    """Generate reads with a trained SMFNet model and write them to samples.h5.
Regions come from a FASTA file (regions.source=fasta), a BED file and reference
genome (regions.source=bed) or a split of the data set (regions.source=split).

Example:
  pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-ont-mesc \\
      regions.source=fasta regions.fasta_path=sequences.fa""",
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )
