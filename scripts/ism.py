import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import ZenStore, make_custom_builds_fn, zen

from smf_net.cli import load_model_for_inference, resolve_model_source, store_help
from smf_net.data import SparseDataModuleConfig, SparseSMFDataModule
from smf_net.evaluation.ism import ProfileISMConfig, profile_ism_regions
from smf_net.evaluation.ism_io import write_profile_ism_h5
from smf_net.evaluation.regions import (
    RegionInputConfig,
    regions_from_bed,
    regions_from_fasta,
    regions_from_split,
)
from smf_net.model import validate_data_model_compatibility

builds_full = make_custom_builds_fn(populate_full_signature=True)


@dataclass(frozen=True)
class ISMPlotConfig:
    """Optional first-N profile ISM plots generated from the output file."""

    enabled: bool = False
    n_regions: int = 10
    scaled: bool = False

    def __post_init__(self) -> None:
        if self.n_regions < 1:
            raise ValueError("n_regions must be positive")


def main(
    checkpoint: str,
    data: SparseSMFDataModule,
    profile_ism: ProfileISMConfig,
    regions: RegionInputConfig,
    plot: ISMPlotConfig,
    device: str,
) -> None:
    # A local checkpoint or save_pretrained directory, or a Hub repository ID.
    checkpoint = resolve_model_source(checkpoint)
    # ISM scores in full float32 precision.
    model = load_model_for_inference(checkpoint, device)

    output_path = Path(HydraConfig.get().runtime.output_dir) / "profile_ism.h5"
    output_length = model.cfg.input_length // 2
    metadata: dict[str, str | int | float | bool] = {
        "checkpoint": checkpoint,
        "device": device,
        "source": regions.source,
        "model_input_length": model.cfg.input_length,
        "length": profile_ism.length or output_length,
        "batch_size": profile_ism.batch_size,
        "importance_window": profile_ism.importance_window,
        "softmax_temperature": profile_ism.softmax_temperature,
        "verbose": profile_ism.verbose,
    }
    if profile_ism.mutation_start is not None:
        metadata["mutation_start"] = profile_ism.mutation_start

    if regions.source == "split":
        validate_data_model_compatibility(data.cfg, model.cfg)
        metadata["split"] = regions.split
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
        sequence_regions = regions_from_split(data, regions.split)
    elif regions.source == "fasta":
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

    num_regions = write_profile_ism_h5(
        output_path,
        profile_ism_regions(model, sequence_regions, profile_ism),
        metadata=metadata,
    )
    print(f"Wrote profile ISM for {num_regions} regions to {output_path}")

    if plot.enabled:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from smf_net.evaluation.ism_io import read_profile_ism_h5
        from smf_net.evaluation.ism_plotting import plot_profile_ism
        from smf_net.evaluation.plot_style import REPORT_FONT_SIZE, plot_style

        plot_path = output_path.with_suffix(".pdf")
        # Some settings, such as how fonts are embedded, apply when saving.
        with plot_style(REPORT_FONT_SIZE), PdfPages(plot_path) as pdf:
            for region in islice(
                read_profile_ism_h5(output_path),
                plot.n_regions,
            ):
                figure = plot_profile_ism(
                    region.result,
                    region_id=region.region_id,
                    scaled=plot.scaled,
                )
                pdf.savefig(figure)
                plt.close(figure)
        print(
            f"Wrote profile ISM plots for {min(plot.n_regions, num_regions)} regions to {plot_path}"
        )


ISMAppConfig = builds_full(
    main,
    data=builds_full(
        SparseSMFDataModule,
        cfg=builds_full(
            SparseDataModuleConfig,
            fasta_path="",
            h5_path="",
        ),
    ),
    profile_ism=builds_full(ProfileISMConfig),
    regions=builds_full(RegionInputConfig),
    plot=builds_full(ISMPlotConfig),
    device="cuda",
)

store = ZenStore()
store(ISMAppConfig, name="base")
store_help(
    store,
    "ism",
    """Mutate every base in a window and write the change in the predicted profile
(profile ISM) to profile_ism.h5. Takes regions the same ways as sample.py.

Example:
  pixi run python scripts/ism.py checkpoint=kundajelab/smfnet-ont-mesc \\
      regions.source=fasta regions.fasta_path=sequences.fa profile_ism.length=200""",
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )
