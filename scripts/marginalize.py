import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import ZenStore, make_custom_builds_fn, zen

from smf_net.cli import load_model_for_inference, resolve_dtype, resolve_model_source, store_help
from smf_net.evaluation.marginalization import (
    MarginalizationConfig,
    MotifSourceConfig,
    generate_marginalization_design,
    read_builtin_motifs,
    read_motifs,
    sample_marginalization,
    select_motifs,
)
from smf_net.evaluation.marginalization_io import write_marginalizations_h5
from smf_net.evaluation.sampling import SamplingConfig

builds_full = make_custom_builds_fn(populate_full_signature=True)


@dataclass(frozen=True)
class MarginalizationPlotConfig:
    """Optional plot of every motif, from the completed marginalization HDF5 file.

    See :func:`smf_net.evaluation.marginalization_plotting.plot_marginalizations`.
    """

    enabled: bool = False
    smoothing_window: int = 31
    statistic: Literal["median", "mean"] = "mean"
    interval: tuple[float, float] | None = None
    combine_orientations: bool = True


def main(
    checkpoint: str,
    motifs: MotifSourceConfig,
    marginalization: MarginalizationConfig,
    sampling: SamplingConfig,
    plot: MarginalizationPlotConfig,
    device: str,
    dtype: str,
) -> None:
    # A local checkpoint or save_pretrained directory, or a Hub repository ID.
    checkpoint = resolve_model_source(checkpoint)
    model = load_model_for_inference(checkpoint, device)

    if motifs.path is None:
        motif_specs = read_builtin_motifs(motifs.panel)
        motif_source = f"builtin:{motifs.panel}"
    else:
        motif_path = to_absolute_path(motifs.path)
        motif_specs = read_motifs(motif_path)
        motif_source = motif_path
    motif_specs = select_motifs(motif_specs, motifs.names)

    design = generate_marginalization_design(
        motif_specs,
        sequence_length=model.cfg.input_length,
        config=marginalization,
    )
    output_path = Path(HydraConfig.get().runtime.output_dir) / "marginalizations.h5"
    sample_steps = sampling.sample_steps or model.cfg.sample_steps
    metadata: dict[str, str | int | float | bool] = {
        "checkpoint": checkpoint,
        "assay_type": model.cfg.assay_type,
        "device": device,
        "dtype": dtype,
        "motif_source": motif_source,
        "n_backgrounds": marginalization.n_backgrounds,
        "n_gc_dinucleotides": marginalization.n_gc_dinucleotides,
        "base_probs": json.dumps(marginalization.base_probs),
        "orientations": marginalization.orientations,
        "background_seed": marginalization.seed,
        "reads_per_background": sampling.reads_per_region,
        "reads_per_chunk": sampling.reads_per_chunk,
        "backgrounds_per_batch": sampling.regions_per_batch,
        "sample_steps": sample_steps,
        "sampling_progress": sampling.progress,
        "sampling_verbose": sampling.verbose,
    }
    if marginalization.insertion_start is not None:
        metadata["insertion_start"] = marginalization.insertion_start
    if sampling.seed is not None:
        metadata["sampling_seed"] = sampling.seed

    write_marginalizations_h5(
        output_path,
        design,
        sample_marginalization(
            model,
            design,
            replace(sampling, progress=False),
            compute_dtype=resolve_dtype(dtype),
        ),
        metadata=metadata,
        show_progress=sampling.progress,
    )
    print(
        f"Wrote marginalizations for {design.n_motifs} motifs and "
        f"{design.n_backgrounds} backgrounds to {output_path}"
    )

    if plot.enabled:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from smf_net.evaluation.marginalization_plotting import plot_marginalizations_h5
        from smf_net.evaluation.plot_style import REPORT_FONT_SIZE, plot_style

        plot_path = output_path.with_suffix(".pdf")
        figure = plot_marginalizations_h5(
            output_path,
            combine_orientations=plot.combine_orientations,
            smoothing_window=plot.smoothing_window,
            statistic=plot.statistic,
            interval=plot.interval,
        )
        # Some settings, such as how fonts are embedded, apply when saving.
        with plot_style(REPORT_FONT_SIZE), PdfPages(plot_path) as pdf:
            pdf.savefig(figure)
            plt.close(figure)
        print(f"Wrote marginalization plots to {plot_path}")


MarginalizationAppConfig = builds_full(
    main,
    motifs=builds_full(MotifSourceConfig),
    marginalization=builds_full(MarginalizationConfig),
    sampling=builds_full(SamplingConfig),
    plot=builds_full(MarginalizationPlotConfig),
    device="cuda",
    dtype="bfloat16",
)

store = ZenStore()
store(MarginalizationAppConfig, name="base")
store_help(
    store,
    "marginalize",
    """Insert motifs into random backgrounds, generate reads with and without them,
and write marginalizations.h5 and, with plot.enabled=true, a PDF of profiles.

Example:
  pixi run python scripts/marginalize.py checkpoint=kundajelab/smfnet-ont-mesc \\
      "motifs.names=[CTCF,NRF1]" plot.enabled=true""",
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )
