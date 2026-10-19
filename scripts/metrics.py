from dataclasses import dataclass
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import ZenStore, make_custom_builds_fn, zen

from smf_net.cli import store_help
from smf_net.evaluation.metrics import MetricsConfig
from smf_net.evaluation.reporting import (
    train_baselines_for,
    write_covariation_decay,
    write_drop_summary,
    write_gap_closed_tsv,
    write_metrics_tsv,
    write_run_length_histograms,
    write_site_correlations_tsv,
)

builds_full = make_custom_builds_fn(populate_full_signature=True)


@dataclass(frozen=True)
class MetricsPlotConfig:
    """Optional summary figure generated from the metrics table."""

    enabled: bool = False
    bins: int = 40


def main(
    samples_path: str,
    metrics: MetricsConfig,
    plot: MetricsPlotConfig,
) -> None:
    samples_path = to_absolute_path(samples_path)
    output_path = Path(HydraConfig.get().runtime.output_dir) / "metrics.tsv"

    baselines, flank_baseline = train_baselines_for(samples_path, metrics)
    if baselines is not None:
        baselines_path = output_path.with_name("positional_baselines.npz")
        baselines.save(baselines_path)
        print(
            f"Fit positional baselines on {baselines.n_profile_regions} train "
            f"regions ({baselines.n_pair_regions} for pairs); wrote {baselines_path}"
        )
    if flank_baseline is not None:
        flank_path = output_path.with_name("flank_baselines.npz")
        flank_baseline.save(flank_path)
        print(
            f"Fit flank baseline on {flank_baseline.n_regions} train regions "
            f"({dict(zip(flank_baseline.site_type_names, flank_baseline.n_sites.tolist()))} "
            f"sites); wrote {flank_path}"
        )

    table = write_metrics_tsv(samples_path, str(output_path), metrics, baselines, flank_baseline)
    print(f"Wrote metrics for {len(table)} regions to {output_path}")

    site_path = output_path.with_name("site_correlations.tsv")
    site_summary = write_site_correlations_tsv(table, site_path, metrics)
    print(f"Wrote pooled site correlations to {site_path}")
    print(site_summary.to_string(index=False))

    summary_path = output_path.with_name("metrics_summary.tsv")
    summary = write_gap_closed_tsv(table, summary_path)
    if len(summary):
        print(f"Wrote pooled gap-closed summary to {summary_path}")
        print(summary.to_string(index=False))

    drops = write_drop_summary(table, output_path.parent, metrics)
    print(f"Wrote drop reasons to {output_path.with_name('metrics_drops.tsv')}")
    region_rows = drops[drops["unit"] == "regions"]
    by_metric = region_rows.pivot(index="metric", columns="reason", values="count")
    print(
        by_metric.reindex(
            index=region_rows["metric"].unique(),
            columns=region_rows["reason"].unique(),
        ).to_string()
    )
    totals = drops.loc[
        drops["unit"] != "regions",
        ["metric", "unit", "reason", "count", "fraction"],
    ]
    print(totals.to_string(index=False))

    histogram_path = output_path.with_name("run_length_histograms.npz")
    if write_run_length_histograms(table, histogram_path, metrics):
        print(f"Wrote run-length histograms to {histogram_path}")

    decay_path = output_path.with_name("covariation_decay.npz")
    if write_covariation_decay(table, decay_path, metrics):
        print(f"Wrote covariation decay curves to {decay_path}")

    if plot.enabled:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        from smf_net.evaluation.metrics_plotting import (
            plot_covariation,
            plot_metrics,
            plot_run_lengths,
            plot_site_correlations,
        )
        from smf_net.evaluation.plot_style import REPORT_FONT_SIZE, plot_style

        figures = [
            plot_metrics(
                table,
                min_correlation_sites=metrics.min_correlation_sites,
                bins=plot.bins,
            ),
            plot_site_correlations(site_summary),
        ]
        if metrics.include_run_lengths:
            figures.append(plot_run_lengths(table, metrics.run_length_bins))
        if metrics.include_covariation:
            figures.append(plot_covariation(table, metrics.separation_bins))

        plot_path = output_path.with_suffix(".pdf")
        # Some settings, such as how fonts are embedded, apply when saving.
        with plot_style(REPORT_FONT_SIZE), PdfPages(plot_path) as pdf:
            for figure in figures:
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
        print(f"Wrote metric plots to {plot_path}")


MetricsAppConfig = builds_full(
    main,
    metrics=builds_full(MetricsConfig),
    plot=builds_full(MetricsPlotConfig),
)

store = ZenStore()
store(MetricsAppConfig, name="base")
store_help(
    store,
    "metrics",
    """Compare the generated and observed reads of a samples.h5 file and write
metrics.tsv, summary tables and, with plot.enabled=true, a PDF of plots.

Example:
  pixi run python scripts/metrics.py samples_path=samples.h5 plot.enabled=true""",
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="base",
        config_path=None,
        version_base="1.3",
    )
