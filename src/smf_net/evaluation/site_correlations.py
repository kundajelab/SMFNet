"""Correlations of site means pooled across region occurrences.

Each eligible site occurrence has equal weight. Overlapping input regions are
not deduplicated. Arrays retain site means, not reads, for exact pooled ranks.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SITE_PREFIX = "site_means_"


def site_mean_pairs(
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    sites: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Keep a common finite site mask for all comparisons in a read group."""

    valid = sites & torch.isfinite(target) & (target != -1)
    for prediction in predictions.values():
        valid &= torch.isfinite(prediction) & (prediction != -1)
    return {
        SITE_PREFIX + name: torch.stack((prediction[valid], target[valid]), dim=1).cpu().numpy()
        for name, prediction in predictions.items()
    }


def site_correlation_summary(metrics: pd.DataFrame, min_sites: int = 10) -> pd.DataFrame:
    """Correlate concatenated site means, applying the size floor after pooling."""

    rows = []
    for column in metrics.columns:
        if not column.startswith(SITE_PREFIX):
            continue
        pairs = np.concatenate(metrics[column].tolist(), axis=0).astype(np.float64)
        n_sites = len(pairs)
        pearson = spearman = float("nan")
        if n_sites >= max(2, min_sites):
            x, y = pairs.T
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                pearson = float(np.corrcoef(x, y)[0, 1])
                spearman = float(spearmanr(x, y).statistic)
        rows.append(
            {
                "comparison": column.removeprefix(SITE_PREFIX),
                "pearson": pearson,
                "spearman": spearman,
                "n_sites": n_sites,
            }
        )
    return pd.DataFrame(rows, columns=["comparison", "pearson", "spearman", "n_sites"])
