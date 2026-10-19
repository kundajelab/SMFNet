"""Correlation coefficients with the guards evaluation metrics need.

Both helpers return NaN rather than raising for the degenerate inputs a region
can produce: fewer than two points, or a constant vector. Metrics tables carry
one row per region and NaN is how they record "this region could not answer".
"""

import torch
from scipy.stats import spearmanr


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if denominator == 0:
        return float("nan")
    return float((x * y).sum() / denominator)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or x.unique().numel() < 2 or y.unique().numel() < 2:
        return float("nan")
    return float(spearmanr(x.numpy(), y.numpy()).statistic)
