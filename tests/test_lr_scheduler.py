import math

import pytest
import torch

from smf_net.lr_scheduler import LinearWarmupCosineAnnealingLR


def test_lr_warms_up_linearly_then_follows_a_cosine_to_eta_min():
    base_lr, warmup, total = 1e-3, 10, 100
    start = eta_min = base_lr / 100
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup, total, warmup_start_lr=start, eta_min=eta_min
    )

    lrs = []
    for _ in range(total + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    def expected(step: int) -> float:
        if step < warmup:
            return start + step * (base_lr - start) / (warmup - 1)
        progress = (step - warmup) / (total - warmup)
        return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * progress))

    assert lrs == pytest.approx([expected(step) for step in range(total + 1)])
    assert lrs[0] == pytest.approx(start)
    assert max(lrs) == pytest.approx(base_lr)
    assert lrs[total] == pytest.approx(eta_min)
