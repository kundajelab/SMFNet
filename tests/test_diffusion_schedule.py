import torch

from smf_net.diffusion_schedule import CosineDiffusionSchedule


def test_p_t_computes_probabilities_in_float32_under_autocast():
    schedule = CosineDiffusionSchedule()
    x = torch.tensor([[[0, 1, -1], [1, 0, 1]]])
    t = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        probabilities = schedule.p_t(x, t).probs

    assert probabilities.dtype == torch.float32
    # At t=0 calls are unmasked and the unobserved site is masked; at t=1
    # everything is masked.
    unmasked_0, unmasked_1, masked = torch.eye(3)
    assert torch.equal(probabilities[0, 0], torch.stack([unmasked_0, unmasked_1, masked]))
    assert torch.equal(probabilities[0, 1], masked.expand(3, 3))
