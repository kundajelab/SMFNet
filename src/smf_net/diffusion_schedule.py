import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# SMF calls are tokens 0..MASK_TOKEN - 1 (0 methylated/accessible, 1 unmethylated/protected)
# and the masked state of the diffusion process is the last token.
MASK_TOKEN = 2


class CosineDiffusionSchedule:
    @staticmethod
    def alpha_t(t: torch.Tensor) -> torch.Tensor:
        """The probability that a call is still unmasked at time t."""
        return 1 - torch.cos(torch.pi / 2 * (1 - t))

    def p_t(self, x: torch.Tensor, t: torch.Tensor) -> Categorical:
        """Noising distribution over tokens 0..MASK_TOKEN at time t.

        x: [B, R, L] calls, -1 where unobserved
        t: [B, R]
        Returns: Categorical with probs [B, R, L, MASK_TOKEN + 1]. Observed
        calls stay unmasked with probability alpha_t, and unobserved sites
        are always masked.
        """
        # float32 keeps the probabilities exact under bf16 autocast.
        keep = self.alpha_t(t.float())[..., None] * (x != -1)  # [B, R, L]
        one_hot = F.one_hot(x.clamp(min=0).long(), MASK_TOKEN) * keep[..., None]
        probs = torch.cat([one_hot, (1 - keep)[..., None]], dim=-1)
        return Categorical(probs=probs.clamp(min=0.0, max=1.0))

    @staticmethod
    def cross_entropy_loss_weight(t: torch.Tensor) -> torch.Tensor:
        # Table 4 of https://arxiv.org/abs/2406.04329
        return (
            (math.pi / 2 * torch.tan(math.pi / 2 * (1 - t))).clamp(min=1e-2, max=1e2).unsqueeze(-1)
        )
