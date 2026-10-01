"""Learning rate schedule: cosine annealing with linear warmup.

Decays to 10% of peak LR by default. Call scheduler.step() once per
training step, AFTER the optimizer step.
"""

import math

from torch.optim.lr_scheduler import LambdaLR


def get_lr_scheduler(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine annealing with linear warmup.

    - Steps [0, warmup): Linear ramp from 0 to base_lr
    - Steps [warmup, max_steps]: Cosine decay from base_lr to base_lr * min_lr_ratio

    Example (overnight):
      warmup=500, max_steps=5000, base_lr=3e-4
      Step 0:    LR = 0
      Step 250:  LR = 1.5e-4 (halfway through warmup)
      Step 500:  LR = 3e-4   (peak)
      Step 2750: LR = 1.65e-4 (midpoint of cosine)
      Step 5000: LR = 3e-5   (10% of peak)
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
