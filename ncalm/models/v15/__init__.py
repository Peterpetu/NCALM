"""AENCA V15: 2D Latent Reaction-Diffusion Morphogenetic Network.

This package implements the V15 architecture where information routing
emerges from dual-species Turing reaction-diffusion dynamics on a 2D
latent grid — no explicit attention, no 1D dilated convolutions.

Key innovations over V14:
  - 1D token sequence folded into 2D grid (1024 → 32×32)
  - Dual-species morphogen: activator (3×3) + inhibitor (5×5)
  - Turing instability guarantees spatial pattern formation
  - Causal 2D neighborhood (masked depthwise, no dilation)
  - Content conditioning injected at every NCA step (c-NCA)
  - Stochastic asynchronous updates (p=0.5)
  - O(√N) information propagation, O(N) total compute

Importing this module triggers the @register_model("v15") decorator,
making the model discoverable by get_model_class("v15").
"""

from ncalm.models.v15.config import V15Config
from ncalm.models.v15.model import AENCAv15Model

__all__ = ["AENCAv15Model", "V15Config"]
