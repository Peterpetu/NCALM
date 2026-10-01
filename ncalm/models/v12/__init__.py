"""AENCA V12: Morphogenetic Mixture of Hebbian Rules (MMHR).

K=8 specialized Hebbian rules soft-routed per-cell-per-step by an expanded
morphogen field (d_h=512). Integrates MNCA probabilistic rule-mixture
(arxiv 2506.20486 §3.1) into the two-field NCA paradigm.

Key advances over V11:
  - n_rules=8: K parallel Hebbian rules (MNCA mixture) → 5.5× more non-embed params
  - d_h=512: morphogen becomes PRIMARY routing substrate (MNCA §3.2 intrinsic noise)
  - Gumbel-softmax routing: rule assignment differentiable via π(h, η) network
  - Step-annealed temperature: attention GROWS as morphogen matures over 64 CA steps
  - Diversity loss: −λ·H(weights) prevents rule collapse (MNCA Appendix F insight)

Non-embed params: ~31.8M (vs V11's 5.77M). Grand total: ~137M.
"""

from ncalm.models.v12.config import V12Config
from ncalm.models.v12.model import AENCAv12Model

__all__ = ["V12Config", "AENCAv12Model"]
