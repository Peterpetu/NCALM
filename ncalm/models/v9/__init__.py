"""AENCA V9: Wolfram Cellular Automata Language Model.

ONE shared Hebbian kernel, 3 dilated steps, 9 GEMMs, no attention.
Target: ~45K tok/s on RTX 4080, matching GPT-2 throughput.

Reference: /memories/session/plan.md
"""

from ncalm.models.v9.config import V9Config
from ncalm.models.v9.model import AENCAv9Model

__all__ = ["V9Config", "AENCAv9Model"]
