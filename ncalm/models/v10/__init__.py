"""AENCA V10: Persistent Growth NCA Language Model.

ONE shared tiny rule (4.9M), 64 growth steps, settlement-based permanence.
True cellular automata: simple rule, many iterations, permanent structures.
"""

from ncalm.models.v10.config import V10Config
from ncalm.models.v10.model import AENCAv10Model

__all__ = ["V10Config", "AENCAv10Model"]
