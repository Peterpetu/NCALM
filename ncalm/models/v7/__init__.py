"""AENCA V7.1: BDH-inspired Hebbian NCA Language Model.

This package implements the V7.1 architecture with two-timescale hierarchy:
- Gamma level: shared-weight NCA kernel with HebbianKernel (ReLU concept encoding)
- Theta level: unique-weight attention + HebbianFFN (cross-modal concept gating)
- Adaptive halting at theta boundaries (PonderNet-style)

Importing this module triggers the @register_model("v7") decorator,
making the model discoverable by get_model_class("v7").
"""

from ncalm.models.v7.config import V7Config
from ncalm.models.v7.model import AENCAv7Model

__all__ = ["AENCAv7Model", "V7Config"]
