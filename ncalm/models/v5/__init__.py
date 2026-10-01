"""AENCA V5: Precision-Routed NCA Language Model.

This package implements the V5 architecture with PRMP global communication,
phase-specialized kernel bank, and autoregressive training.

Importing this module triggers the @register_model("v5") decorator,
making the model discoverable by get_model_class("v5").
"""

from ncalm.models.v5.config import V5Config
from ncalm.models.v5.model import AENCAv5Model

__all__ = ["AENCAv5Model", "V5Config"]
