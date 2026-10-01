"""AENCA V6: Oscillator-Controlled Adaptive NCA Language Model.

This package implements the V6 architecture with per-cell coupled Kuramoto
oscillators controlling all temporal dynamics. Adaptive computation via
oscillator-driven readout accumulation (per-position halting).

Importing this module triggers the @register_model("v6") decorator,
making the model discoverable by get_model_class("v6").
"""

from ncalm.models.v6.config import V6Config
from ncalm.models.v6.model import AENCAv6Model

__all__ = ["AENCAv6Model", "V6Config"]
