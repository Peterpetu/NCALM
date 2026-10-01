"""AENCA V13: Morphogenetic Attention Language Model.

This package implements the V13 architecture where a lightweight morphogen
field evolves via local NCA dynamics and steers attention Q/K projections.

Importing this module triggers the @register_model("v13") decorator,
making the model discoverable by get_model_class("v13").
"""

from ncalm.models.v13.config import V13Config
from ncalm.models.v13.model import AENCAv13Model

__all__ = ["AENCAv13Model", "V13Config"]
