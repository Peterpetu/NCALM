"""AENCA V14: Emergent Dynamics Language Model.

This package implements the V14 architecture where information routing
emerges from content-enriched NCA dynamics on a morphogen field —
no explicit attention primitives (Q/K/V, dot-products, RoPE).

Importing this module triggers the @register_model("v14") decorator,
making the model discoverable by get_model_class("v14").
"""

from ncalm.models.v14.config import V14Config
from ncalm.models.v14.model import AENCAv14Model

__all__ = ["AENCAv14Model", "V14Config"]
