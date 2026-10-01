"""AENCA V4/V4.1: Diffusion Neural Cellular Automata Language Model.

Importing this package triggers @register_model decorators, making V4/V4.1
discoverable by get_model_class("aenca_v4"/"aenca_v41") from the training script.
"""

from ncalm.models.v4.config import AENCAv4Config
from ncalm.models.v4.model import AENCAv4Model
from ncalm.models.v4.model_v41 import AENCAv41Model

__all__ = ["AENCAv4Model", "AENCAv4Config", "AENCAv41Model"]
