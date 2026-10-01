"""AENCA V11: Morphogenetic Field NCA Language Model.

Two-field coupled CA: content z + morphogen h. Morphogen gates content
perception, enabling emergent information routing from local-only dynamics.
No attention. All computational structures must be GROWN.
"""

from ncalm.models.v11.config import V11Config
from ncalm.models.v11.model import AENCAv11Model

__all__ = ["V11Config", "AENCAv11Model"]
