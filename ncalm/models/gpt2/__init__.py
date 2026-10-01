"""GPT-2/NeoX baseline model package.

Provides a standard decoder-only transformer baseline matching Pythia-160M
for academic comparison against AENCA v5.1.
"""

from ncalm.models.gpt2.config import GPT2Config
from ncalm.models.gpt2.model import GPT2Model

__all__ = ["GPT2Config", "GPT2Model"]
