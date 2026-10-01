"""NCALM model implementations.

Importing this package triggers @register_model decorators, making models
discoverable by get_model_class() from the training script.
"""

from ncalm.models.aenca_v3 import AENCAv3Model, AENCAv3Config
from ncalm.models.v4 import AENCAv4Model, AENCAv4Config, AENCAv41Model
from ncalm.models.v5 import AENCAv5Model, V5Config
from ncalm.models.v6 import AENCAv6Model, V6Config
from ncalm.models.v7 import AENCAv7Model, V7Config
from ncalm.models.v8 import AENCAv8Model, V8Config
from ncalm.models.v9 import AENCAv9Model, V9Config
from ncalm.models.v10 import AENCAv10Model, V10Config
from ncalm.models.v11 import AENCAv11Model, V11Config
from ncalm.models.v12 import AENCAv12Model, V12Config
from ncalm.models.v13 import AENCAv13Model, V13Config
from ncalm.models.v14 import AENCAv14Model, V14Config
from ncalm.models.v15 import AENCAv15Model, V15Config
from ncalm.models.gpt2 import GPT2Model, GPT2Config

__all__ = [
    "AENCAv3Model", "AENCAv3Config",
    "AENCAv4Model", "AENCAv4Config",
    "AENCAv41Model",
    "AENCAv5Model", "V5Config",
    "AENCAv6Model", "V6Config",
    "AENCAv7Model", "V7Config",
    "AENCAv8Model", "V8Config",
    "AENCAv9Model", "V9Config",
    "AENCAv10Model", "V10Config",
    "AENCAv11Model", "V11Config",
    "AENCAv12Model", "V12Config",
    "AENCAv13Model", "V13Config",
    "AENCAv14Model", "V14Config",
    "AENCAv15Model", "V15Config",
    "GPT2Model", "GPT2Config",
]
