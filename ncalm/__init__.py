"""NCALM: Model-agnostic training infrastructure for language model R&D."""

__version__ = "0.1.0"

from ncalm.model import NCALMModel, register_model, get_model_class, list_models
from ncalm.trainer import Trainer
from ncalm.profiles import PROFILES, SMOKE_PROFILE, OVERNIGHT_PROFILE
from ncalm.scheduler import get_lr_scheduler
from ncalm.data.dataset import TinyStoriesData, create_data_loader
from ncalm.data.tokenizer import TokenizerInterface, BPETokenizer

# Import model implementations to trigger @register_model decorators
import ncalm.models  # noqa: F401  (registers aenca_v3 and future models)
