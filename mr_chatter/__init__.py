"""Mr Chatter package.

This package exposes the core components required to build, train, and
interact with the Mr Chatter decoder-only transformer.
"""

from .config import MrChatterConfig
from .model import MrChatter
from .tokenizer import CharacterTokenizer
from .trainer import MrChatterTrainer, TrainingConfig
from .data import load_corpus

__all__ = [
    "MrChatterConfig",
    "MrChatter",
    "CharacterTokenizer",
    "MrChatterTrainer",
    "TrainingConfig",
    "load_corpus",
]
