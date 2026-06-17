"""
NER System — Production-grade Named Entity Recognition with spaCy.
"""

from .pipeline import NERPipeline
from .trainer import NERTrainer
from .evaluator import NEREvaluator
from .preprocessor import TextPreprocessor
from .postprocessor import EntityPostprocessor
from .entity_store import EntityStore

__all__ = [
    "NERPipeline",
    "NERTrainer",
    "NEREvaluator",
    "TextPreprocessor",
    "EntityPostprocessor",
    "EntityStore",
]

__version__ = "1.0.0"
