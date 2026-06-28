"""
ragprobe: Pluggable evaluation framework for RAG systems.

Usage:
    from ragprobe import RagEvaluator, EvalAdapter
    from ragprobe.dataset import DatasetGenerator
"""

from ragprobe.evaluator.core import RagEvaluator
from ragprobe.adapters.base import EvalAdapter
from ragprobe.dataset.generator import DatasetGenerator

__version__ = "0.1.0"
__all__ = ["RagEvaluator", "EvalAdapter", "DatasetGenerator"]
