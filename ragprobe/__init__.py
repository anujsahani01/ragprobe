"""
rag-eval: Pluggable evaluation framework for RAG systems.

Usage:
    from rag_eval import RagEvaluator, EvalAdapter
    from rag_eval.dataset import DatasetGenerator
"""

from rag_eval.evaluator.core import RagEvaluator
from rag_eval.adapters.base import EvalAdapter
from rag_eval.dataset.generator import DatasetGenerator

__version__ = "0.1.0"
__all__ = ["RagEvaluator", "EvalAdapter", "DatasetGenerator"]
