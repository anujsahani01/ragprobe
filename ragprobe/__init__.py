"""
ragprobe: Pluggable evaluation framework for RAG systems.

Usage:
    # Batch evaluation
    from ragprobe import RagEvaluator, EvalAdapter, DatasetGenerator
    evaluator = RagEvaluator(adapter=my_adapter)
    results = evaluator.evaluate(dataset)
    print(results.summary())
    print(results.detailed_report())
    results.save("eval_results/run_001.json")

    # Live evaluation (decorator)
    from ragprobe import live_eval
    @live_eval(metrics=["faithfulness", "answer_relevancy"])
    def my_pipeline(query): ...

    # Regression detection
    regression = results.compare_with("eval_results/run_000.json")

    # CI/CD gate
    sys.exit(results.exit_code())
"""

from ragprobe.evaluator.core import RagEvaluator, EvalResults, SampleResult, ComponentScore
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult
from ragprobe.dataset.generator import DatasetGenerator, EvalDataset, EvalSample
from ragprobe.evaluator.live import live_eval, get_scores_summary, export_scores

__version__ = "0.2.0"
__all__ = [
    "RagEvaluator",
    "EvalResults",
    "SampleResult",
    "ComponentScore",
    "EvalAdapter",
    "RetrievalResult",
    "GenerationResult",
    "DatasetGenerator",
    "EvalDataset",
    "EvalSample",
    "live_eval",
    "get_scores_summary",
    "export_scores",
]
