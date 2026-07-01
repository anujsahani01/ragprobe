"""
ragprobe: Pluggable evaluation framework for RAG systems and AI agents.

Usage:
    # RAG Batch evaluation
    from ragprobe import RagEvaluator, EvalAdapter, DatasetGenerator
    evaluator = RagEvaluator(adapter=my_adapter)
    results = evaluator.evaluate(dataset)
    print(results.summary())
    print(results.detailed_report())
    results.save("eval_results/run_001.json")

    # Agent/MCP Tool evaluation
    from ragprobe import AgentEvaluator, AgentTestCase
    agent_eval = AgentEvaluator(available_tools=["retrieval", "query_rewriter"])
    agent_results = agent_eval.evaluate_batch(test_cases)

    # Report generation
    from ragprobe import ReportGenerator
    report = ReportGenerator(results)
    report.to_markdown("reports/eval.md")

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
from ragprobe.evaluator.agent_eval import (
    AgentEvaluator,
    AgentTestCase,
    AgentDataset,
    AgentEvalResults,
    AgentScore,
)
from ragprobe.reporting import ReportGenerator

__version__ = "0.2.0"
__all__ = [
    # RAG evaluation
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
    # Agent evaluation
    "AgentEvaluator",
    "AgentTestCase",
    "AgentDataset",
    "AgentEvalResults",
    "AgentScore",
    # Reporting
    "ReportGenerator",
    # Live evaluation
    "live_eval",
    "get_scores_summary",
    "export_scores",
]
