"""
ragprobe: Pluggable evaluation framework for RAG systems and AI agents.

Usage:
    # === RAG Evaluation ===
    from ragprobe import RagEvaluator, EvalAdapter, DatasetGenerator

    evaluator = RagEvaluator(adapter=my_adapter)
    results = evaluator.evaluate(dataset)
    print(results.summary())
    print(results.detailed_report())
    results.save("eval_results/run_001.json")

    # === Agent/MCP Evaluation (reference-free) ===
    from ragprobe import AgentEvaluator, ToolDefinition, AgentTrace

    evaluator = AgentEvaluator(tools=[
        ToolDefinition(name="retrieval", description="Search code knowledge base"),
    ])
    # OR auto-extract from MCP server:
    # evaluator = AgentEvaluator.from_mcp_server(my_server)

    trace = AgentTrace(query="How does auth work?", tools_called=["retrieval"], output="...")
    result = evaluator.evaluate_trace(trace)

    # === Report Generation ===
    from ragprobe import ReportGenerator
    ReportGenerator(results).to_markdown("reports/eval.md")

    # === Live Evaluation ===
    from ragprobe import live_eval
    @live_eval(metrics=["faithfulness", "answer_relevancy"])
    def my_pipeline(query): ...

    # === CI/CD ===
    sys.exit(results.exit_code())
"""

from ragprobe.evaluator.core import RagEvaluator, EvalResults, SampleResult, ComponentScore
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult
from ragprobe.dataset.generator import DatasetGenerator, EvalDataset, EvalSample
from ragprobe.evaluator.live import live_eval, get_scores_summary, export_scores
from ragprobe.evaluator.agent_eval import (
    AgentEvaluator,
    AgentTrace,
    AgentTraceResult,
    AgentEvalResults,
    ToolDefinition,
)
from ragprobe.reporting import ReportGenerator
from ragprobe.tracing import trace_agent, traced_tool, TraceStore

__version__ = "0.3.0"
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
    # Agent evaluation (reference-free)
    "AgentEvaluator",
    "AgentTrace",
    "AgentTraceResult",
    "AgentEvalResults",
    "ToolDefinition",
    # Tracing (auto-capture)
    "trace_agent",
    "traced_tool",
    "TraceStore",
    # Reporting
    "ReportGenerator",
    # Live evaluation
    "live_eval",
    "get_scores_summary",
    "export_scores",
]
