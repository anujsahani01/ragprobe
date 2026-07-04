"""
MLflow Integration
==================
Logs ragprobe evaluation results to MLflow for visual tracking,
comparison across runs, and team collaboration.

Logs:
- Per-run summary metrics (pass_rate, avg scores, latency)
- Per-query details (query text, scores, retrieved chunks, generated answer)
- Parameters (threshold, model, top_k, etc.)
- Artifacts (full JSON results, markdown report)

Usage:
    from ragprobe.integrations.mlflow_logger import log_to_mlflow

    results = evaluator.evaluate(dataset)
    log_to_mlflow(results, experiment_name="my_rag_eval")

    # For agent evaluation
    agent_results = agent_evaluator.evaluate_batch(traces)
    log_agent_to_mlflow(agent_results, experiment_name="my_agent_eval")

    # Then run: mlflow ui
    # Opens dashboard at http://localhost:5000

Requirements:
    pip install mlflow
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _check_mlflow():
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "MLflow is not installed. Install it with: pip install mlflow\n"
            "MLflow is an optional dependency of ragprobe."
        )


# =============================================================================
# RAG Evaluation → MLflow
# =============================================================================


def log_to_mlflow(
    results: Any,  # EvalResults
    experiment_name: str = "ragprobe_rag_eval",
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log RAG evaluation results to MLflow.

    Args:
        results: EvalResults from RagEvaluator.evaluate()
        experiment_name: MLflow experiment name (creates if doesn't exist)
        run_name: Optional run name (defaults to results.run_id)
        params: Optional extra parameters to log (e.g., {"top_k": 5, "model": "gpt-4o-mini"})
        tags: Optional tags for the run

    Returns:
        MLflow run ID string.
    """
    _check_mlflow()

    mlflow.set_experiment(experiment_name)

    run_name = run_name or f"ragprobe_{results.run_id}"

    with mlflow.start_run(run_name=run_name) as run:
        # --- Parameters ---
        mlflow.log_param("threshold", results.threshold)
        mlflow.log_param("total_samples", len(results.sample_results))
        mlflow.log_param("run_id", results.run_id)
        mlflow.log_param("timestamp", results.timestamp)

        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # --- Tags ---
        mlflow.set_tag("evaluator", "ragprobe")
        mlflow.set_tag("eval_type", "rag")
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # --- Summary Metrics ---
        mlflow.log_metric("pass_rate", results.pass_rate)
        mlflow.log_metric("retrieval_hit_rate", results.retrieval_hit_rate)
        mlflow.log_metric("avg_latency_ms", results.avg_latency_ms)
        mlflow.log_metric("total_time_seconds", results.total_time_seconds)

        for metric_name, score in results.avg_scores_by_metric.items():
            mlflow.log_metric(f"avg_{metric_name}", score)

        for component, score in results.avg_scores_by_component.items():
            mlflow.log_metric(f"component_{component}", score)

        # --- Per-Query Logging (as a table artifact) ---
        query_table = _build_rag_query_table(results)
        _log_table_artifact(query_table, "per_query_results.json")

        # --- Full Results Artifact ---
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            results.save(f.name)
            mlflow.log_artifact(f.name, artifact_path="results")

        # --- Per-Query Metrics (step-based for charts) ---
        for i, sample in enumerate(results.sample_results):
            step = i + 1
            for score in sample.component_scores:
                mlflow.log_metric(f"query_{score.metric_name}", score.score, step=step)
            mlflow.log_metric("query_latency_ms", sample.total_latency_ms, step=step)

        return run.info.run_id


def _build_rag_query_table(results: Any) -> list[dict]:
    """Build a per-query table with all details for MLflow artifact logging."""
    table = []
    for i, sample in enumerate(results.sample_results, 1):
        row = {
            "index": i,
            "query": sample.question,
            "expected_answer": sample.expected_answer[:200],
            "generated_answer": sample.generated_answer[:200],
            "retrieval_hit": sample.retrieval_hit,
            "passed_all": sample.passed_all,
            "avg_score": sample.avg_score,
            "latency_ms": sample.total_latency_ms,
            "retrieved_chunks_count": len(sample.retrieved_chunks),
        }

        # Add individual metric scores
        for score in sample.component_scores:
            row[f"score_{score.metric_name}"] = score.score
            row[f"reason_{score.metric_name}"] = score.reason[:150]

        table.append(row)

    return table


# =============================================================================
# Agent Evaluation → MLflow
# =============================================================================


def log_agent_to_mlflow(
    results: Any,  # AgentEvalResults
    experiment_name: str = "ragprobe_agent_eval",
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log Agent evaluation results to MLflow.

    Args:
        results: AgentEvalResults from AgentEvaluator.evaluate_batch()
        experiment_name: MLflow experiment name
        run_name: Optional run name
        params: Optional extra parameters
        tags: Optional tags

    Returns:
        MLflow run ID string.
    """
    _check_mlflow()

    mlflow.set_experiment(experiment_name)

    run_name = run_name or f"ragprobe_agent_{results.run_id}"

    with mlflow.start_run(run_name=run_name) as run:
        # --- Parameters ---
        mlflow.log_param("threshold", results.threshold)
        mlflow.log_param("total_traces", len(results.trace_results))
        mlflow.log_param("run_id", results.run_id)

        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)

        # --- Tags ---
        mlflow.set_tag("evaluator", "ragprobe")
        mlflow.set_tag("eval_type", "agent")
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # --- Summary Metrics ---
        mlflow.log_metric("pass_rate", results.pass_rate)
        mlflow.log_metric("avg_composite_score", results.avg_composite_score)
        mlflow.log_metric("total_time_seconds", results.total_time_seconds)

        for signal, score in results.avg_scores_by_signal.items():
            mlflow.log_metric(f"avg_{signal}", score)

        # --- Per-Trace Logging ---
        trace_table = _build_agent_trace_table(results)
        _log_table_artifact(trace_table, "per_trace_results.json")

        # --- Per-Trace Metrics (step-based) ---
        for i, trace_result in enumerate(results.trace_results):
            step = i + 1
            mlflow.log_metric("trace_composite_score", trace_result.composite_score, step=step)
            for score in trace_result.scores:
                mlflow.log_metric(f"trace_{score.signal}", score.score, step=step)

        # --- Full Results Artifact ---
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            results.save(f.name)
            mlflow.log_artifact(f.name, artifact_path="results")

        return run.info.run_id


def _build_agent_trace_table(results: Any) -> list[dict]:
    """Build a per-trace table for MLflow artifact."""
    table = []
    for i, trace_result in enumerate(results.trace_results, 1):
        row = {
            "index": i,
            "query": trace_result.query,
            "tools_called": trace_result.tools_called,
            "output": trace_result.output[:200],
            "composite_score": trace_result.composite_score,
            "passed": trace_result.passed,
        }

        for score in trace_result.scores:
            row[f"score_{score.signal}"] = score.score
            row[f"reason_{score.signal}"] = score.reason[:150]

        table.append(row)

    return table


# =============================================================================
# Helpers
# =============================================================================


def _log_table_artifact(table: list[dict], filename: str) -> None:
    """Log a list of dicts as a JSON artifact in MLflow."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="tables")
