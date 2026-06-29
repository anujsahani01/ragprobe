"""
Live Evaluator
==============
Decorator-based evaluation that scores every query as it flows through your pipeline.
Instead of running batch tests after-the-fact, this scores in real-time.

Usage:
    from ragprobe.evaluator.live import live_eval, get_scores_summary, export_scores

    @live_eval(metrics=["faithfulness", "answer_relevancy"])
    def my_rag_pipeline(query: str) -> dict:
        chunks = retrieve(query)
        answer = generate(query, chunks)
        return {
            "answer": answer,
            "retrieval_context": chunks,
        }

    # Every call now logs scores automatically
    result = my_rag_pipeline("How does auth work?")

    # Check scores anytime
    print(get_scores_summary())

    # Export to file for tracking
    export_scores("live_scores/session_001.json")
"""

from __future__ import annotations

import json
import time
import functools
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
)


@dataclass
class LiveScore:
    """A score recorded during a live pipeline call."""
    timestamp: str
    query: str
    metric_name: str
    score: float
    reason: str = ""
    latency_ms: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# In-memory score store
_score_store: list[LiveScore] = []


def get_scores() -> list[LiveScore]:
    """Get all recorded live scores."""
    return _score_store.copy()


def clear_scores() -> None:
    """Clear the score store."""
    _score_store.clear()


def get_scores_summary() -> dict[str, Any]:
    """Get summary stats from recorded scores."""
    if not _score_store:
        return {"total_queries": 0, "metrics": {}}

    metrics: dict[str, list[float]] = {}
    for score in _score_store:
        if score.score >= 0:  # Skip failed scores (-1)
            metrics.setdefault(score.metric_name, []).append(score.score)

    queries = set(s.query for s in _score_store)
    passed_queries = set()
    failed_queries = set()

    for s in _score_store:
        if not s.passed:
            failed_queries.add(s.query)
        else:
            passed_queries.add(s.query)

    # A query is only "passed" if it didn't appear in failed
    fully_passed = passed_queries - failed_queries

    return {
        "total_queries": len(queries),
        "total_scores": len(_score_store),
        "pass_rate": len(fully_passed) / len(queries) if queries else 0,
        "metrics": {
            name: {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            for name, values in metrics.items()
        },
    }


def export_scores(path: str | Path) -> None:
    """Export all live scores to a JSON file for tracking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "exported_at": datetime.now().isoformat(),
        "summary": get_scores_summary(),
        "scores": [s.to_dict() for s in _score_store],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


METRIC_MAP = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "context_relevancy": ContextualRelevancyMetric,
    "context_precision": ContextualPrecisionMetric,
}


def live_eval(
    metrics: list[str] | None = None,
    threshold: float = 0.7,
    log_to_file: str | None = None,
):
    """
    Decorator that scores your RAG function on every call.

    Your function MUST return a dict with:
    - "answer": str (the generated answer)
    - "retrieval_context": list[str] (the retrieved chunks)

    The first argument to your function is treated as the query.

    Args:
        metrics: Which metrics to score. Default = ["faithfulness", "answer_relevancy"]
        threshold: Score threshold for pass/fail.
        log_to_file: Optional file path to append scores per-call.
    """
    metric_names = metrics or ["faithfulness", "answer_relevancy"]

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Extract query (first positional arg or 'query' kwarg)
            query = args[0] if args else kwargs.get("query", "")

            # Time the pipeline call
            start = time.time()
            result = fn(*args, **kwargs)
            latency = (time.time() - start) * 1000

            # Extract required fields from result
            if isinstance(result, dict):
                answer = result.get("answer", "")
                context = result.get("retrieval_context", [])
            else:
                answer = str(result)
                context = []

            # Score (only if we have enough data)
            if answer and context:
                _score_pipeline(
                    query=str(query),
                    answer=answer,
                    context=context,
                    metric_names=metric_names,
                    threshold=threshold,
                    latency_ms=latency,
                    log_to_file=log_to_file,
                )

            return result

        # Attach helper methods to the wrapped function
        wrapper.get_scores = get_scores
        wrapper.get_summary = get_scores_summary
        wrapper.clear_scores = clear_scores
        wrapper.export_scores = export_scores

        return wrapper
    return decorator


def _score_pipeline(
    query: str,
    answer: str,
    context: list[str],
    metric_names: list[str],
    threshold: float,
    latency_ms: float,
    log_to_file: str | None = None,
) -> None:
    """Score a pipeline call and store results."""
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,
    )

    timestamp = datetime.now().isoformat()

    for metric_name in metric_names:
        metric_class = METRIC_MAP.get(metric_name)
        if not metric_class:
            continue

        try:
            metric = metric_class(threshold=threshold)
            metric.measure(test_case)

            score_entry = LiveScore(
                timestamp=timestamp,
                query=query,
                metric_name=metric_name,
                score=metric.score if metric.score is not None else 0.0,
                reason=metric.reason or "",
                latency_ms=latency_ms,
                passed=(metric.score or 0) >= threshold,
            )
            _score_store.append(score_entry)

            # Optionally append to file
            if log_to_file:
                _append_to_file(log_to_file, score_entry)

        except Exception as e:
            _score_store.append(LiveScore(
                timestamp=timestamp,
                query=query,
                metric_name=metric_name,
                score=-1.0,
                reason=f"Scoring failed: {str(e)}",
                latency_ms=latency_ms,
                passed=False,
            ))


def _append_to_file(path: str, score: LiveScore) -> None:
    """Append a single score to a JSONL file (one JSON per line)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "a", encoding="utf-8") as f:
        f.write(json.dumps(score.to_dict()) + "\n")
