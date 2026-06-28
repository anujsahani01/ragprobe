"""
Live Evaluator
==============
Decorator-based evaluation that scores every query as it flows through your pipeline.
Instead of running batch tests after-the-fact, this scores in real-time.

Usage:
    from rag_eval import live_eval

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
    # result still returns your normal output
    # But scores are logged to rag_eval's score store
"""

from __future__ import annotations

import time
import functools
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
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


# In-memory score store (replace with DB/file in production)
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
        metrics.setdefault(score.metric_name, []).append(score.score)

    return {
        "total_queries": len(set(s.query for s in _score_store)),
        "total_scores": len(_score_store),
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


METRIC_MAP = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
    "context_relevancy": ContextualRelevancyMetric,
}


def live_eval(
    metrics: list[str] | None = None,
    threshold: float = 0.7,
    async_scoring: bool = False,
):
    """
    Decorator that scores your RAG function on every call.

    Your function MUST return a dict with:
    - "answer": str (the generated answer)
    - "retrieval_context": list[str] (the retrieved chunks)

    The first argument to your function is treated as the query.

    Args:
        metrics: Which metrics to score. Default = ["faithfulness", "answer_relevancy"]
        threshold: Score threshold (for logging purposes).
        async_scoring: If True, scoring happens in background (non-blocking). TODO: implement.
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
                # If result is just a string, treat it as the answer
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
                )

            return result

        return wrapper
    return decorator


def _score_pipeline(
    query: str,
    answer: str,
    context: list[str],
    metric_names: list[str],
    threshold: float,
    latency_ms: float,
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

            _score_store.append(LiveScore(
                timestamp=timestamp,
                query=query,
                metric_name=metric_name,
                score=metric.score,
                reason=metric.reason or "",
                latency_ms=latency_ms,
            ))
        except Exception:
            # Don't break the pipeline if scoring fails
            _score_store.append(LiveScore(
                timestamp=timestamp,
                query=query,
                metric_name=metric_name,
                score=-1.0,
                reason="Scoring failed",
                latency_ms=latency_ms,
            ))
