"""
Component Scorer
================
Evaluates individual components in isolation.
Use this to benchmark specific parts of your pipeline:

- "Is my retriever getting the right chunks?"
- "Is my reranker actually improving order?"
- "Is my generator hallucinating?"

Usage:
    scorer = ComponentScorer(judge_model="gpt-4o-mini")

    # Score retrieval only
    retrieval_score = scorer.score_retrieval(
        query="How does auth work?",
        retrieved_chunks=["def auth_middleware()..."],
        expected_chunk="def auth_middleware()..."
    )

    # Score generation only
    gen_score = scorer.score_generation(
        query="How does auth work?",
        answer="The auth middleware validates JWT tokens...",
        context=["def auth_middleware()..."]
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import SingleTurnParams


@dataclass
class ScoreResult:
    """Result from scoring a single component."""
    metric: str
    score: float
    reason: str
    passed: bool


class ComponentScorer:
    """
    Score individual pipeline components in isolation.

    This lets you pinpoint EXACTLY which component is degrading quality.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def score_retrieval(
        self,
        query: str,
        retrieved_chunks: list[str],
        expected_answer: str,
        actual_answer: str = "",
    ) -> list[ScoreResult]:
        """
        Score retrieval quality.

        Metrics:
        - Context Precision: Are relevant chunks ranked higher?
        - Context Recall: Is all relevant info retrieved?
        - Context Relevancy: Is retrieved context relevant (no noise)?

        Args:
            query: The user's question.
            retrieved_chunks: What your retriever returned.
            expected_answer: The ground truth answer (from dataset).
            actual_answer: The generated answer (optional, can be empty).
        """
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_answer or expected_answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_chunks,
        )

        metrics = [
            ("context_precision", ContextualPrecisionMetric(threshold=self.threshold)),
            ("context_recall", ContextualRecallMetric(threshold=self.threshold)),
            ("context_relevancy", ContextualRelevancyMetric(threshold=self.threshold)),
        ]

        return self._run_metrics(test_case, metrics)

    def score_generation(
        self,
        query: str,
        answer: str,
        context: list[str],
    ) -> list[ScoreResult]:
        """
        Score generation quality.

        Metrics:
        - Answer Relevancy: Does the answer address the question?
        - Faithfulness: Is the answer grounded in context (no hallucinations)?

        Args:
            query: The user's question.
            answer: What your generator produced.
            context: The chunks that were fed to the generator.
        """
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=context,
        )

        metrics = [
            ("answer_relevancy", AnswerRelevancyMetric(threshold=self.threshold)),
            ("faithfulness", FaithfulnessMetric(threshold=self.threshold)),
        ]

        return self._run_metrics(test_case, metrics)

    def score_reranker(
        self,
        query: str,
        original_order: list[str],
        reranked_order: list[str],
        expected_answer: str,
    ) -> list[ScoreResult]:
        """
        Score reranker quality by comparing precision before/after reranking.

        Args:
            query: The user's question.
            original_order: Chunks in their original retrieval order.
            reranked_order: Chunks after reranking.
            expected_answer: The ground truth answer.
        """
        # Score precision of original order
        original_case = LLMTestCase(
            input=query,
            actual_output=expected_answer,
            expected_output=expected_answer,
            retrieval_context=original_order,
        )
        original_metric = ContextualPrecisionMetric(threshold=self.threshold)
        original_metric.measure(original_case)

        # Score precision of reranked order
        reranked_case = LLMTestCase(
            input=query,
            actual_output=expected_answer,
            expected_output=expected_answer,
            retrieval_context=reranked_order,
        )
        reranked_metric = ContextualPrecisionMetric(threshold=self.threshold)
        reranked_metric.measure(reranked_case)

        improvement = (reranked_metric.score or 0) - (original_metric.score or 0)

        return [
            ScoreResult(
                metric="reranker_precision_before",
                score=original_metric.score or 0,
                reason=original_metric.reason or "",
                passed=(original_metric.score or 0) >= self.threshold,
            ),
            ScoreResult(
                metric="reranker_precision_after",
                score=reranked_metric.score or 0,
                reason=reranked_metric.reason or "",
                passed=(reranked_metric.score or 0) >= self.threshold,
            ),
            ScoreResult(
                metric="reranker_improvement",
                score=improvement,
                reason=f"Reranker {'improved' if improvement > 0 else 'degraded'} precision by {abs(improvement):.3f}",
                passed=improvement >= 0,
            ),
        ]

    def score_chunk_quality(self, chunk: str) -> list[ScoreResult]:
        """
        Score a single chunk's quality.

        Metrics (via GEval):
        - Coherence: Is the chunk self-contained and understandable?
        - Information density: Does it contain useful, queryable information?
        """
        test_case = LLMTestCase(
            input="Evaluate this code chunk quality.",
            actual_output=chunk,
        )

        coherence = GEval(
            name="coherence",
            criteria="Is this chunk semantically coherent — a complete, understandable unit of code?",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            threshold=self.threshold,
        )

        density = GEval(
            name="info_density",
            criteria="Does this chunk contain meaningful logic, not just boilerplate or empty lines?",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
            threshold=0.5,
        )

        return self._run_metrics(test_case, [("coherence", coherence), ("info_density", density)])

    def _run_metrics(
        self,
        test_case: LLMTestCase,
        metrics: list[tuple[str, Any]],
    ) -> list[ScoreResult]:
        """Run a list of metrics against a test case."""
        results = []
        for name, metric in metrics:
            try:
                metric.measure(test_case)
                results.append(ScoreResult(
                    metric=name,
                    score=metric.score or 0,
                    reason=metric.reason or "",
                    passed=(metric.score or 0) >= self.threshold,
                ))
            except Exception as e:
                results.append(ScoreResult(
                    metric=name,
                    score=0.0,
                    reason=f"Error: {str(e)}",
                    passed=False,
                ))
        return results
