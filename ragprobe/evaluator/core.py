"""
RagEvaluator — The Main Evaluation Engine
==========================================
Orchestrates evaluation of any RAG system by:
1. Running queries from a dataset through the adapter
2. Scoring each component (retrieval, reranking, generation) individually
3. Producing per-component and aggregate scores

Usage:
    from ragprobe import RagEvaluator, EvalAdapter

    evaluator = RagEvaluator(adapter=my_adapter, judge_model="gpt-4o-mini")
    results = evaluator.evaluate(dataset)
    results.summary()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval import evaluate

from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult
from ragprobe.dataset.generator import EvalDataset, EvalSample


@dataclass
class ComponentScore:
    """Score for a single component on a single sample."""
    component: str          # "retrieval", "generation", "reranker", etc.
    metric_name: str        # "context_recall", "faithfulness", etc.
    score: float            # 0.0 - 1.0
    reason: str = ""
    latency_ms: float = 0.0


@dataclass
class SampleResult:
    """Full evaluation result for one query."""
    question: str
    expected_answer: str
    source_chunk: str
    retrieved_chunks: list[str]
    generated_answer: str
    component_scores: list[ComponentScore] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def retrieval_hit(self) -> bool:
        """Did the source chunk appear in retrieved results?"""
        for chunk in self.retrieved_chunks:
            # Fuzzy match — source chunk content should overlap significantly
            if self.source_chunk[:200] in chunk or chunk in self.source_chunk:
                return True
        return False

    @property
    def avg_score(self) -> float:
        """Average score across all components."""
        scores = [s.score for s in self.component_scores if s.score is not None]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class EvalResults:
    """Aggregate results from a full evaluation run."""
    sample_results: list[SampleResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    @property
    def retrieval_hit_rate(self) -> float:
        """% of queries where the correct chunk was retrieved."""
        if not self.sample_results:
            return 0.0
        hits = sum(1 for r in self.sample_results if r.retrieval_hit)
        return hits / len(self.sample_results)

    @property
    def avg_scores_by_component(self) -> dict[str, float]:
        """Average score per component across all samples."""
        component_totals: dict[str, list[float]] = {}
        for result in self.sample_results:
            for score in result.component_scores:
                if score.score is not None:
                    component_totals.setdefault(score.metric_name, []).append(score.score)

        return {
            name: sum(scores) / len(scores)
            for name, scores in component_totals.items()
        }

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            "=" * 60,
            "RAG EVALUATION RESULTS",
            "=" * 60,
            f"Total samples evaluated: {len(self.sample_results)}",
            f"Total time: {self.total_time_seconds:.1f}s",
            f"Retrieval hit rate: {self.retrieval_hit_rate:.1%}",
            "",
            "Per-metric averages:",
        ]
        for metric, score in sorted(self.avg_scores_by_component.items()):
            status = "✓" if score >= 0.7 else "✗"
            lines.append(f"  {status} {metric}: {score:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class RagEvaluator:
    """
    Core evaluator that runs your RAG pipeline against a dataset
    and scores every component.

    Args:
        adapter: Your RAG system wrapped in an EvalAdapter.
        metrics: List of metric names to evaluate. Default = all RAG metrics.
        threshold: Minimum passing score. Default = 0.7.
    """

    AVAILABLE_METRICS = {
        "answer_relevancy": AnswerRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "context_precision": ContextualPrecisionMetric,
        "context_recall": ContextualRecallMetric,
        "context_relevancy": ContextualRelevancyMetric,
    }

    def __init__(
        self,
        adapter: EvalAdapter,
        metrics: list[str] | None = None,
        threshold: float = 0.7,
    ):
        self.adapter = adapter
        self.threshold = threshold
        self.metric_names = metrics or list(self.AVAILABLE_METRICS.keys())

    def evaluate(self, dataset: EvalDataset) -> EvalResults:
        """
        Run the full evaluation pipeline.

        For each sample in the dataset:
        1. Call adapter.retrieve(question) → get chunks
        2. Call adapter.generate(question, chunks) → get answer
        3. Score retrieval quality (did we get the right chunks?)
        4. Score generation quality (is the answer faithful + relevant?)

        Returns:
            EvalResults with per-sample and aggregate scores.
        """
        results = EvalResults()
        start_time = time.time()

        for sample in dataset:
            sample_result = self._evaluate_sample(sample)
            results.sample_results.append(sample_result)

        results.total_time_seconds = time.time() - start_time
        return results

    def _evaluate_sample(self, sample: EvalSample) -> SampleResult:
        """Evaluate a single sample end-to-end."""
        start = time.time()

        # Step 1: Retrieve
        t0 = time.time()
        retrieval = self.adapter.retrieve(sample.question)
        retrieval_latency = (time.time() - t0) * 1000

        # Step 2: Generate
        t0 = time.time()
        generation = self.adapter.generate(sample.question, retrieval.chunks)
        generation_latency = (time.time() - t0) * 1000

        total_latency = (time.time() - start) * 1000

        # Step 3: Score with DeepEval
        component_scores = self._score_with_deepeval(
            question=sample.question,
            expected_answer=sample.expected_answer,
            retrieved_chunks=retrieval.chunks,
            generated_answer=generation.answer,
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
        )

        return SampleResult(
            question=sample.question,
            expected_answer=sample.expected_answer,
            source_chunk=sample.source_chunk,
            retrieved_chunks=retrieval.chunks,
            generated_answer=generation.answer,
            component_scores=component_scores,
            total_latency_ms=total_latency,
        )

    def _score_with_deepeval(
        self,
        question: str,
        expected_answer: str,
        retrieved_chunks: list[str],
        generated_answer: str,
        retrieval_latency: float,
        generation_latency: float,
    ) -> list[ComponentScore]:
        """Run DeepEval metrics and return component scores."""
        scores: list[ComponentScore] = []

        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_chunks,
        )

        for metric_name in self.metric_names:
            metric_class = self.AVAILABLE_METRICS.get(metric_name)
            if not metric_class:
                continue

            try:
                metric = metric_class(threshold=self.threshold)
                metric.measure(test_case)

                # Determine which component this metric evaluates
                component = self._metric_to_component(metric_name)

                scores.append(ComponentScore(
                    component=component,
                    metric_name=metric_name,
                    score=metric.score,
                    reason=metric.reason or "",
                    latency_ms=retrieval_latency if component == "retrieval" else generation_latency,
                ))
            except Exception as e:
                scores.append(ComponentScore(
                    component="error",
                    metric_name=metric_name,
                    score=0.0,
                    reason=f"Error: {str(e)}",
                ))

        return scores

    def _metric_to_component(self, metric_name: str) -> str:
        """Map metric name to pipeline component."""
        retrieval_metrics = {"context_precision", "context_recall", "context_relevancy"}
        generation_metrics = {"answer_relevancy", "faithfulness"}

        if metric_name in retrieval_metrics:
            return "retrieval"
        elif metric_name in generation_metrics:
            return "generation"
        return "pipeline"
