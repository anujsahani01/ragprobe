"""
RagEvaluator — The Main Evaluation Engine
==========================================
Orchestrates evaluation of any RAG system by:
1. Running queries from a dataset through the adapter
2. Scoring each component (retrieval, reranking, generation) individually
3. Producing per-component and aggregate scores
4. Per-query drill-down — see exactly which queries failed and why
5. Score persistence — save results, compare runs, detect regressions

Usage:
    from ragprobe import RagEvaluator, EvalAdapter

    evaluator = RagEvaluator(adapter=my_adapter)
    results = evaluator.evaluate(dataset)

    # Summary view
    print(results.summary())

    # Per-query drill-down
    print(results.detailed_report())

    # Save for regression tracking
    results.save("eval_results/run_001.json")

    # Compare with previous run
    regression = results.compare_with("eval_results/run_000.json")
    print(regression)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

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
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


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
            if self.source_chunk[:200] in chunk or chunk in self.source_chunk:
                return True
        return False

    @property
    def avg_score(self) -> float:
        """Average score across all components."""
        scores = [s.score for s in self.component_scores if s.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def failed_metrics(self) -> list[ComponentScore]:
        """Metrics that failed (below threshold)."""
        return [s for s in self.component_scores if not s.passed]

    @property
    def passed_all(self) -> bool:
        """Did this sample pass all metrics?"""
        return all(s.passed for s in self.component_scores)

    def drill_down(self) -> str:
        """Detailed breakdown for this single query."""
        lines = [
            f"  Q: {self.question[:80]}{'...' if len(self.question) > 80 else ''}",
            f"  Retrieval hit: {'✓' if self.retrieval_hit else '✗'}",
            f"  Latency: {self.total_latency_ms:.0f}ms",
        ]
        for score in self.component_scores:
            status = "✓" if score.passed else "✗"
            lines.append(f"    {status} {score.metric_name}: {score.score:.3f}")
            if not score.passed and score.reason:
                # Show reason for failures (truncated)
                reason_short = score.reason[:120] + "..." if len(score.reason) > 120 else score.reason
                lines.append(f"      → {reason_short}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "retrieval_hit": self.retrieval_hit,
            "avg_score": self.avg_score,
            "passed_all": self.passed_all,
            "total_latency_ms": self.total_latency_ms,
            "component_scores": [s.to_dict() for s in self.component_scores],
            "retrieved_chunks_count": len(self.retrieved_chunks),
        }


@dataclass
class EvalResults:
    """Aggregate results from a full evaluation run."""
    sample_results: list[SampleResult] = field(default_factory=list)
    total_time_seconds: float = 0.0
    run_id: str = ""
    timestamp: str = ""
    threshold: float = 0.7

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def retrieval_hit_rate(self) -> float:
        """% of queries where the correct chunk was retrieved."""
        if not self.sample_results:
            return 0.0
        hits = sum(1 for r in self.sample_results if r.retrieval_hit)
        return hits / len(self.sample_results)

    @property
    def pass_rate(self) -> float:
        """% of samples that passed ALL metrics."""
        if not self.sample_results:
            return 0.0
        passed = sum(1 for r in self.sample_results if r.passed_all)
        return passed / len(self.sample_results)

    @property
    def avg_scores_by_metric(self) -> dict[str, float]:
        """Average score per metric across all samples."""
        metric_totals: dict[str, list[float]] = {}
        for result in self.sample_results:
            for score in result.component_scores:
                if score.score is not None:
                    metric_totals.setdefault(score.metric_name, []).append(score.score)

        return {
            name: sum(scores) / len(scores)
            for name, scores in metric_totals.items()
        }

    @property
    def avg_scores_by_component(self) -> dict[str, float]:
        """Average score per component (retrieval vs generation)."""
        component_totals: dict[str, list[float]] = {}
        for result in self.sample_results:
            for score in result.component_scores:
                if score.score is not None:
                    component_totals.setdefault(score.component, []).append(score.score)

        return {
            name: sum(scores) / len(scores)
            for name, scores in component_totals.items()
        }

    @property
    def failed_samples(self) -> list[SampleResult]:
        """Samples that failed at least one metric."""
        return [r for r in self.sample_results if not r.passed_all]

    @property
    def avg_latency_ms(self) -> float:
        """Average total latency per query."""
        if not self.sample_results:
            return 0.0
        return sum(r.total_latency_ms for r in self.sample_results) / len(self.sample_results)

    # =========================================================================
    # Output Formats
    # =========================================================================

    def summary(self) -> str:
        """Compact human-readable summary."""
        lines = [
            "=" * 60,
            "RAG EVALUATION RESULTS",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Samples: {len(self.sample_results)} | "
            f"Pass rate: {self.pass_rate:.1%} | "
            f"Time: {self.total_time_seconds:.1f}s",
            f"Retrieval hit rate: {self.retrieval_hit_rate:.1%}",
            f"Avg latency: {self.avg_latency_ms:.0f}ms/query",
            "",
            "Per-metric averages:",
        ]
        for metric, score in sorted(self.avg_scores_by_metric.items()):
            status = "✓" if score >= self.threshold else "✗"
            lines.append(f"  {status} {metric}: {score:.3f}")

        lines.append("")
        lines.append("Per-component averages:")
        for component, score in sorted(self.avg_scores_by_component.items()):
            status = "✓" if score >= self.threshold else "✗"
            lines.append(f"  {status} {component}: {score:.3f}")

        if self.failed_samples:
            lines.append("")
            lines.append(f"Failed samples: {len(self.failed_samples)}/{len(self.sample_results)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def detailed_report(self) -> str:
        """
        Per-query drill-down — shows each query, its scores, and failure reasons.
        This is what makes scores ACTIONABLE.
        """
        lines = [
            "=" * 60,
            "DETAILED EVALUATION REPORT",
            "=" * 60,
            "",
        ]

        # Show failed samples first
        if self.failed_samples:
            lines.append(f"─── FAILED ({len(self.failed_samples)} samples) ───")
            lines.append("")
            for i, result in enumerate(self.failed_samples, 1):
                lines.append(f"[{i}] {result.drill_down()}")
                lines.append("")

        # Then passed samples
        passed = [r for r in self.sample_results if r.passed_all]
        if passed:
            lines.append(f"─── PASSED ({len(passed)} samples) ───")
            lines.append("")
            for i, result in enumerate(passed, 1):
                lines.append(f"[{i}] {result.drill_down()}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # =========================================================================
    # Persistence — Save and Load Results
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save evaluation results to JSON for regression tracking.

        Usage:
            results.save("eval_results/run_001.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "threshold": self.threshold,
            "total_time_seconds": self.total_time_seconds,
            "summary": {
                "total_samples": len(self.sample_results),
                "pass_rate": self.pass_rate,
                "retrieval_hit_rate": self.retrieval_hit_rate,
                "avg_latency_ms": self.avg_latency_ms,
                "avg_scores_by_metric": self.avg_scores_by_metric,
                "avg_scores_by_component": self.avg_scores_by_component,
            },
            "samples": [r.to_dict() for r in self.sample_results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "EvalResults":
        """Load previously saved results."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            threshold=data.get("threshold", 0.7),
            total_time_seconds=data["total_time_seconds"],
        )

        for sample_data in data.get("samples", []):
            scores = [
                ComponentScore(**s) for s in sample_data.get("component_scores", [])
            ]
            results.sample_results.append(SampleResult(
                question=sample_data["question"],
                expected_answer=sample_data["expected_answer"],
                source_chunk="",  # Not stored in saved results
                retrieved_chunks=[],
                generated_answer=sample_data["generated_answer"],
                component_scores=scores,
                total_latency_ms=sample_data.get("total_latency_ms", 0),
            ))

        return results

    # =========================================================================
    # Regression Detection
    # =========================================================================

    def compare_with(self, previous_path: str | Path) -> str:
        """
        Compare current results with a previous run. Detect regressions.

        Usage:
            report = results.compare_with("eval_results/run_000.json")
            print(report)

        Returns:
            Human-readable regression report.
        """
        previous = EvalResults.load(previous_path)
        return self._generate_comparison(previous)

    def _generate_comparison(self, previous: "EvalResults") -> str:
        """Generate a comparison report between two runs."""
        lines = [
            "=" * 60,
            "REGRESSION REPORT",
            f"Current:  {self.run_id} ({self.timestamp})",
            f"Previous: {previous.run_id} ({previous.timestamp})",
            "=" * 60,
            "",
        ]

        # Compare pass rates
        pass_diff = self.pass_rate - previous.pass_rate
        indicator = "↑" if pass_diff > 0 else "↓" if pass_diff < 0 else "→"
        lines.append(f"Pass rate: {previous.pass_rate:.1%} → {self.pass_rate:.1%} ({indicator} {abs(pass_diff):.1%})")

        # Compare retrieval hit rate
        hit_diff = self.retrieval_hit_rate - previous.retrieval_hit_rate
        indicator = "↑" if hit_diff > 0 else "↓" if hit_diff < 0 else "→"
        lines.append(f"Retrieval hit rate: {previous.retrieval_hit_rate:.1%} → {self.retrieval_hit_rate:.1%} ({indicator} {abs(hit_diff):.1%})")

        # Compare per-metric scores
        lines.append("")
        lines.append("Per-metric changes:")
        prev_metrics = previous.avg_scores_by_metric
        curr_metrics = self.avg_scores_by_metric

        all_metrics = set(list(prev_metrics.keys()) + list(curr_metrics.keys()))
        regressions = []

        for metric in sorted(all_metrics):
            prev_score = prev_metrics.get(metric, 0)
            curr_score = curr_metrics.get(metric, 0)
            diff = curr_score - prev_score

            if abs(diff) < 0.01:
                indicator = "→"
            elif diff > 0:
                indicator = "↑"
            else:
                indicator = "↓"
                regressions.append((metric, prev_score, curr_score, diff))

            lines.append(f"  {indicator} {metric}: {prev_score:.3f} → {curr_score:.3f} ({diff:+.3f})")

        # Latency comparison
        lines.append("")
        latency_diff = self.avg_latency_ms - previous.avg_latency_ms
        indicator = "↑" if latency_diff > 0 else "↓"
        lines.append(f"Avg latency: {previous.avg_latency_ms:.0f}ms → {self.avg_latency_ms:.0f}ms ({indicator} {abs(latency_diff):.0f}ms)")

        # Verdict
        lines.append("")
        if regressions:
            lines.append(f"⚠ REGRESSIONS DETECTED in {len(regressions)} metric(s):")
            for metric, prev, curr, diff in regressions:
                lines.append(f"    {metric}: dropped {abs(diff):.3f} ({prev:.3f} → {curr:.3f})")
        else:
            lines.append("✓ No regressions detected.")

        lines.append("=" * 60)
        return "\n".join(lines)

    # =========================================================================
    # CI/CD Integration
    # =========================================================================

    def passed(self) -> bool:
        """
        Returns True if ALL metrics pass on average.
        Use this for CI/CD gates:

            results = evaluator.evaluate(dataset)
            if not results.passed():
                sys.exit(1)  # Fail the CI build
        """
        for score in self.avg_scores_by_metric.values():
            if score < self.threshold:
                return False
        return True

    def exit_code(self) -> int:
        """Return 0 if passed, 1 if failed. For CI/CD scripts."""
        return 0 if self.passed() else 1


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

    def evaluate(self, dataset: EvalDataset, run_id: str | None = None) -> EvalResults:
        """
        Run the full evaluation pipeline.

        For each sample in the dataset:
        1. Call adapter.retrieve(question) → get chunks
        2. Call adapter.generate(question, chunks) → get answer
        3. Score retrieval quality (did we get the right chunks?)
        4. Score generation quality (is the answer faithful + relevant?)

        Args:
            dataset: Generated evaluation dataset.
            run_id: Optional identifier for this run (for tracking).

        Returns:
            EvalResults with per-sample scores, drill-down, persistence, and regression detection.
        """
        results = EvalResults(threshold=self.threshold)
        if run_id:
            results.run_id = run_id

        start_time = time.time()

        for i, sample in enumerate(dataset, 1):
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

                component = self._metric_to_component(metric_name)
                metric_score = metric.score if metric.score is not None else 0.0

                scores.append(ComponentScore(
                    component=component,
                    metric_name=metric_name,
                    score=metric_score,
                    reason=metric.reason or "",
                    latency_ms=retrieval_latency if component == "retrieval" else generation_latency,
                    passed=metric_score >= self.threshold,
                ))
            except Exception as e:
                scores.append(ComponentScore(
                    component="error",
                    metric_name=metric_name,
                    score=0.0,
                    reason=f"Error: {str(e)}",
                    passed=False,
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
