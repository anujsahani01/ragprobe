"""
Report Generator
================
Generates structured evaluation reports in multiple formats.

Formats:
- Markdown (.md) — shareable, readable, works in GitHub PRs
- JSON (.json) — machine-readable, for dashboards and CI
- Terminal — colored console output

Usage:
    from ragprobe.reporting import ReportGenerator

    report = ReportGenerator(results)
    report.to_markdown("reports/eval_report.md")
    report.to_json("reports/eval_report.json")
    print(report.to_terminal())
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ragprobe.evaluator.core import EvalResults, SampleResult, ComponentScore


class ReportGenerator:
    """
    Generates evaluation reports from EvalResults.

    Args:
        results: The evaluation results to report on.
        title: Optional report title.
    """

    def __init__(self, results: EvalResults, title: str = "RAG Pipeline Evaluation Report"):
        self.results = results
        self.title = title

    def to_markdown(self, path: str | Path | None = None) -> str:
        """
        Generate a markdown evaluation report.

        Args:
            path: If provided, saves to file. Otherwise returns string.

        Returns:
            Markdown string.
        """
        r = self.results
        lines = [
            f"# {self.title}",
            "",
            f"**Run ID:** {r.run_id}  ",
            f"**Timestamp:** {r.timestamp}  ",
            f"**Samples:** {len(r.sample_results)}  ",
            f"**Total Time:** {r.total_time_seconds:.1f}s  ",
            f"**Avg Latency:** {r.avg_latency_ms:.0f}ms/query  ",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Score | Status |",
            f"|--------|-------|--------|",
        ]

        for metric, score in sorted(r.avg_scores_by_metric.items()):
            status = "✅ Pass" if score >= r.threshold else "❌ Fail"
            lines.append(f"| {metric} | {score:.3f} | {status} |")

        lines.extend([
            "",
            f"**Overall Pass Rate:** {r.pass_rate:.1%}  ",
            f"**Retrieval Hit Rate:** {r.retrieval_hit_rate:.1%}  ",
            "",
            "---",
            "",
            "## Component Breakdown",
            "",
            "| Component | Avg Score | Status |",
            "|-----------|-----------|--------|",
        ])

        for component, score in sorted(r.avg_scores_by_component.items()):
            status = "✅" if score >= r.threshold else "❌"
            lines.append(f"| {component} | {score:.3f} | {status} |")

        # Per-sample details
        lines.extend([
            "",
            "---",
            "",
            "## Per-Query Results",
            "",
        ])

        failed = r.failed_samples
        passed = [s for s in r.sample_results if s.passed_all]

        if failed:
            lines.append(f"### ❌ Failed Samples ({len(failed)}/{len(r.sample_results)})")
            lines.append("")
            for i, sample in enumerate(failed, 1):
                lines.extend(self._format_sample_md(i, sample))

        if passed:
            lines.append(f"### ✅ Passed Samples ({len(passed)}/{len(r.sample_results)})")
            lines.append("")
            for i, sample in enumerate(passed, 1):
                lines.extend(self._format_sample_md(i, sample))

        # Recommendations
        lines.extend([
            "",
            "---",
            "",
            "## Recommendations",
            "",
        ])
        lines.extend(self._generate_recommendations())

        content = "\n".join(lines)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def to_json(self, path: str | Path | None = None) -> dict:
        """
        Generate a JSON report.

        Args:
            path: If provided, saves to file. Otherwise returns dict.
        """
        r = self.results
        report = {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "run_id": r.run_id,
            "timestamp": r.timestamp,
            "config": {
                "threshold": r.threshold,
            },
            "summary": {
                "total_samples": len(r.sample_results),
                "pass_rate": r.pass_rate,
                "retrieval_hit_rate": r.retrieval_hit_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "total_time_seconds": r.total_time_seconds,
            },
            "scores_by_metric": r.avg_scores_by_metric,
            "scores_by_component": r.avg_scores_by_component,
            "samples": [s.to_dict() for s in r.sample_results],
            "recommendations": self._generate_recommendations(),
            "verdict": "PASS" if r.passed() else "FAIL",
        }

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def to_terminal(self) -> str:
        """Generate terminal-friendly output (same as results.summary + detailed)."""
        return self.results.summary() + "\n\n" + self.results.detailed_report()

    def _format_sample_md(self, index: int, sample: SampleResult) -> list[str]:
        """Format a single sample as markdown."""
        lines = [
            f"<details>",
            f"<summary><b>{index}. {sample.question[:80]}</b> (avg: {sample.avg_score:.3f})</summary>",
            f"",
            f"- **Expected:** {sample.expected_answer[:150]}",
            f"- **Generated:** {sample.generated_answer[:150]}",
            f"- **Retrieval Hit:** {'✓' if sample.retrieval_hit else '✗'}",
            f"- **Latency:** {sample.total_latency_ms:.0f}ms",
            f"",
            f"| Metric | Score | Status | Reason |",
            f"|--------|-------|--------|--------|",
        ]

        for score in sample.component_scores:
            status = "✅" if score.passed else "❌"
            reason_short = score.reason[:80].replace("|", "\\|") if score.reason else ""
            lines.append(f"| {score.metric_name} | {score.score:.3f} | {status} | {reason_short} |")

        lines.extend(["", "</details>", ""])
        return lines

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on scores."""
        r = self.results
        recs = []
        metrics = r.avg_scores_by_metric

        if metrics.get("context_relevancy", 1) < 0.5:
            recs.append(
                "- **Reduce retrieval noise:** Your `context_relevancy` is very low. "
                "Try reducing `top_k`, increasing `similarity_threshold`, or adding a reranker."
            )

        if metrics.get("faithfulness", 1) < 0.8:
            recs.append(
                "- **Reduce hallucinations:** `faithfulness` is below 0.8. "
                "Tighten your system prompt to instruct the LLM to only use provided context."
            )

        if metrics.get("context_recall", 1) < 0.7:
            recs.append(
                "- **Improve retrieval coverage:** `context_recall` is low. "
                "Try increasing `top_k`, using a different embedding model, or adjusting chunk size."
            )

        if metrics.get("answer_relevancy", 1) < 0.7:
            recs.append(
                "- **Improve answer quality:** `answer_relevancy` is low. "
                "Review your prompt template — the LLM may be generating off-topic responses."
            )

        if r.avg_latency_ms > 5000:
            recs.append(
                "- **Reduce latency:** Average query takes over 5 seconds. "
                "Consider caching embeddings, using a faster model, or reducing `top_k`."
            )

        if r.retrieval_hit_rate < 0.8:
            recs.append(
                "- **Fix retrieval accuracy:** Only {:.0%} of queries retrieve the correct chunk. "
                "Check your embedding model quality and chunk boundaries.".format(r.retrieval_hit_rate)
            )

        if not recs:
            recs.append("- All metrics are within acceptable ranges. Consider increasing thresholds for stricter quality.")

        return recs
