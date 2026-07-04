"""
Report Generator
================
Generates structured evaluation reports in multiple formats.
Supports BOTH RAG evaluation results and Agent/MCP evaluation results.

Formats:
- Markdown (.md) — shareable, readable, works in GitHub PRs
- JSON (.json) — machine-readable, for dashboards and CI
- Terminal — console output

Usage:
    from ragprobe.reporting import ReportGenerator

    # For RAG results
    report = ReportGenerator(rag_results)
    report.to_markdown("reports/rag_report.md")

    # For Agent results (same interface!)
    report = ReportGenerator(agent_results)
    report.to_markdown("reports/agent_report.md")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ragprobe.evaluator.core import EvalResults, SampleResult, ComponentScore
from ragprobe.evaluator.agent_eval import AgentEvalResults, AgentTraceResult


class ReportGenerator:
    """
    Generates evaluation reports from EvalResults OR AgentEvalResults.

    Automatically detects the result type and formats accordingly.

    Args:
        results: EvalResults (RAG) or AgentEvalResults (Agent/MCP).
        title: Optional report title.
    """

    def __init__(self, results: EvalResults | AgentEvalResults, title: str | None = None):
        self.results = results
        self._is_agent = isinstance(results, AgentEvalResults)

        if title:
            self.title = title
        elif self._is_agent:
            self.title = "Agent/MCP Evaluation Report"
        else:
            self.title = "RAG Pipeline Evaluation Report"

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Generate a markdown report. Saves to file if path provided."""
        content = self._agent_markdown() if self._is_agent else self._rag_markdown()

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def to_json(self, path: str | Path | None = None) -> dict:
        """Generate a JSON report. Saves to file if path provided."""
        report = self._agent_json() if self._is_agent else self._rag_json()

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def to_terminal(self) -> str:
        """Generate terminal-friendly output."""
        return self.results.summary() + "\n\n" + self.results.detailed_report()

    # =========================================================================
    # RAG Report
    # =========================================================================

    def _rag_markdown(self) -> str:
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
            "| Metric | Score | Status |",
            "|--------|-------|--------|",
        ]

        for metric, score in sorted(r.avg_scores_by_metric.items()):
            status = "Pass" if score >= r.threshold else "Fail"
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
            status = "Pass" if score >= r.threshold else "Fail"
            lines.append(f"| {component} | {score:.3f} | {status} |")

        lines.extend(["", "---", "", "## Per-Query Results", ""])

        failed = r.failed_samples
        passed = [s for s in r.sample_results if s.passed_all]

        if failed:
            lines.append(f"### Failed Samples ({len(failed)}/{len(r.sample_results)})")
            lines.append("")
            for i, sample in enumerate(failed, 1):
                lines.extend(self._format_sample_md(i, sample))

        if passed:
            lines.append(f"### Passed Samples ({len(passed)}/{len(r.sample_results)})")
            lines.append("")
            for i, sample in enumerate(passed, 1):
                lines.extend(self._format_sample_md(i, sample))

        lines.extend(["", "---", "", "## Recommendations", ""])
        lines.extend(self._rag_recommendations())

        return "\n".join(lines)

    def _rag_json(self) -> dict:
        r = self.results
        return {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "run_id": r.run_id,
            "timestamp": r.timestamp,
            "config": {"threshold": r.threshold},
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
            "recommendations": self._rag_recommendations(),
            "verdict": "PASS" if r.passed() else "FAIL",
        }

    def _format_sample_md(self, index: int, sample: SampleResult) -> list[str]:
        lines = [
            f"<details>",
            f"<summary><b>{index}. {sample.question[:80]}</b> (avg: {sample.avg_score:.3f})</summary>",
            "",
            f"- **Expected:** {sample.expected_answer[:150]}",
            f"- **Generated:** {sample.generated_answer[:150]}",
            f"- **Retrieval Hit:** {'Yes' if sample.retrieval_hit else 'No'}",
            f"- **Latency:** {sample.total_latency_ms:.0f}ms",
            "",
            "| Metric | Score | Status | Reason |",
            "|--------|-------|--------|--------|",
        ]

        for score in sample.component_scores:
            status = "Pass" if score.passed else "Fail"
            reason_short = score.reason[:80].replace("|", "\\|") if score.reason else ""
            lines.append(f"| {score.metric_name} | {score.score:.3f} | {status} | {reason_short} |")

        lines.extend(["", "</details>", ""])
        return lines

    def _rag_recommendations(self) -> list[str]:
        r = self.results
        recs = []
        metrics = r.avg_scores_by_metric

        if metrics.get("context_relevancy", 1) < 0.5:
            recs.append("- **Reduce retrieval noise:** context_relevancy is very low. Reduce top_k or add a reranker.")
        if metrics.get("faithfulness", 1) < 0.8:
            recs.append("- **Reduce hallucinations:** faithfulness is below 0.8. Tighten your system prompt.")
        if metrics.get("context_recall", 1) < 0.7:
            recs.append("- **Improve retrieval coverage:** context_recall is low. Increase top_k or adjust chunk size.")
        if metrics.get("answer_relevancy", 1) < 0.7:
            recs.append("- **Improve answer quality:** answer_relevancy is low. Review your prompt template.")
        if r.avg_latency_ms > 5000:
            recs.append("- **Reduce latency:** Average query takes over 5 seconds. Cache embeddings or reduce top_k.")
        if r.retrieval_hit_rate < 0.8:
            recs.append(f"- **Fix retrieval accuracy:** Only {r.retrieval_hit_rate:.0%} of queries retrieve the correct chunk.")
        if not recs:
            recs.append("- All metrics within acceptable ranges. Consider increasing thresholds for stricter quality.")

        return recs

    # =========================================================================
    # Agent Report
    # =========================================================================

    def _agent_markdown(self) -> str:
        r = self.results
        lines = [
            f"# {self.title}",
            "",
            f"**Run ID:** {r.run_id}  ",
            f"**Timestamp:** {r.timestamp}  ",
            f"**Traces Evaluated:** {len(r.trace_results)}  ",
            f"**Total Time:** {r.total_time_seconds:.1f}s  ",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Signal | Avg Score | Status |",
            "|--------|-----------|--------|",
        ]

        for signal, score in sorted(r.avg_scores_by_signal.items()):
            status = "Pass" if score >= r.threshold else "Fail"
            lines.append(f"| {signal} | {score:.3f} | {status} |")

        lines.extend([
            "",
            f"**Pass Rate:** {r.pass_rate:.1%}  ",
            f"**Avg Composite Score:** {r.avg_composite_score:.3f}  ",
            "",
            "---",
            "",
            "## Per-Trace Results",
            "",
        ])

        failed = r.failed_traces
        passed = [t for t in r.trace_results if t.passed]

        if failed:
            lines.append(f"### Failed Traces ({len(failed)}/{len(r.trace_results)})")
            lines.append("")
            for i, trace in enumerate(failed, 1):
                lines.extend(self._format_trace_md(i, trace))

        if passed:
            lines.append(f"### Passed Traces ({len(passed)}/{len(r.trace_results)})")
            lines.append("")
            for i, trace in enumerate(passed, 1):
                lines.extend(self._format_trace_md(i, trace))

        lines.extend(["", "---", "", "## Recommendations", ""])
        lines.extend(self._agent_recommendations())

        return "\n".join(lines)

    def _format_trace_md(self, index: int, trace: AgentTraceResult) -> list[str]:
        lines = [
            f"<details>",
            f"<summary><b>{index}. {trace.query[:80]}</b> (score: {trace.composite_score:.3f})</summary>",
            "",
            f"- **Tools Called:** {', '.join(trace.tools_called) or 'None'}",
            f"- **Output:** {trace.output[:200]}",
            "",
            "| Signal | Score | Status | Reason |",
            "|--------|-------|--------|--------|",
        ]

        for s in trace.scores:
            status = "Pass" if s.passed else "Fail"
            reason_short = s.reason[:80].replace("|", "\\|") if s.reason else ""
            lines.append(f"| {s.signal} | {s.score:.3f} | {status} | {reason_short} |")

        lines.extend(["", "</details>", ""])
        return lines

    def _agent_json(self) -> dict:
        r = self.results
        return {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "run_id": r.run_id,
            "timestamp": r.timestamp,
            "config": {"threshold": r.threshold},
            "summary": {
                "total_traces": len(r.trace_results),
                "pass_rate": r.pass_rate,
                "avg_composite_score": r.avg_composite_score,
                "total_time_seconds": r.total_time_seconds,
            },
            "scores_by_signal": r.avg_scores_by_signal,
            "traces": [tr.to_dict() for tr in r.trace_results],
            "recommendations": self._agent_recommendations(),
            "verdict": "PASS" if r.passed_all() else "FAIL",
        }

    def _agent_recommendations(self) -> list[str]:
        r = self.results
        recs = []
        signals = r.avg_scores_by_signal

        if signals.get("selection_logic", 1) < 0.7:
            recs.append("- **Improve tool selection:** Agent choosing suboptimal tools. Review tool descriptions for ambiguity.")
        if signals.get("journey_coherence", 1) < 0.7:
            recs.append("- **Fix tool ordering:** Tool call sequence is illogical. Add planning prompts or restrict flows.")
        if signals.get("goal_achievement", 1) < 0.7:
            recs.append("- **Improve task completion:** Final output doesn't address the query. Check generation prompts.")
        if not recs:
            recs.append("- All signals within acceptable ranges. Agent is performing well.")

        return recs
