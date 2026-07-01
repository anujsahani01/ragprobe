"""
Agent / MCP Tool Evaluator
===========================
Evaluates AI agent tool selection, planning, and execution efficiency.

This goes beyond RAG — it answers:
- Did the agent pick the RIGHT tool for the query?
- Did it call tools in the correct ORDER?
- Did it use unnecessary tools (over-tooling)?
- Were tool arguments correct?
- Did it complete the user's task?

Integrates with DeepEval's agentic metrics:
- ToolCorrectnessMetric
- TaskCompletionMetric (via GEval)

Usage:
    from ragprobe.evaluator.agent_eval import AgentEvaluator, ToolCall, AgentTestCase

    evaluator = AgentEvaluator(available_tools=["retrieval", "query_rewriter", "clarify_query"])

    result = evaluator.evaluate_tool_selection(
        query="How does auth work?",
        tools_called=[ToolCall(name="retrieval")],
        expected_tools=[ToolCall(name="retrieval")],
    )

    # Or batch evaluate from a dataset
    results = evaluator.evaluate_batch(agent_dataset)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval.test_case import SingleTurnParams


@dataclass
class AgentTestCase:
    """
    A test case for agent evaluation.

    Represents a user query + what the agent did (tools called)
    + what the agent should have done (expected tools).
    """
    query: str                                      # User's input
    actual_output: str                              # Agent's final response
    tools_called: list[ToolCall]                    # Tools the agent actually used
    expected_tools: list[ToolCall]                  # Tools it SHOULD have used
    category: str = ""                              # Optional: "simple_query", "extraction", etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "actual_output": self.actual_output,
            "tools_called": [{"name": t.name} for t in self.tools_called],
            "expected_tools": [{"name": t.name} for t in self.expected_tools],
            "category": self.category,
            "metadata": self.metadata,
        }


@dataclass
class AgentScore:
    """Score for a single agent test case."""
    query: str
    metric_name: str
    score: float
    reason: str = ""
    passed: bool = False
    tools_called: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    category: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentEvalResults:
    """Aggregate results from agent evaluation."""
    scores: list[AgentScore] = field(default_factory=list)
    total_time_seconds: float = 0.0
    run_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s.passed) / len(self.scores)

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def scores_by_metric(self) -> dict[str, float]:
        """Average score per metric."""
        metric_totals: dict[str, list[float]] = {}
        for s in self.scores:
            metric_totals.setdefault(s.metric_name, []).append(s.score)
        return {k: sum(v) / len(v) for k, v in metric_totals.items()}

    @property
    def scores_by_category(self) -> dict[str, float]:
        """Average score per test category."""
        cat_totals: dict[str, list[float]] = {}
        for s in self.scores:
            if s.category:
                cat_totals.setdefault(s.category, []).append(s.score)
        return {k: sum(v) / len(v) for k, v in cat_totals.items()}

    @property
    def failed_cases(self) -> list[AgentScore]:
        return [s for s in self.scores if not s.passed]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "AGENT / MCP TOOL EVALUATION RESULTS",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Total test cases: {len(self.scores)}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"Avg score: {self.avg_score:.3f}",
            f"Time: {self.total_time_seconds:.1f}s",
            "",
            "Per-metric averages:",
        ]
        for metric, score in sorted(self.scores_by_metric.items()):
            status = "✓" if score >= 0.7 else "✗"
            lines.append(f"  {status} {metric}: {score:.3f}")

        if self.scores_by_category:
            lines.append("")
            lines.append("Per-category averages:")
            for cat, score in sorted(self.scores_by_category.items()):
                status = "✓" if score >= 0.7 else "✗"
                lines.append(f"  {status} {cat}: {score:.3f}")

        if self.failed_cases:
            lines.append("")
            lines.append(f"─── FAILED ({len(self.failed_cases)} cases) ───")
            for s in self.failed_cases:
                lines.append(f"  ✗ Q: {s.query[:60]}...")
                lines.append(f"    Called: {s.tools_called}")
                lines.append(f"    Expected: {s.expected_tools}")
                if s.reason:
                    lines.append(f"    Reason: {s.reason[:100]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def detailed_report(self) -> str:
        """Per-case drill-down."""
        lines = [
            "=" * 60,
            "AGENT EVALUATION DETAILED REPORT",
            "=" * 60,
            "",
        ]

        for i, s in enumerate(self.scores, 1):
            status = "✓" if s.passed else "✗"
            lines.append(f"[{i}] {status} {s.query[:70]}")
            lines.append(f"     Metric: {s.metric_name} | Score: {s.score:.3f}")
            lines.append(f"     Called:   {s.tools_called}")
            lines.append(f"     Expected: {s.expected_tools}")
            if s.reason:
                lines.append(f"     Reason: {s.reason[:150]}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_time_seconds": self.total_time_seconds,
            "summary": {
                "total_cases": len(self.scores),
                "pass_rate": self.pass_rate,
                "avg_score": self.avg_score,
                "scores_by_metric": self.scores_by_metric,
                "scores_by_category": self.scores_by_category,
            },
            "scores": [s.to_dict() for s in self.scores],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class AgentEvaluator:
    """
    Evaluates agent/MCP tool selection and task completion.

    Args:
        available_tools: List of all tool names the agent can use.
                        Provides context for evaluating selection optimality.
        threshold: Minimum passing score.
    """

    def __init__(
        self,
        available_tools: list[str] | None = None,
        threshold: float = 0.8,
    ):
        self.available_tools = available_tools or []
        self.threshold = threshold

    def evaluate_tool_selection(
        self,
        query: str,
        actual_output: str,
        tools_called: list[ToolCall],
        expected_tools: list[ToolCall],
        category: str = "",
    ) -> list[AgentScore]:
        """
        Evaluate a single agent interaction's tool selection.

        Returns scores for:
        - tool_correctness: Did it use the right tools?
        - task_completion: Did it accomplish what the user asked? (GEval)
        - efficiency: Did it use unnecessary tools? (deterministic)
        """
        scores: list[AgentScore] = []

        # 1. Tool Correctness (DeepEval metric)
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )

        try:
            available = [ToolCall(name=t) for t in self.available_tools] if self.available_tools else None
            metric = ToolCorrectnessMetric(
                threshold=self.threshold,
                available_tools=available,
                include_reason=True,
            )
            metric.measure(test_case)

            scores.append(AgentScore(
                query=query,
                metric_name="tool_correctness",
                score=metric.score if metric.score is not None else 0.0,
                reason=metric.reason or "",
                passed=(metric.score or 0) >= self.threshold,
                tools_called=[t.name for t in tools_called],
                expected_tools=[t.name for t in expected_tools],
                category=category,
            ))
        except Exception as e:
            scores.append(AgentScore(
                query=query,
                metric_name="tool_correctness",
                score=0.0,
                reason=f"Error: {str(e)}",
                passed=False,
                tools_called=[t.name for t in tools_called],
                expected_tools=[t.name for t in expected_tools],
                category=category,
            ))

        # 2. Task Completion (GEval — did the agent accomplish the user's goal?)
        task_completion = GEval(
            name="task_completion",
            criteria=(
                "Did the agent's response fully accomplish what the user asked for? "
                "Consider whether the correct tools were used and whether the final "
                "output addresses the user's intent completely."
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            threshold=self.threshold,
        )

        try:
            task_case = LLMTestCase(input=query, actual_output=actual_output)
            task_completion.measure(task_case)

            scores.append(AgentScore(
                query=query,
                metric_name="task_completion",
                score=task_completion.score if task_completion.score is not None else 0.0,
                reason=task_completion.reason or "",
                passed=(task_completion.score or 0) >= self.threshold,
                tools_called=[t.name for t in tools_called],
                expected_tools=[t.name for t in expected_tools],
                category=category,
            ))
        except Exception as e:
            scores.append(AgentScore(
                query=query,
                metric_name="task_completion",
                score=0.0,
                reason=f"Error: {str(e)}",
                passed=False,
                tools_called=[t.name for t in tools_called],
                expected_tools=[t.name for t in expected_tools],
                category=category,
            ))

        # 3. Efficiency (deterministic — penalize unnecessary tool calls)
        called_names = set(t.name for t in tools_called)
        expected_names = set(t.name for t in expected_tools)
        unnecessary = called_names - expected_names
        missing = expected_names - called_names

        if not called_names and not expected_names:
            efficiency_score = 1.0
            efficiency_reason = "No tools expected or called."
        elif not expected_names:
            efficiency_score = 0.0 if called_names else 1.0
            efficiency_reason = f"Called {len(called_names)} tools when none expected."
        else:
            correct = called_names & expected_names
            efficiency_score = len(correct) / max(len(called_names), len(expected_names))
            parts = []
            if unnecessary:
                parts.append(f"Unnecessary tools: {list(unnecessary)}")
            if missing:
                parts.append(f"Missing tools: {list(missing)}")
            if not parts:
                parts.append("All tools used correctly, no waste.")
            efficiency_reason = " | ".join(parts)

        scores.append(AgentScore(
            query=query,
            metric_name="tool_efficiency",
            score=efficiency_score,
            reason=efficiency_reason,
            passed=efficiency_score >= self.threshold,
            tools_called=[t.name for t in tools_called],
            expected_tools=[t.name for t in expected_tools],
            category=category,
        ))

        return scores

    def evaluate_batch(self, test_cases: list[AgentTestCase]) -> AgentEvalResults:
        """
        Evaluate a batch of agent test cases.

        Args:
            test_cases: List of AgentTestCase with query, tools_called, expected_tools.

        Returns:
            AgentEvalResults with summary, drill-down, and persistence.
        """
        results = AgentEvalResults()
        start = time.time()

        for case in test_cases:
            case_scores = self.evaluate_tool_selection(
                query=case.query,
                actual_output=case.actual_output,
                tools_called=case.tools_called,
                expected_tools=case.expected_tools,
                category=case.category,
            )
            results.scores.extend(case_scores)

        results.total_time_seconds = time.time() - start
        return results


# =============================================================================
# Dataset helpers for agent evaluation
# =============================================================================

@dataclass
class AgentDataset:
    """Collection of agent test cases."""
    test_cases: list[AgentTestCase] = field(default_factory=list)
    name: str = "agent_eval_dataset"

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "AgentDataset":
        """Load dataset from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_cases = []
        for tc in data.get("test_cases", []):
            test_cases.append(AgentTestCase(
                query=tc["query"],
                actual_output=tc["actual_output"],
                tools_called=[ToolCall(name=t["name"]) for t in tc["tools_called"]],
                expected_tools=[ToolCall(name=t["name"]) for t in tc["expected_tools"]],
                category=tc.get("category", ""),
                metadata=tc.get("metadata", {}),
            ))
        return cls(test_cases=test_cases, name=data.get("name", ""))
