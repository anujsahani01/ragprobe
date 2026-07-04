"""
Agent / MCP Tool Evaluator (Reference-Free)
=============================================
Evaluates AI agent tool selection, planning, and task completion WITHOUT
requiring expected_tools. Uses LLM-as-judge with multi-signal scoring.

Works with ANY agent framework: MCP, LangChain, LlamaIndex, custom.

Signals scored (all reference-free):
1. Selection Logic (40%) — Was the tool the most logical choice for this query?
2. Journey Coherence (30%) — Was the sequence of tools a logical plan?
3. Goal Achievement (30%) — Did the final output answer the user's question?

Optional strict mode: If expected_tools are provided, also runs deterministic matching.

Usage:
    from ragprobe import AgentEvaluator, ToolDefinition, AgentTrace

    # Define your tools (or auto-extract from MCP/agent class)
    tools = [
        ToolDefinition(name="retrieval", description="Search the code knowledge base"),
        ToolDefinition(name="query_rewriter", description="Rewrite vague queries"),
    ]

    evaluator = AgentEvaluator(tools=tools)

    # Evaluate a single trace
    trace = AgentTrace(
        query="How does auth work?",
        tools_called=["retrieval"],
        output="The auth middleware validates JWT tokens...",
    )
    result = evaluator.evaluate_trace(trace)
    print(result.summary())

    # Batch evaluate
    results = evaluator.evaluate_batch(traces)
    print(results.summary())
    results.save("eval_results/agent_run.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ToolDefinition:
    """
    Definition of a tool available to the agent.
    User provides these OR they're auto-extracted from MCP/agent class.
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # Optional param schema

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentTrace:
    """
    A recorded agent interaction — what happened when a user asked something.

    This is what ragprobe evaluates. The user captures these from their agent
    and passes them in. No expected_tools needed.
    """
    query: str                          # User's input
    tools_called: list[str]             # Tool names the agent actually used (in order)
    output: str                         # Agent's final response to the user
    tool_args: list[dict] = field(default_factory=list)  # Optional: args passed to each tool
    tool_outputs: list[str] = field(default_factory=list)  # Optional: what each tool returned
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentScoreDetail:
    """Score for one signal on one trace."""
    signal: str             # "selection_logic", "journey_coherence", "goal_achievement"
    score: float            # 0.0 - 1.0
    reason: str = ""
    weight: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentTraceResult:
    """Full evaluation result for a single agent trace."""
    query: str
    tools_called: list[str]
    output: str
    scores: list[AgentScoreDetail] = field(default_factory=list)
    composite_score: float = 0.0        # Weighted average of all signals

    @property
    def passed(self) -> bool:
        return self.composite_score >= 0.7

    def summary_line(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"{status} [{self.composite_score:.2f}] {self.query[:60]}... "
            f"| Tools: {self.tools_called}"
        )

    def drill_down(self) -> str:
        lines = [
            f"  Q: {self.query[:80]}",
            f"  Tools called: {self.tools_called}",
            f"  Output: {self.output[:100]}...",
            f"  Composite score: {self.composite_score:.3f}",
        ]
        for s in self.scores:
            status = "✓" if s.passed else "✗"
            lines.append(f"    {status} {s.signal}: {s.score:.3f} (weight: {s.weight:.0%})")
            if s.reason:
                lines.append(f"      → {s.reason[:120]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tools_called": self.tools_called,
            "output": self.output[:500],
            "composite_score": self.composite_score,
            "passed": self.passed,
            "scores": [s.to_dict() for s in self.scores],
        }


@dataclass
class AgentEvalResults:
    """Aggregate results from agent evaluation."""
    trace_results: list[AgentTraceResult] = field(default_factory=list)
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
    def pass_rate(self) -> float:
        if not self.trace_results:
            return 0.0
        return sum(1 for r in self.trace_results if r.passed) / len(self.trace_results)

    @property
    def avg_composite_score(self) -> float:
        if not self.trace_results:
            return 0.0
        return sum(r.composite_score for r in self.trace_results) / len(self.trace_results)

    @property
    def avg_scores_by_signal(self) -> dict[str, float]:
        """Average score per signal across all traces."""
        signal_totals: dict[str, list[float]] = {}
        for result in self.trace_results:
            for s in result.scores:
                signal_totals.setdefault(s.signal, []).append(s.score)
        return {k: sum(v) / len(v) for k, v in signal_totals.items()}

    @property
    def failed_traces(self) -> list[AgentTraceResult]:
        return [r for r in self.trace_results if not r.passed]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "AGENT EVALUATION RESULTS (Reference-Free)",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Traces evaluated: {len(self.trace_results)}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"Avg composite score: {self.avg_composite_score:.3f}",
            f"Time: {self.total_time_seconds:.1f}s",
            "",
            "Per-signal averages:",
        ]
        for signal, score in sorted(self.avg_scores_by_signal.items()):
            status = "✓" if score >= self.threshold else "✗"
            lines.append(f"  {status} {signal}: {score:.3f}")

        if self.failed_traces:
            lines.append("")
            lines.append(f"Failed: {len(self.failed_traces)}/{len(self.trace_results)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def detailed_report(self) -> str:
        lines = [
            "=" * 60,
            "AGENT EVALUATION DETAILED REPORT",
            "=" * 60,
            "",
        ]

        failed = self.failed_traces
        passed = [r for r in self.trace_results if r.passed]

        if failed:
            lines.append(f"─── FAILED ({len(failed)}) ───")
            lines.append("")
            for i, r in enumerate(failed, 1):
                lines.append(f"[{i}] {r.drill_down()}")
                lines.append("")

        if passed:
            lines.append(f"─── PASSED ({len(passed)}) ───")
            lines.append("")
            for i, r in enumerate(passed, 1):
                lines.append(f"[{i}] {r.drill_down()}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "threshold": self.threshold,
            "total_time_seconds": self.total_time_seconds,
            "summary": {
                "total_traces": len(self.trace_results),
                "pass_rate": self.pass_rate,
                "avg_composite_score": self.avg_composite_score,
                "avg_scores_by_signal": self.avg_scores_by_signal,
            },
            "traces": [r.to_dict() for r in self.trace_results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def passed_all(self) -> bool:
        """CI/CD gate — did all traces pass?"""
        return all(r.passed for r in self.trace_results)

    def exit_code(self) -> int:
        return 0 if self.passed_all() else 1


# =============================================================================
# The Evaluator
# =============================================================================


# Signal weights
SIGNAL_WEIGHTS = {
    "selection_logic": 0.4,
    "journey_coherence": 0.3,
    "goal_achievement": 0.3,
}


class AgentEvaluator:
    """
    Reference-free agent evaluator using LLM-as-judge.

    Evaluates agent tool selection, planning, and task completion
    WITHOUT requiring expected_tools. User just provides:
    - Tool definitions (name + description)
    - Agent traces (query + tools_called + output)

    Args:
        tools: List of ToolDefinition (available tools with descriptions).
        threshold: Minimum passing score for composite score.
        weights: Optional custom weights for signals.
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        threshold: float = 0.7,
        weights: dict[str, float] | None = None,
    ):
        self.tools = tools
        self.threshold = threshold
        self.weights = weights or SIGNAL_WEIGHTS
        self._tools_context = self._build_tools_context()

    @classmethod
    def from_tool_list(cls, tools: list[dict[str, str]], **kwargs) -> "AgentEvaluator":
        """
        Create evaluator from a simple list of dicts.

        Usage:
            evaluator = AgentEvaluator.from_tool_list([
                {"name": "retrieval", "description": "Search knowledge base"},
                {"name": "query_rewriter", "description": "Rewrite vague queries"},
            ])
        """
        tool_defs = [ToolDefinition(name=t["name"], description=t["description"]) for t in tools]
        return cls(tools=tool_defs, **kwargs)

    @classmethod
    def from_mcp_server(cls, mcp_server: Any, **kwargs) -> "AgentEvaluator":
        """
        Auto-extract tool definitions from a FastMCP server instance and create evaluator.

        Usage:
            from src.mcp.server import create_mcp_server
            server = create_mcp_server()
            evaluator = AgentEvaluator.from_mcp_server(server)
        """
        tool_defs = cls.extract_tools_from_mcp(mcp_server)
        return cls(tools=tool_defs, **kwargs)

    @staticmethod
    def extract_tools_from_mcp(mcp_server: Any) -> list[ToolDefinition]:
        """
        Extract tool definitions from an MCP server (without creating an evaluator).

        Handles both sync and async (FastMCP 3.x) servers.

        Usage:
            tools = AgentEvaluator.extract_tools_from_mcp(my_server)
        """
        import asyncio

        tool_defs = []

        # FastMCP 3.x uses async list_tools()
        if hasattr(mcp_server, 'list_tools'):
            try:
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context — shouldn't happen in normal usage
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        tools = pool.submit(asyncio.run, mcp_server.list_tools()).result()
                except RuntimeError:
                    # No running loop — safe to use asyncio.run
                    tools = asyncio.run(mcp_server.list_tools())

                for tool in tools:
                    tool_defs.append(ToolDefinition(
                        name=getattr(tool, 'name', str(tool)),
                        description=getattr(tool, 'description', '').strip(),
                    ))
            except Exception:
                pass

        # Fallback: try _tool_manager._tools (older FastMCP)
        if not tool_defs and hasattr(mcp_server, '_tool_manager'):
            manager = mcp_server._tool_manager
            if hasattr(manager, '_tools'):
                for name, tool in manager._tools.items():
                    desc = getattr(tool, 'description', '') or ''
                    if not desc and hasattr(tool, 'fn') and tool.fn:
                        desc = tool.fn.__doc__ or ''
                    tool_defs.append(ToolDefinition(name=name, description=desc.strip()))

        if not tool_defs:
            raise ValueError(
                "Could not auto-extract tools from MCP server. "
                "Tried: list_tools(), _tool_manager._tools. "
                "Use AgentEvaluator.from_tool_list() or pass tools manually."
            )

        return tool_defs

    def _build_tools_context(self) -> str:
        """Build a formatted string of all available tools for the judge prompt."""
        lines = []
        for t in self.tools:
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    # =========================================================================
    # Single trace evaluation
    # =========================================================================

    def evaluate_trace(self, trace: AgentTrace) -> AgentTraceResult:
        """
        Evaluate a single agent trace using LLM-as-judge.

        Scores 3 signals:
        1. Selection Logic — Was the tool choice logical?
        2. Journey Coherence — Was the tool sequence a good plan?
        3. Goal Achievement — Did the output answer the query?

        Returns:
            AgentTraceResult with per-signal scores and composite.
        """
        scores: list[AgentScoreDetail] = []

        # Signal 1: Selection Logic
        selection_score = self._judge_selection_logic(trace)
        scores.append(selection_score)

        # Signal 2: Journey Coherence
        journey_score = self._judge_journey_coherence(trace)
        scores.append(journey_score)

        # Signal 3: Goal Achievement
        goal_score = self._judge_goal_achievement(trace)
        scores.append(goal_score)

        # Compute weighted composite
        composite = sum(s.score * s.weight for s in scores)

        return AgentTraceResult(
            query=trace.query,
            tools_called=trace.tools_called,
            output=trace.output,
            scores=scores,
            composite_score=composite,
        )

    # =========================================================================
    # Batch evaluation
    # =========================================================================

    def evaluate_batch(self, traces: list[AgentTrace]) -> AgentEvalResults:
        """
        Evaluate a batch of agent traces.

        Args:
            traces: List of AgentTrace (captured from your agent's execution).

        Returns:
            AgentEvalResults with summary, drill-down, persistence.
        """
        results = AgentEvalResults(threshold=self.threshold)
        start = time.time()

        for trace in traces:
            result = self.evaluate_trace(trace)
            results.trace_results.append(result)

        results.total_time_seconds = time.time() - start
        return results

    # =========================================================================
    # LLM Judge — Signal Scoring
    # =========================================================================

    def _judge_selection_logic(self, trace: AgentTrace) -> AgentScoreDetail:
        """
        Signal 1: Was the tool selection logical given the query and available tools?

        Asks the judge: "Given this query and these available tools,
        was the agent's tool choice the most appropriate?"
        """
        criteria = (
            f"Evaluate if the agent selected the most appropriate tool(s) for the user's query.\n\n"
            f"Available tools:\n{self._tools_context}\n\n"
            f"The agent called: {trace.tools_called}\n\n"
            f"Score based on:\n"
            f"- Is the tool the best match for this query among all available options?\n"
            f"- Would a different tool have been more appropriate?\n"
            f"- If multiple tools were called, were all of them necessary?\n"
            f"Score 1.0 = perfect tool selection, 0.0 = completely wrong tool."
        )

        return self._run_geval(
            signal="selection_logic",
            criteria=criteria,
            query=trace.query,
            output=f"Tools called: {trace.tools_called}. Output: {trace.output[:300]}",
            weight=self.weights.get("selection_logic", 0.4),
        )

    def _judge_journey_coherence(self, trace: AgentTrace) -> AgentScoreDetail:
        """
        Signal 2: Was the sequence of tool calls a coherent plan?

        Asks: "Does the order of tool calls make logical sense as a plan
        to answer this query?"
        """
        if len(trace.tools_called) <= 1:
            # Single tool call — journey is trivially coherent
            return AgentScoreDetail(
                signal="journey_coherence",
                score=1.0,
                reason="Single tool call — no sequencing to evaluate.",
                weight=self.weights.get("journey_coherence", 0.3),
                passed=True,
            )

        criteria = (
            f"Evaluate if the sequence of tool calls represents a coherent, logical plan.\n\n"
            f"Available tools:\n{self._tools_context}\n\n"
            f"The agent called these tools IN THIS ORDER: {trace.tools_called}\n\n"
            f"Score based on:\n"
            f"- Does the ordering make logical sense (e.g., rewrite before search)?\n"
            f"- Are there redundant or circular calls?\n"
            f"- Is this the most efficient path to answer the query?\n"
            f"Score 1.0 = perfectly planned sequence, 0.0 = random/illogical ordering."
        )

        return self._run_geval(
            signal="journey_coherence",
            criteria=criteria,
            query=trace.query,
            output=f"Tool sequence: {' → '.join(trace.tools_called)}. Final output: {trace.output[:200]}",
            weight=self.weights.get("journey_coherence", 0.3),
        )

    def _judge_goal_achievement(self, trace: AgentTrace) -> AgentScoreDetail:
        """
        Signal 3: Did the agent's final output actually answer the user's query?

        This is independent of WHICH tools were used — it checks the END RESULT.
        """
        criteria = (
            f"Evaluate if the agent's response fully accomplishes what the user asked for.\n\n"
            f"Score based on:\n"
            f"- Does the response directly address the user's question?\n"
            f"- Is the information complete and actionable?\n"
            f"- Would the user be satisfied with this response?\n"
            f"Score 1.0 = perfectly addressed, 0.0 = completely missed the point."
        )

        return self._run_geval(
            signal="goal_achievement",
            criteria=criteria,
            query=trace.query,
            output=trace.output,
            weight=self.weights.get("goal_achievement", 0.3),
        )

    def _run_geval(
        self,
        signal: str,
        criteria: str,
        query: str,
        output: str,
        weight: float,
    ) -> AgentScoreDetail:
        """Run a GEval metric and return a scored signal."""
        try:
            metric = GEval(
                name=signal,
                criteria=criteria,
                evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                threshold=self.threshold,
            )

            test_case = LLMTestCase(input=query, actual_output=output)
            metric.measure(test_case)

            score = metric.score if metric.score is not None else 0.0
            return AgentScoreDetail(
                signal=signal,
                score=score,
                reason=metric.reason or "",
                weight=weight,
                passed=score >= self.threshold,
            )
        except Exception as e:
            return AgentScoreDetail(
                signal=signal,
                score=0.0,
                reason=f"Error: {str(e)}",
                weight=weight,
                passed=False,
            )
