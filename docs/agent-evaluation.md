# Agent / MCP evaluation

Score tool-using agents **without ground truth**. You don't tell ragprobe which tools *should* have been called — an LLM judge reasons about whether the agent's choices were sound. Works with any framework: MCP, LangChain, LlamaIndex, or custom.

---

## The mental model

ragprobe evaluates a **trace** — a record of one interaction — against three reference-free **signals**:

| Signal | Weight | Question |
|--------|--------|----------|
| `selection_logic` | 40% | Given the available tools, was the chosen tool the best fit? |
| `journey_coherence` | 30% | Did the *order* of tool calls form a logical plan? |
| `goal_achievement` | 30% | Did the final output actually answer the query? |

The **composite score** is the weighted average; a trace passes if composite ≥ threshold (default `0.7`). A single-tool call scores `journey_coherence = 1.0` automatically (nothing to sequence).

---

## 1. Define the available tools

The judge needs to know what the agent *could* have picked. Provide `ToolDefinition`s (name + description):

```python
from ragprobe import AgentEvaluator, ToolDefinition

evaluator = AgentEvaluator(
    tools=[
        ToolDefinition(name="retrieval", description="Search the code knowledge base"),
        ToolDefinition(name="query_rewriter", description="Rewrite vague or ambiguous queries"),
        ToolDefinition(name="clarify_query", description="Ask the user for clarification"),
    ],
    threshold=0.7,
)
```

Shortcuts:

```python
# From a list of dicts
evaluator = AgentEvaluator.from_tool_list([
    {"name": "retrieval", "description": "Search the code knowledge base"},
])

# Auto-extract from a FastMCP server (reads the tools registered on it)
from src.mcp.server import create_mcp_server
evaluator = AgentEvaluator.from_mcp_server(create_mcp_server())
```

---

## 2. Provide traces

A trace is what the agent did. Build one directly:

```python
from ragprobe import AgentTrace

trace = AgentTrace(
    query="How does authentication work?",
    tools_called=["query_rewriter", "retrieval"],   # in call order
    output="The auth middleware validates JWT tokens from the Authorization header.",
    tool_args=[{"q": "..."}, {"q": "auth middleware"}],   # optional
    tool_outputs=["auth middleware jwt", "class AuthMiddleware: ..."],  # optional
)
```

Most of the time you won't build these by hand — you'll **capture** them automatically. See **[Tracing](tracing.md)** for `@trace_agent`, `@traced_tool`, and `MCPTracer`.

---

## 3. Evaluate

```python
# Single trace
result = evaluator.evaluate_trace(trace)
print(result.summary_line())
print(result.drill_down())

# A batch
results = evaluator.evaluate_batch([trace1, trace2, trace3])
print(results.summary())
```

```
============================================================
AGENT EVALUATION RESULTS (Reference-Free)
============================================================
Traces evaluated: 3
Pass rate: 66.7%
Avg composite score: 0.741
Per-signal averages:
  ✓ selection_logic: 0.83
  ✓ journey_coherence: 0.90
  ✗ goal_achievement: 0.61
============================================================
```

`drill_down()` shows each signal's score, weight, and the judge's reasoning — so a failure tells you *whether the agent picked the wrong tool, sequenced poorly, or just produced a weak answer.*

---

## 4. Inspect, report, and gate

```python
results.avg_scores_by_signal    # {"selection_logic": 0.83, ...}
results.failed_traces           # traces below threshold
results.avg_composite_score

# Save / report
results.save("eval_results/agent_run.json")
from ragprobe import ReportGenerator
ReportGenerator(results).to_markdown("reports/agent.md")

# CI gate — passes only if EVERY trace passed
import sys
sys.exit(results.exit_code())
```

---

## Tuning the signal weights

If your use case cares more about end results than tool choreography, re-weight:

```python
evaluator = AgentEvaluator(
    tools=tools,
    weights={"selection_logic": 0.2, "journey_coherence": 0.2, "goal_achievement": 0.6},
)
```

Weights should sum to 1.0. Defaults are `0.4 / 0.3 / 0.3`.

---

## When to use what

| Situation | Approach |
|-----------|----------|
| Python-function agent | `@trace_agent` + `@traced_tool` → `evaluate_batch(fn.get_traces())` |
| FastMCP server | `MCPTracer` (middleware or auto-intercept) → `evaluate_batch(traces)` |
| Already have logs of tool calls | Build `AgentTrace`s directly |
| You *do* know the expected tools | Provide them for strict matching alongside the reference-free signals |

Next: **[Tracing](tracing.md)** →
