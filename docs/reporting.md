# Reporting & MLflow

Turn raw results into something you can share in a PR, archive for compliance, or track across runs on a dashboard. Both `EvalResults` (RAG) and `AgentEvalResults` (agent) work with the same tools.

---

## ReportGenerator

`ReportGenerator` auto-detects whether you passed RAG or agent results and formats accordingly.

```python
from ragprobe import ReportGenerator

report = ReportGenerator(results, title="Nightly RAG Eval")

report.to_markdown("reports/eval.md")   # shareable — renders in GitHub PRs
report.to_json("reports/eval.json")     # machine-readable — dashboards / CI
print(report.to_terminal())             # summary + drill-down to stdout
```

Each method returns the content and (for `to_markdown`/`to_json`) also writes it if you pass a path.

### What's in a Markdown report

- Run metadata (id, timestamp, sample/trace count, latency)
- A summary table of per-metric / per-signal scores with Pass/Fail
- Component (RAG) or per-trace breakdown
- Collapsible per-item details with the judge's reasons
- **Auto-generated recommendations** — e.g. *"faithfulness is below 0.8, tighten your system prompt"* or *"tool selection is weak, review tool descriptions for ambiguity."*

### What's in a JSON report

The same data as a structured object: `summary`, `scores_by_metric` / `scores_by_signal`, every `sample`/`trace`, `recommendations`, and a top-level `verdict` (`"PASS"` / `"FAIL"`) — ready to feed a dashboard or a CI status check.

---

## MLflow logging

Track evaluations as MLflow runs to compare quality over time, across branches, or across config changes. MLflow is an **optional** dependency.

```bash
pip install "ragprobe[mlflow]"
```

> The MLflow helpers live in `ragprobe.integrations.mlflow_logger` — import them from there (they are not exposed at the top level).

### Log a RAG run

```python
from ragprobe.integrations.mlflow_logger import log_to_mlflow

run_id = log_to_mlflow(
    results,
    experiment_name="my_rag_eval",
    run_name="baseline",
    params={"top_k": 5, "model": "gpt-4o-mini"},   # your config, for comparison
    tags={"branch": "main"},
)
```

Logs summary metrics (`pass_rate`, `retrieval_hit_rate`, per-metric and per-component averages, latency), per-query metrics as chart steps, and the full JSON results + a per-query table as artifacts.

### Log an agent run

```python
from ragprobe.integrations.mlflow_logger import log_agent_to_mlflow

log_agent_to_mlflow(
    agent_results,
    experiment_name="my_agent_eval",
    run_name="agent_baseline",
)
```

Logs `pass_rate`, `avg_composite_score`, per-signal averages, per-trace metrics, and artifacts.

### View the dashboard

```bash
mlflow ui        # → http://localhost:5000
```

Compare runs side by side, chart per-query scores, and download the archived JSON/Markdown from each run's artifacts.

---

## A complete reporting workflow

```python
import sys
from ragprobe import RagEvaluator, ReportGenerator
from ragprobe.integrations.mlflow_logger import log_to_mlflow

results = RagEvaluator(adapter=MyRAG()).evaluate(dataset, run_id="run_042")

# 1. Persist raw results (for regression comparison next time)
results.save("eval_results/run_042.json")

# 2. Human-readable report for the PR
ReportGenerator(results).to_markdown("reports/run_042.md")

# 3. Track it in MLflow
log_to_mlflow(results, experiment_name="rag_eval", run_name="run_042")

# 4. Gate the build
sys.exit(results.exit_code())
```

Next: **[API reference](api-reference.md)** →
