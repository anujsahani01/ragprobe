# API reference

Every public symbol, grouped by area. Import top-level names from `ragprobe`; a few extras live in submodules (noted below).

```python
from ragprobe import (
    RagEvaluator, EvalResults, SampleResult, ComponentScore, ComponentScorer,
    EvalAdapter, RetrievalResult, GenerationResult,
    DatasetGenerator, EvalDataset, EvalSample,
    AgentEvaluator, AgentTrace, AgentTraceResult, AgentEvalResults, ToolDefinition,
    trace_agent, traced_tool, TraceStore, MCPTracer,
    ReportGenerator,
    live_eval, get_scores_summary, export_scores,
)
```

---

## Adapters

### `EvalAdapter` (abstract base)

Subclass and implement your pipeline's behavior.

| Method | Signature | Required | Returns |
|--------|-----------|----------|---------|
| `retrieve` | `(self, query: str)` | ✅ | `RetrievalResult` |
| `generate` | `(self, query: str, context: list[str])` | ✅ | `GenerationResult` |
| `rerank` | `(self, query: str, chunks: list[str])` | — | `list[str]` (default: unchanged) |
| `rewrite_query` | `(self, query: str)` | — | `str` (default: unchanged) |
| `get_tools` | `(self)` | — | `list[str]` (default: `[]`) |
| `call_tool` | `(self, tool_name: str, **kwargs)` | — | `Any` (default: raises) |

### `RetrievalResult`

```python
RetrievalResult(chunks: list[str], scores: list[float] = [], metadata: list[dict] = [])
```

### `GenerationResult`

```python
GenerationResult(answer: str, metadata: dict = {})
```

---

## Dataset generation

### `DatasetGenerator`

```python
DatasetGenerator(llm_fn: Callable[[str], str])
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `generate_from_chunks` | `(chunks: list[dict], questions_per_chunk=1, max_samples=None)` | `EvalDataset` |

`chunks` = list of dicts, each with a required `"content"` key (plus optional `chunk_id` and arbitrary metadata). Chunks under 50 chars are skipped.

### `EvalDataset`

Iterable, `len()`-able collection of `EvalSample`.

| Member | Signature |
|--------|-----------|
| `samples` | `list[EvalSample]` |
| `save` | `(path)` → JSON |
| `load` *(classmethod)* | `(path)` → `EvalDataset` |

### `EvalSample`

```python
EvalSample(question, expected_answer, source_chunk, chunk_id="", metadata={})
```

---

## RAG evaluation

### `RagEvaluator`

```python
RagEvaluator(adapter: EvalAdapter, metrics: list[str] | None = None, threshold: float = 0.7)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `evaluate` | `(dataset: EvalDataset, run_id: str | None = None)` | `EvalResults` |

**Available metrics** (`metrics=None` uses all): `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`, `context_relevancy`.

### `EvalResults`

| Property / method | Returns | Notes |
|-------------------|---------|-------|
| `summary()` | `str` | scoreboard |
| `detailed_report()` | `str` | per-query drill-down |
| `save(path)` / `load(path)` | — / `EvalResults` | JSON persistence |
| `compare_with(previous_path)` | `str` | regression report |
| `passed()` | `bool` | all metric averages ≥ threshold |
| `exit_code()` | `int` | `0` pass / `1` fail |
| `pass_rate` | `float` | % samples passing all metrics |
| `retrieval_hit_rate` | `float` | % where source chunk retrieved |
| `avg_scores_by_metric` | `dict[str, float]` | |
| `avg_scores_by_component` | `dict[str, float]` | `retrieval` / `generation` |
| `avg_latency_ms` | `float` | |
| `failed_samples` | `list[SampleResult]` | |
| `sample_results` | `list[SampleResult]` | |

### `SampleResult`

`retrieval_hit`, `avg_score`, `passed_all`, `failed_metrics`, `drill_down()`, `component_scores: list[ComponentScore]`.

### `ComponentScore`

```python
ComponentScore(component, metric_name, score, reason="", latency_ms=0.0, passed=False)
```

---

## Agent / MCP evaluation

### `AgentEvaluator`

```python
AgentEvaluator(tools: list[ToolDefinition], threshold: float = 0.7, weights: dict | None = None)
```

| Constructor / method | Signature | Returns |
|----------------------|-----------|---------|
| `from_tool_list` *(classmethod)* | `(tools: list[dict], **kw)` | `AgentEvaluator` |
| `from_mcp_server` *(classmethod)* | `(mcp_server, **kw)` | `AgentEvaluator` |
| `extract_tools_from_mcp` *(staticmethod)* | `(mcp_server)` | `list[ToolDefinition]` |
| `evaluate_trace` | `(trace: AgentTrace)` | `AgentTraceResult` |
| `evaluate_batch` | `(traces: list[AgentTrace])` | `AgentEvalResults` |

Default `weights`: `{"selection_logic": 0.4, "journey_coherence": 0.3, "goal_achievement": 0.3}`.

### `ToolDefinition`

```python
ToolDefinition(name: str, description: str, parameters: dict = {})
```

### `AgentTrace`

```python
AgentTrace(query, tools_called: list[str], output,
           tool_args: list[dict] = [], tool_outputs: list[str] = [], metadata: dict = {})
```

### `AgentTraceResult`

`query`, `tools_called`, `output`, `scores: list[AgentScoreDetail]`, `composite_score`, `passed`, `summary_line()`, `drill_down()`.

### `AgentEvalResults`

| Property / method | Returns |
|-------------------|---------|
| `summary()` / `detailed_report()` | `str` |
| `save(path)` | — |
| `passed_all()` | `bool` |
| `exit_code()` | `int` |
| `pass_rate` | `float` |
| `avg_composite_score` | `float` |
| `avg_scores_by_signal` | `dict[str, float]` |
| `failed_traces` | `list[AgentTraceResult]` |
| `trace_results` | `list[AgentTraceResult]` |

---

## Tracing

### `@trace_agent` / `@traced_tool`

```python
from ragprobe.tracing import trace_agent, traced_tool

@traced_tool                    # or @traced_tool(name="custom")
def my_tool(...): ...

@trace_agent
def my_agent(query): ...
```

`@trace_agent` attaches to the wrapped function: `get_traces()`, `clear_traces()`, `export_traces(path)`, `trace_count`.

### `MCPTracer`

```python
MCPTracer(server=None, auto_intercept: bool = False, persist: str | None = None)
```

| Method | Signature |
|--------|-----------|
| `start_interaction` | `(query: str)` |
| `record_call` | `(tool_name, args=None, output=None, duration_ms=0.0, query=None)` |
| `end_interaction` | `(final_output: str)` |
| `get_traces` | `() → list[AgentTrace]` |
| `get_interactions` | `() → list[MCPInteraction]` |
| `export` | `(path)` |
| `clear` | `()` |
| `load_traces_from_file` *(classmethod)* | `(path) → list[AgentTrace]` |
| `tools` *(property)* | `list[ToolDefinition]` |

### `TraceStore`

`start_trace(query)`, `record_tool_call(...)`, `end_trace(output)`, `get_traces()`, `clear()`, `export(path)`, `load(path)` *(classmethod)*, `trace_count`.

---

## Reporting

### `ReportGenerator`

```python
ReportGenerator(results: EvalResults | AgentEvalResults, title: str | None = None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `to_markdown` | `(path=None)` | `str` |
| `to_json` | `(path=None)` | `dict` |
| `to_terminal` | `()` | `str` |

---

## Live evaluation

### `@live_eval`

```python
from ragprobe import live_eval

@live_eval(metrics: list[str] | None = None, threshold: float = 0.7, log_to_file: str | None = None)
def pipeline(query) -> dict:      # must return {"answer": str, "retrieval_context": list[str]}
    ...
```

Available live metrics: `answer_relevancy`, `faithfulness`, `context_relevancy`, `context_precision`.

### Module helpers

```python
from ragprobe import get_scores_summary, export_scores
from ragprobe.evaluator.live import get_scores, clear_scores
```

| Function | Returns |
|----------|---------|
| `get_scores_summary()` | `dict` rolling stats |
| `export_scores(path)` | — |
| `get_scores()` | `list[LiveScore]` |
| `clear_scores()` | — |

---

## Component scoring

Score retrieval and generation in isolation.

```python
from ragprobe import ComponentScorer

ComponentScorer(threshold: float = 0.7)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `score_from_dataset` | `(dataset, adapter=None)` | `list[dict]` — per-sample `{question, retrieval, generation}` |
| `score_retrieval` | `(query, retrieved_chunks, expected_answer, actual_answer="")` | `list[ScoreResult]` |
| `score_generation` | `(query, answer, context)` | `list[ScoreResult]` |
| `score_reranker` | `(query, original_order, reranked_order, expected_answer)` | `list[ScoreResult]` |
| `score_chunk_quality` | `(chunk)` | `list[ScoreResult]` |

`ScoreResult` = `(metric, score, reason, passed)`. When `adapter` is passed to `score_from_dataset`, it runs `retrieve` + `generate` live; when omitted, it scores against the dataset's own `source_chunk` / `expected_answer`.

---

## Integrations (submodule)

```python
from ragprobe.integrations import log_to_mlflow, log_agent_to_mlflow
```

| Function | Signature |
|----------|-----------|
| `log_to_mlflow` | `(results, experiment_name="ragprobe_rag_eval", run_name=None, params=None, tags=None) → str` |
| `log_agent_to_mlflow` | `(results, experiment_name="ragprobe_agent_eval", run_name=None, params=None, tags=None) → str` |

Requires the `mlflow` extra: `pip install "ragprobe[mlflow] @ git+https://github.com/anujsahani01/ragprobe.git"`. (These are re-exported from `ragprobe.integrations`; the underlying module is `ragprobe.integrations.mlflow_logger`.)

---

## Notes & known gotchas

- **Judge key required:** metrics call an LLM judge (DeepEval), so `OPENAI_API_KEY` must be set.
- **`Logs/` directory:** importing ragprobe configures a log file under `Logs/`. Create the directory (`mkdir Logs`) before first run to avoid a startup error.
- **MLflow helpers import from `ragprobe.integrations`** (not the top-level package): `log_to_mlflow` / `log_agent_to_mlflow`. They require the `[mlflow]` extra.
