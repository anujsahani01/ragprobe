# RAG evaluation

Everything for scoring a Retrieval-Augmented Generation pipeline: writing the adapter, generating a dataset, choosing metrics, reading results, tracking regressions, and gating CI.

---

## 1. Write an adapter

The adapter is your pipeline's contract with ragprobe. Subclass `EvalAdapter` and implement the two required methods.

```python
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult

class MyRAG(EvalAdapter):
    def __init__(self):
        self.db = my_vector_db
        self.llm = my_llm_client

    def retrieve(self, query: str) -> RetrievalResult:
        hits = self.db.search(query, top_k=5)
        return RetrievalResult(
            chunks=[h.text for h in hits],      # required: list[str]
            scores=[h.score for h in hits],     # optional
            metadata=[h.meta for h in hits],    # optional
        )

    def generate(self, query: str, context: list[str]) -> GenerationResult:
        answer = self.llm.complete(query=query, context=context)
        return GenerationResult(answer=answer)  # required: str
```

### Optional hooks

Override these only if your pipeline has them:

| Method | Default | Use when |
|--------|---------|----------|
| `rerank(query, chunks)` | returns chunks unchanged | you have a reranker |
| `rewrite_query(query)` | returns query unchanged | you do query rewriting |
| `get_tools()` | `[]` | you want agent evaluation too |
| `call_tool(name, **kw)` | raises | your adapter can invoke tools |

---

## 2. Generate a dataset

`DatasetGenerator` turns your real chunks into `{question, expected_answer, source_chunk}` samples using any LLM you provide.

```python
from ragprobe import DatasetGenerator

def my_llm_fn(prompt: str) -> str:      # prompt in → text out
    return my_llm.complete(prompt)

gen = DatasetGenerator(llm_fn=my_llm_fn)
dataset = gen.generate_from_chunks(
    chunks,                     # list[dict] — each MUST have a "content" key
    questions_per_chunk=2,      # more questions = broader coverage
    max_samples=50,             # cap total (None = no cap)
)

dataset.save("eval/dataset.json")
```

**Chunk format:** a list of dicts. `content` is required; `chunk_id` and any other keys (`file_path`, `repo`, …) are preserved as metadata. Chunks shorter than 50 characters are skipped. Reload later with `EvalDataset.load("eval/dataset.json")`.

---

## 3. Run the evaluator

```python
from ragprobe import RagEvaluator

evaluator = RagEvaluator(
    adapter=MyRAG(),
    metrics=None,        # None = all 5 metrics; or pass a subset (see below)
    threshold=0.7,       # pass/fail line for every metric
)

results = evaluator.evaluate(dataset, run_id="baseline")
```

For each sample the evaluator calls `retrieve()` → `generate()`, times both, then scores the result with DeepEval.

---

## 4. The metrics

| Metric | Component | Answers |
|--------|-----------|---------|
| `context_precision` | retrieval | Are the *relevant* chunks ranked near the top? |
| `context_recall` | retrieval | Did retrieval fetch everything needed to answer? |
| `context_relevancy` | retrieval | Is the retrieved context on-topic (low noise)? |
| `answer_relevancy` | generation | Does the answer address the question? |
| `faithfulness` | generation | Is the answer grounded in context (no hallucination)? |

Evaluate a **subset** by name:

```python
evaluator = RagEvaluator(adapter=MyRAG(), metrics=["faithfulness", "context_recall"])
```

---

## 5. Read the results

### Scoreboard

```python
print(results.summary())
```

```
============================================================
RAG EVALUATION RESULTS
============================================================
Samples: 20 | Pass rate: 85.0% | Time: 34.2s
Retrieval hit rate: 90.0%
Avg latency: 812ms/query
Per-metric averages:
  ✓ answer_relevancy: 0.910
  ✓ faithfulness: 0.880
  ✗ context_relevancy: 0.620
Per-component averages:
  ✓ generation: 0.895
  ✗ retrieval: 0.700
============================================================
```

### Per-query drill-down

```python
print(results.detailed_report())   # shows FAILED samples first, with reasons
```

### Useful properties

```python
results.pass_rate               # % samples passing ALL metrics
results.retrieval_hit_rate      # % where the source chunk was retrieved
results.avg_scores_by_metric    # {"faithfulness": 0.88, ...}
results.avg_scores_by_component # {"retrieval": 0.70, "generation": 0.90}
results.avg_latency_ms
results.failed_samples          # list[SampleResult] that failed something
```

Each `SampleResult` exposes `.retrieval_hit`, `.avg_score`, `.passed_all`, `.failed_metrics`, and `.drill_down()`.

---

## 6. Persist & detect regressions

Save every run, then compare a new run against a previous one:

```python
results.save("eval_results/run_002.json")

report = results.compare_with("eval_results/run_001.json")
print(report)
```

```
============================================================
REGRESSION REPORT
Current:  run_002    Previous: run_001
============================================================
Pass rate: 85.0% → 78.0% (↓ 7.0%)
Per-metric changes:
  ↓ faithfulness: 0.880 → 0.810 (-0.070)
  → answer_relevancy: 0.910 → 0.912 (+0.002)
⚠ REGRESSIONS DETECTED in 1 metric(s):
    faithfulness: dropped 0.070 (0.880 → 0.810)
============================================================
```

---

## 7. Gate your CI

`passed()` returns `True` only if **every** metric's average clears the threshold. `exit_code()` maps that to `0`/`1`.

```python
import sys
results = evaluator.evaluate(dataset)
results.save("eval_results/latest.json")
if not results.passed():
    print(results.detailed_report())
sys.exit(results.exit_code())     # non-zero fails the pipeline
```

```yaml
# .github/workflows/eval.yml
- run: python run_eval.py
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## 8. Score components in isolation

`RagEvaluator` gives you the aggregate picture. When you want to drill into *one component at a time* — "is it retrieval or generation that's dragging the score down?" — use `ComponentScorer`. It runs the retrieval and generation metrics separately.

```python
from ragprobe import ComponentScorer

scorer = ComponentScorer(threshold=0.7)

# Score a whole dataset through your adapter (runs retrieve + generate live)
results = scorer.score_from_dataset(dataset, adapter=MyRAG())

for row in results:
    print(row["question"])
    for component in ("retrieval", "generation"):
        for metric, data in row[component].items():
            status = "PASS" if data["passed"] else "FAIL"
            print(f"  [{component}] {metric}: {data['score']:.3f} ({status})")
```

`score_from_dataset(dataset, adapter=...)` returns a list of dicts — one per sample — each shaped like:

```python
{
  "question": "...",
  "retrieval":  {"context_precision": {"score": .., "passed": .., "reason": ".."}, ...},
  "generation": {"answer_relevancy":  {"score": .., "passed": .., "reason": ".."}, ...},
}
```

Omit `adapter` to score against the dataset's own `source_chunk`/`expected_answer` (a sanity check on the data itself). For finer control there are also `score_retrieval(...)`, `score_generation(...)`, `score_reranker(...)`, and `score_chunk_quality(...)`.

---

## 9. Live evaluation (score in production)

Instead of a batch run, score every call as it flows through your pipeline with the `@live_eval` decorator. Your function must return a dict with `answer` and `retrieval_context`.

```python
from ragprobe import live_eval, get_scores_summary, export_scores

@live_eval(metrics=["faithfulness", "answer_relevancy"], threshold=0.7)
def my_pipeline(query: str) -> dict:
    chunks = retrieve(query)
    return {"answer": generate(query, chunks), "retrieval_context": chunks}

my_pipeline("How does auth work?")   # scored automatically

print(get_scores_summary())          # rolling stats
export_scores("live_scores/session.json")
```

> Live scoring calls the LLM judge on every request — great for staging/canary traffic, use sampling for high-volume production.

---

Next: **[Agent / MCP evaluation](agent-evaluation.md)** →
