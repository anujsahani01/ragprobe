# ragprobe

**Pluggable evaluation for RAG pipelines and AI agents.**

`pip install ragprobe` → wrap your pipeline in an adapter → get per-component scores. No test cases to hand-write, no ground truth required.

```
   Your RAG / Agent  ──▶  ragprobe  ──▶  Scores · Reports · CI gates
                            │
        ┌───────────────────┼───────────────────┐
     RAG eval           Agent eval            Tracing
  (per component)   (reference-free judge)  (auto-capture)
```

ragprobe is a thin, opinionated layer on top of [DeepEval](https://github.com/confident-ai/deepeval). DeepEval gives you the metrics; ragprobe gives you the **plumbing** — adapters, auto-generated datasets, per-component scoring, reference-free agent evaluation, trace capture, reports, MLflow logging, and CI regression gates — so you stop wiring metrics by hand.

---

## 30-second example

```python
from ragprobe import RagEvaluator, EvalAdapter, DatasetGenerator
from ragprobe.adapters.base import RetrievalResult, GenerationResult

# 1. Tell ragprobe how to call YOUR pipeline
class MyRAG(EvalAdapter):
    def retrieve(self, query):
        return RetrievalResult(chunks=my_db.search(query, top_k=5))
    def generate(self, query, context):
        return GenerationResult(answer=my_llm(query, context))

# 2. Auto-build a test set from your real chunks (no manual QA writing)
dataset = DatasetGenerator(llm_fn=my_llm_fn).generate_from_chunks(my_chunks)

# 3. Score every component
results = RagEvaluator(adapter=MyRAG()).evaluate(dataset)
print(results.summary())
```

```
============================================================
RAG EVALUATION RESULTS
============================================================
Samples: 20 | Pass rate: 85.0% | Time: 34.2s
Retrieval hit rate: 90.0%
Per-metric averages:
  ✓ answer_relevancy: 0.91
  ✓ faithfulness: 0.88
  ✗ context_relevancy: 0.62   ← your retriever is pulling in noise
============================================================
```

---

## What it evaluates

| Component | Metrics | Ground truth needed? |
|-----------|---------|----------------------|
| **Retrieval** | context precision, recall, relevancy | No — auto-generated from your chunks |
| **Generation** | faithfulness, answer relevancy | No |
| **Agent / MCP** | tool selection, journey coherence, goal achievement | No — reference-free LLM judge |

Plus: **auto dataset generation**, **live scoring** decorators, **trace capture** for agents & MCP, **Markdown/JSON reports**, **MLflow logging**, and **CI/CD gates** with regression detection.

---

## Install

```bash
pip install ragprobe                 # core
pip install "ragprobe[mlflow]"       # + MLflow logging
```

Requires **Python 3.11+**. The metrics use an LLM judge (via DeepEval), so set an API key:

```bash
export OPENAI_API_KEY=sk-...         # macOS/Linux
setx OPENAI_API_KEY "sk-..."         # Windows
```

---

## Documentation

New here? Read in this order:

1. **[Quickstart](docs/quickstart.md)** — install to first scores in 5 minutes.
2. **[Core concepts](docs/concepts.md)** — the mental model (adapter, dataset, metrics, signals, traces).
3. **[RAG evaluation](docs/rag-evaluation.md)** — datasets, metrics, regression tracking, CI gates.
4. **[Agent / MCP evaluation](docs/agent-evaluation.md)** — reference-free scoring of tool-using agents.
5. **[Tracing](docs/tracing.md)** — capture agent & MCP traces with decorators or middleware.
6. **[Reporting & MLflow](docs/reporting.md)** — shareable reports and experiment tracking.
7. **[API reference](docs/api-reference.md)** — every public symbol.

Want to see it on a **real system**? → **[examples/evaluation_framework](examples/evaluation_framework/README.md)** — a full GitHub → chunk → embed → RAG → MCP pipeline evaluated end to end with ragprobe.

---

## How ragprobe compares

| | ragprobe | DeepEval (raw) | RAGAS |
|---|----------|----------------|-------|
| Plug into *any* RAG via one adapter | ✅ | ⚠️ manual test cases | ⚠️ dataframe wrangling |
| Auto-generate dataset from your chunks | ✅ | ❌ | ⚠️ separate module |
| Per-component scores (retrieval vs generation) | ✅ | ⚠️ you assemble it | ⚠️ |
| Reference-free agent / MCP evaluation | ✅ | ❌ | ❌ |
| Built-in trace capture (decorators + MCP) | ✅ | ❌ | ❌ |
| Reports + MLflow + CI regression gates | ✅ | ⚠️ partial | ❌ |

ragprobe doesn't replace DeepEval — it *stands on it* and removes the boilerplate.

---

## License

MIT © Anuj Sahani
