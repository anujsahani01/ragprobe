# Quickstart

Go from install to your first scores in about 5 minutes. Copy-paste each block in order.

> **Prerequisites:** Python 3.11+ and an OpenAI API key (ragprobe's metrics use an LLM judge under the hood).

---

## 1. Install

```bash
pip install ragprobe
```

Set your judge key (metrics call an LLM to score quality):

```bash
export OPENAI_API_KEY=sk-...      # macOS / Linux
setx OPENAI_API_KEY "sk-..."      # Windows (new terminal after)
```

> **One-time gotcha:** ragprobe writes a log file to `Logs/` on import. Create it first so nothing errors out:
>
> ```bash
> mkdir Logs
> ```

---

## 2. Wrap your RAG in an adapter

An **adapter** is the only thing you write. It teaches ragprobe how to call *your* retriever and generator. Only two methods are required.

```python
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult

class MyRAG(EvalAdapter):
    def retrieve(self, query: str) -> RetrievalResult:
        # Return your retrieved chunks as a list of strings.
        chunks = ["def login(): ...", "class AuthMiddleware: ..."]
        return RetrievalResult(chunks=chunks)

    def generate(self, query: str, context: list[str]) -> GenerationResult:
        # Return your model's answer given the retrieved context.
        answer = "The auth middleware validates JWT tokens."
        return GenerationResult(answer=answer)
```

Replace the bodies with real calls to your vector DB and LLM. That's the whole integration.

---

## 3. Build an evaluation dataset (automatically)

You don't hand-write questions. `DatasetGenerator` reads your **real chunks** and asks an LLM to produce a question + expected answer for each one. If retrieval later fails to surface the source chunk, you know the retriever is broken.

```python
from ragprobe import DatasetGenerator

# llm_fn is any callable: prompt string in, response string out.
def my_llm_fn(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# Chunks are dicts that must have a "content" key (chunk_id/metadata optional).
my_chunks = [
    {"content": "def login(user, pw): return verify(user, pw)", "chunk_id": "auth::1"},
    {"content": "class AuthMiddleware: # validates JWT from header", "chunk_id": "auth::2"},
]

dataset = DatasetGenerator(llm_fn=my_llm_fn).generate_from_chunks(
    my_chunks,
    questions_per_chunk=1,
)
dataset.save("eval/dataset.json")   # reuse it later
print(f"Generated {len(dataset)} samples")
```

---

## 4. Run the evaluation

```python
from ragprobe import RagEvaluator

evaluator = RagEvaluator(adapter=MyRAG())
results = evaluator.evaluate(dataset)

print(results.summary())          # the scoreboard
print(results.detailed_report())  # per-query drill-down with failure reasons
```

`summary()` gives you the headline numbers; `detailed_report()` shows exactly which query failed which metric and *why* — that's what makes the scores actionable.

---

## 5. Save, report, and gate

```python
# Persist for regression tracking
results.save("eval_results/run_001.json")

# Generate a shareable Markdown report
from ragprobe import ReportGenerator
ReportGenerator(results).to_markdown("reports/eval.md")

# Fail your CI build if quality drops below threshold (default 0.7)
import sys
sys.exit(results.exit_code())     # 0 = pass, 1 = fail
```

---

## You just evaluated a RAG pipeline end to end.

Where to go next:

- **[Core concepts](concepts.md)** — understand adapters, metrics, and signals properly.
- **[RAG evaluation](rag-evaluation.md)** — pick metrics, track regressions, tune thresholds.
- **[Agent / MCP evaluation](agent-evaluation.md)** — evaluate tool-using agents (no ground truth).
- **[A full real-world example](../examples/evaluation_framework/README.md)** — ragprobe on a production-style pipeline.
