# Example: evaluating a real RAG + MCP pipeline

This is ragprobe used on a **complete, production-style system** — not toy code. The pipeline pulls source from GitHub, chunks it, embeds it into ChromaDB, serves it through a RAG pipeline and a FastMCP tool server, and then evaluates every layer with ragprobe.

> The full runnable source lives in the companion repo **[Evaluation_Framework](https://github.com/anujsahani01/Evaluation_Framework)**. This page walks through how it plugs into ragprobe so you can copy the pattern into your own system.

---

## The pipeline

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   EXTRACT   │──▶│  TRANSFORM  │──▶│    EMBED    │
│  (GitHub)   │   │  (chunking) │   │  (ChromaDB) │
└─────────────┘   └─────────────┘   └─────────────┘
                                           │
                                           ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  ragprobe   │◀──│     MCP     │◀──│     RAG     │
│   (eval)    │   │  (FastMCP)  │   │  (pipeline) │
└─────────────┘   └─────────────┘   └─────────────┘
```

| Stage | What it does |
|-------|--------------|
| Extract | Clone configured GitHub repos, pull matching source files |
| Transform | Code-aware chunking (respects function/class boundaries) |
| Embed | Local or OpenAI embeddings → ChromaDB |
| RAG | Retrieve → optional rerank → generate |
| MCP | FastMCP server exposing `retrieval`, `query_rewriter`, `clarify_query`, `process_query`, … |
| Eval | **ragprobe** scores RAG components *and* MCP agent behavior |

Everything is config-driven (`config/pipeline_config.yaml`) and provider-agnostic (OpenAI, HuggingFace, Ollama, any OpenAI-compatible API).

---

## The ragprobe integration in three pieces

### 1. A RAG adapter over the real pipeline

The adapter wraps the project's `RAGPipeline`, mapping its internals to ragprobe's interface:

```python
from ragprobe.adapters.base import EvalAdapter, RetrievalResult, GenerationResult
from src.rag.pipeline import RAGPipeline

class MyCustomAdapter(EvalAdapter):
    def __init__(self):
        self.rag = RAGPipeline()

    def retrieve(self, query: str) -> RetrievalResult:
        results = self.rag._retrieve(query)
        chunks = results["documents"][0] if results["documents"] else []
        scores = results["distances"][0] if results["distances"] else []
        return RetrievalResult(chunks=chunks, scores=scores)

    def generate(self, query: str, context: list[str]) -> GenerationResult:
        return GenerationResult(answer=self.rag._generate(query, context))
```

### 2. Dataset generated from the real indexed chunks

The same chunks that were embedded into ChromaDB (`chunks.json`) are fed to `DatasetGenerator` — so the test set reflects the actual knowledge base:

```python
from ragprobe import DatasetGenerator, RagEvaluator, ReportGenerator
from src.llm_provider import generate

def my_llm(prompt: str) -> str:
    return generate(prompt).content        # project's provider-agnostic LLM

dataset = DatasetGenerator(llm_fn=my_llm).generate_from_chunks(
    chunks[:20], questions_per_chunk=1, max_samples=3,
)
dataset.save("./eval/datasets/generated_dataset.json")

results = RagEvaluator(adapter=MyCustomAdapter()).evaluate(dataset)
print(results.summary())
results.save("eval_results/run_001.json")
ReportGenerator(results).to_markdown("eval_results/rag_run_001_report.md")
```

### 3. MCP agent evaluation from captured traces

The FastMCP server is traced two ways (see [Tracing](../../docs/tracing.md)):

- **`@traced_tool()`** decorates each tool for in-process capture.
- **`TraceMiddleware`** logs every protocol-level tool call to `traces/session.jsonl`.

Then the agent evaluator scores the captured traces — auto-extracting the tool catalog straight from the server:

```python
from ragprobe import AgentEvaluator, MCPTracer
from src.mcp.server import create_mcp_server

server = create_mcp_server()

# Prefer real captured traces; fall back to a simulation if none exist yet
import os
if os.path.exists("traces/session.jsonl"):
    traces = MCPTracer.load_traces_from_file("traces/session.jsonl")
else:
    tracer = MCPTracer(server)
    tracer.start_interaction("How does the authentication middleware work?")
    tracer.record_call("retrieval", args={"query": "authentication middleware"}, output="...")
    tracer.end_interaction("The auth middleware validates JWT tokens.")
    traces = tracer.get_traces()

evaluator = AgentEvaluator.from_mcp_server(server)
agent_results = evaluator.evaluate_batch(traces)
print(agent_results.summary())
ReportGenerator(agent_results).to_markdown("eval_results/agent_run_001_report.md")
```

Both RAG and agent runs are also logged to MLflow via `log_to_mlflow` / `log_agent_to_mlflow`.

---

## Run it yourself

```bash
# In the Evaluation_Framework repo
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .

cp .env.example .env          # add OPENAI_API_KEY (judge + generation)
mkdir Logs traces             # ragprobe logs + MCP traces

# Build the knowledge base
python -c "from src.extract import GitHubExtractor; GitHubExtractor().extract_all()"
python -c "from src.transform import CodeChunker; from src.embed import Embedder; ..."   # chunk + embed

# Start the MCP server (writes traces/session.jsonl via middleware)
python src/mcp/server.py

# Run the full evaluation (RAG + agent) and generate reports
python ragprobe_adapter.py
```

`Commands.txt` in the repo has the exact copy-paste commands for each stage.

---

## What this example proves

| ragprobe feature | Shown by |
|------------------|----------|
| Plug into any RAG via one adapter | `MyCustomAdapter` over `RAGPipeline` |
| Auto dataset from real data | `generate_from_chunks(chunks)` |
| Per-component RAG scoring | `RagEvaluator.evaluate()` |
| Reference-free agent eval | `AgentEvaluator.from_mcp_server()` |
| Trace capture (decorator + middleware) | `@traced_tool()` + `TraceMiddleware` |
| Reports + MLflow | `ReportGenerator` + `log_*_to_mlflow` |

← Back to the [main docs](../../README.md)
