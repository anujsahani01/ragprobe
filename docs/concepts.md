# Core concepts

Five ideas explain everything ragprobe does. Once these click, the rest of the docs are just detail.

---

## The big picture

ragprobe sits *between* your system and an LLM judge. You describe your pipeline once (an **adapter** or a **trace**), ragprobe runs it against test data, and DeepEval's metrics score the output.

```
                 ┌──────────── ragprobe ────────────┐
  Your system ──▶│  Adapter / Trace  →  Metrics/Judge │──▶  Results → Report → CI
                 └───────────────────────────────────┘
```

---

## 1. The Adapter (RAG)

An **adapter** is a small class that tells ragprobe how to call your pipeline. You implement two required methods:

| Method | You return | Purpose |
|--------|-----------|---------|
| `retrieve(query)` | `RetrievalResult(chunks=[...])` | what your retriever fetched |
| `generate(query, context)` | `GenerationResult(answer="...")` | what your generator produced |

Optional hooks — `rerank()`, `rewrite_query()`, `get_tools()`, `call_tool()` — let you expose more of your pipeline when you have it. Everything scoring-related is handled for you.

> The adapter is the *only* code you must write to evaluate RAG. See [RAG evaluation](rag-evaluation.md).

---

## 2. The Dataset

An evaluation needs questions with known answers. Instead of writing them by hand, `DatasetGenerator` **reads your real chunks** and generates them:

```
chunk of your code/docs  ──LLM──▶  { question, expected_answer, source_chunk, chunk_id }
```

Because each sample remembers its `source_chunk`, ragprobe can check whether retrieval actually surfaced the right chunk (the **retrieval hit rate**). A dataset is a list of `EvalSample`s and can be `.save()`d / `.load()`ed as JSON.

---

## 3. Metrics (RAG) vs Signals (Agent)

ragprobe scores two very different things, so it uses two vocabularies.

**RAG metrics** (need retrieved context; grouped by component):

| Component | Metrics |
|-----------|---------|
| Retrieval | `context_precision`, `context_recall`, `context_relevancy` |
| Generation | `answer_relevancy`, `faithfulness` |

**Agent signals** (reference-free — no expected tools required):

| Signal | Weight | Question it answers |
|--------|--------|---------------------|
| `selection_logic` | 40% | Was the chosen tool the best fit for the query? |
| `journey_coherence` | 30% | Did the *sequence* of tool calls make sense as a plan? |
| `goal_achievement` | 30% | Did the final output actually answer the user? |

Every score is 0.0–1.0 and passes if it meets the **threshold** (default `0.7`). The agent composite is the weighted average of the three signals.

---

## 4. Traces (Agent)

To evaluate an agent, ragprobe needs a record of what happened: the query, the tools it called (in order), and the final output. That record is an **`AgentTrace`**.

You can build traces three ways:

- **By hand** — construct `AgentTrace(query=..., tools_called=[...], output=...)`.
- **With decorators** — `@trace_agent` + `@traced_tool` auto-capture them as your agent runs.
- **From MCP** — `MCPTracer` records tool calls from a FastMCP server (manually, via auto-intercept, or via middleware writing JSONL).

See [Tracing](tracing.md).

---

## 5. Results, Reports, and Gates

Every evaluation returns a **results object** (`EvalResults` for RAG, `AgentEvalResults` for agents) that all share the same moves:

```python
results.summary()            # human-readable scoreboard
results.detailed_report()    # per-item drill-down + reasons
results.save("run.json")     # persist
results.exit_code()          # 0 / 1 for CI
```

From there:

- **`ReportGenerator(results)`** → `.to_markdown()` / `.to_json()` / `.to_terminal()`.
- **`compare_with("previous.json")`** (RAG) → regression report between two runs.
- **`log_to_mlflow(results)`** → track experiments over time.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Adapter** | Class wrapping your RAG so ragprobe can call it |
| **Sample** | One `{question, expected_answer, source_chunk}` triple |
| **Metric** | A DeepEval-backed RAG score (retrieval/generation) |
| **Signal** | A reference-free agent score (selection/journey/goal) |
| **Trace** | A recorded agent interaction (query → tools → output) |
| **Threshold** | Minimum score to "pass" (default 0.7) |
| **Hit rate** | % of queries where the correct source chunk was retrieved |

Next: **[RAG evaluation](rag-evaluation.md)** →
