# Tracing

To evaluate an agent, ragprobe needs a **trace** — the query, the tools called in order, and the final output. Tracing captures these for you so you don't build `AgentTrace`s by hand. Pick the tool that matches how your agent runs.

| Your agent is… | Use |
|----------------|-----|
| Plain Python functions | `@trace_agent` + `@traced_tool` |
| A FastMCP server | `MCPTracer` (middleware, auto-intercept, or manual) |
| Anything, and you want to persist to disk | `MCPTracer(persist=...)` or `TraceStore.export()` |

Everything here produces `AgentTrace` objects ready for `AgentEvaluator.evaluate_batch()` — see [Agent evaluation](agent-evaluation.md).

---

## Option A — Decorators (Python-function agents)

Mark each tool with `@traced_tool` and the top-level entrypoint with `@trace_agent`. Calls are recorded only while inside a traced agent, so your tools behave normally elsewhere.

```python
from ragprobe.tracing import trace_agent, traced_tool

@traced_tool
def search_kb(query: str) -> str:
    return vector_db.search(query)

@traced_tool(name="rewrite")          # override the recorded name
def rewrite_query(query: str) -> str:
    return llm.rewrite(query)

@trace_agent
def my_agent(query: str) -> str:
    q = rewrite_query(query)
    hits = search_kb(q)
    return generate_answer(query, hits)

# Run normally — traces accumulate
my_agent("How does auth work?")
my_agent("What database models exist?")

# Retrieve and evaluate (helpers attached to the wrapped function)
traces = my_agent.get_traces()
my_agent.export_traces("traces/session.json")
my_agent.clear_traces()

from ragprobe import AgentEvaluator, ToolDefinition
evaluator = AgentEvaluator(tools=[
    ToolDefinition(name="search_kb", description="Search the knowledge base"),
    ToolDefinition(name="rewrite", description="Rewrite the query"),
])
results = evaluator.evaluate_batch(traces)
```

- `@trace_agent` takes the **first argument** (or `query`/`input` kwarg) as the query and the return value (or its `answer`/`output` key) as the output.
- Tool `args` and `output` are captured and truncated to keep traces small.

---

## Option B — MCPTracer (FastMCP servers)

When an LLM calls MCP tools over the protocol, there's no Python function of yours to decorate. `MCPTracer` records at the server level. It has three modes.

### Mode 1 — Manual recording

Best for tests and simulations.

```python
from ragprobe import MCPTracer

tracer = MCPTracer(server)           # passing the server also extracts tool defs

tracer.start_interaction("How does auth work?")
tracer.record_call("retrieval", args={"query": "auth"}, output="class AuthMiddleware...")
tracer.end_interaction("The auth middleware validates JWT tokens.")

traces = tracer.get_traces()
```

### Mode 2 — Auto-intercept

Monkey-patches the server's `call_tool` to record every call automatically (FastMCP 3.x async).

```python
tracer = MCPTracer(server, auto_intercept=True)
# every tool call the LLM makes is now recorded
traces = tracer.get_traces()
```

### Mode 3 — Persist to JSONL

Write each finished interaction to a file as it happens, then load later — ideal for capturing real sessions in one process and evaluating in another.

```python
tracer = MCPTracer(server, persist="traces/session.jsonl")
# ... interactions happen ...

# Later / elsewhere:
traces = MCPTracer.load_traces_from_file("traces/session.jsonl")
```

Then, in all three modes:

```python
from ragprobe import AgentEvaluator
evaluator = AgentEvaluator.from_mcp_server(server)   # auto-extracts tool definitions
results = evaluator.evaluate_batch(traces)
```

---

## Option C — FastMCP middleware (capture live protocol traffic)

For a running FastMCP server, add middleware that logs every tool call to JSONL at the protocol level — no per-tool decoration needed. This is the pattern used in the [full example](../examples/evaluation_framework/README.md):

```python
import json, time
from pathlib import Path
from datetime import datetime
from fastmcp.server.middleware import Middleware, MiddlewareContext

class TraceMiddleware(Middleware):
    def __init__(self, traces_path="traces/session.jsonl"):
        self.path = Path(traces_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        start = time.time()
        result = await call_next(context)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": getattr(context.request, "name", ""),
            "args": getattr(context.request, "arguments", {}),
            "output": str(result)[:500],
            "duration_ms": round((time.time() - start) * 1000, 2),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return result

# server.add_middleware(TraceMiddleware("traces/session.jsonl"))
```

Load and evaluate exactly like Mode 3: `MCPTracer.load_traces_from_file("traces/session.jsonl")`.

> **Decorator + middleware together:** In the example server, tools are decorated `@mcp.tool()` on top of `@traced_tool()`, *and* the server adds `TraceMiddleware`. The decorator path feeds in-process traces; the middleware path captures real protocol calls to disk. Use whichever fits — you don't need both.

---

## TraceStore (low-level)

`TraceStore` is the thread-safe store behind `@trace_agent`. You rarely use it directly, but it's handy for saving/loading raw traces:

```python
from ragprobe import TraceStore
store = TraceStore.load("traces/session.json")   # from a prior export
traces = store.get_traces()
```

---

## Which capture path should I choose?

| Goal | Path |
|------|------|
| Quick eval of a Python agent | Decorators (Option A) |
| Test/simulate MCP tool choices | `MCPTracer` manual (Mode 1) |
| Record everything with zero tool edits | `MCPTracer` auto-intercept (Mode 2) |
| Capture real production sessions | Middleware or `persist=` → `load_traces_from_file` |

Next: **[Reporting & MLflow](reporting.md)** →
