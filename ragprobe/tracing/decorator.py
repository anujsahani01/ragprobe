"""
Trace Agent Decorator
=====================
Automatic trace capture for any agent function.

Two decorators:
1. @trace_agent — wraps the top-level agent function, captures query + final output
2. @traced_tool — wraps individual tool functions, captures each tool call

Usage:
    from ragprobe.tracing import trace_agent, traced_tool

    # Mark individual tools
    @traced_tool
    def search_knowledge_base(query: str) -> str:
        return vector_db.search(query)

    @traced_tool
    def rewrite_query(query: str) -> str:
        return llm.rewrite(query)

    # Mark the top-level agent
    @trace_agent
    def my_agent(query: str) -> str:
        rewritten = rewrite_query(query)
        results = search_knowledge_base(rewritten)
        answer = generate_answer(query, results)
        return answer

    # Run your agent normally
    my_agent("How does auth work?")
    my_agent("What database models exist?")

    # Get captured traces (zero extra code!)
    traces = my_agent.get_traces()

    # Evaluate
    from ragprobe import AgentEvaluator, ToolDefinition
    evaluator = AgentEvaluator(tools=[...])
    results = evaluator.evaluate_batch(traces)
"""

from __future__ import annotations

import time
import functools
from typing import Callable, Any

from ragprobe.tracing.store import TraceStore


# Global trace store — shared between @trace_agent and @traced_tool
_global_store = TraceStore()

# Flag to track if we're inside a traced agent call
_tracing_active = False


def trace_agent(fn: Callable) -> Callable:
    """
    Decorator for the top-level agent function.

    Captures:
    - The input query (first argument)
    - All @traced_tool calls made during execution
    - The final output

    Attaches helper methods to the wrapped function:
    - fn.get_traces() → list[AgentTrace]
    - fn.clear_traces() → None
    - fn.export_traces(path) → None
    - fn.trace_count → int
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _tracing_active

        # Extract query from first argument
        query = ""
        if args:
            query = str(args[0])
        elif "query" in kwargs:
            query = str(kwargs["query"])
        elif "input" in kwargs:
            query = str(kwargs["input"])

        # Start trace
        _global_store.start_trace(query)
        _tracing_active = True

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            # Still record the trace even on failure
            _global_store.end_trace(f"Error: {str(e)}")
            _tracing_active = False
            raise
        finally:
            _tracing_active = False

        # End trace with output
        output = ""
        if isinstance(result, str):
            output = result
        elif isinstance(result, dict):
            output = result.get("answer", result.get("output", str(result)))
        else:
            output = str(result)

        _global_store.end_trace(output)
        return result

    # Attach trace access methods
    wrapper.get_traces = _global_store.get_traces
    wrapper.clear_traces = _global_store.clear
    wrapper.export_traces = _global_store.export
    wrapper.trace_count = property(lambda self: _global_store.trace_count)
    wrapper._store = _global_store

    return wrapper


def traced_tool(fn: Callable = None, *, name: str | None = None):
    """
    Decorator for individual tool functions.

    Records the tool call (name, args, output) when called inside a @trace_agent.
    If called outside a @trace_agent, behaves normally (no recording).

    Args:
        name: Override tool name (defaults to function name).

    Usage:
        @traced_tool
        def retrieval(query: str) -> str: ...

        @traced_tool(name="custom_search")
        def my_search(query: str) -> str: ...
    """
    # Handle both @traced_tool and @traced_tool(name="...") syntax
    if fn is None:
        # Called with arguments: @traced_tool(name="...")
        def decorator(func: Callable) -> Callable:
            return _wrap_tool(func, name or func.__name__)
        return decorator
    else:
        # Called without arguments: @traced_tool
        return _wrap_tool(fn, name or fn.__name__)


def _wrap_tool(fn: Callable, tool_name: str) -> Callable:
    """Internal: wrap a tool function with tracing."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _tracing_active

        if not _tracing_active:
            # Not inside a @trace_agent call, run normally
            return fn(*args, **kwargs)

        # Capture args
        call_args = {}
        if args:
            # Try to map positional args to param names
            import inspect
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(params):
                    call_args[params[i]] = _safe_serialize(arg)
        call_args.update({k: _safe_serialize(v) for k, v in kwargs.items()})

        # Execute and time it
        start = time.time()
        result = fn(*args, **kwargs)
        duration_ms = (time.time() - start) * 1000

        # Record the tool call
        _global_store.record_tool_call(
            tool_name=tool_name,
            args=call_args,
            output=result,
            duration_ms=duration_ms,
        )

        return result

    # Mark as traced tool for introspection
    wrapper._is_traced_tool = True
    wrapper._tool_name = tool_name

    return wrapper


def _safe_serialize(value: Any) -> Any:
    """Safely serialize a value for storage (avoid huge objects)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            return f"[list of {len(value)} items]"
        return [_safe_serialize(v) for v in value[:10]]
    if isinstance(value, dict):
        if len(value) > 10:
            return f"{{dict with {len(value)} keys}}"
        return {k: _safe_serialize(v) for k, v in list(value.items())[:10]}
    return str(value)[:200]


# =============================================================================
# Utility: Get traces from any store
# =============================================================================

def get_global_traces():
    """Get all traces captured by any @trace_agent decorated function."""
    return _global_store.get_traces()


def clear_global_traces():
    """Clear all globally captured traces."""
    _global_store.clear()
