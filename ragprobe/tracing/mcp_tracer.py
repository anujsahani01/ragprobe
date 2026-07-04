"""
MCP Tracer
==========
Captures tool call traces from MCP servers automatically.

When an LLM calls MCP tools directly (via protocol), there's no Python function
to decorate with @trace_agent. The MCPTracer wraps the MCP server and intercepts
all incoming tool calls at the protocol level.

Usage:
    from ragprobe.tracing import MCPTracer

    # Wrap your MCP server
    server = create_mcp_server()
    tracer = MCPTracer(server)

    # Run the server normally — tracer captures all tool calls
    # (In real usage, the LLM client calls tools via MCP protocol)

    # Simulate tool calls for evaluation
    tracer.record_call(query="How does auth work?", tool_name="retrieval", output="...")
    tracer.record_call(query="How does auth work?", tool_name="process_query", output="The auth...")

    # End the current interaction
    tracer.end_interaction(final_output="The auth middleware validates JWT...")

    # Get traces for evaluation
    traces = tracer.get_traces()

    # Evaluate
    from ragprobe import AgentEvaluator
    evaluator = AgentEvaluator.from_mcp_server(server)
    results = evaluator.evaluate_batch(traces)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from pathlib import Path
import json

from ragprobe.evaluator.agent_eval import AgentTrace, ToolDefinition


@dataclass
class MCPToolCall:
    """A single MCP tool call record."""
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    timestamp: str = ""
    duration_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MCPInteraction:
    """
    A full interaction — one user query that may trigger multiple tool calls.
    Represents the complete journey from query to final answer.
    """
    query: str
    tool_calls: list[MCPToolCall] = field(default_factory=list)
    final_output: str = ""
    start_time: str = ""
    end_time: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()

    def to_agent_trace(self) -> AgentTrace:
        """Convert this interaction to an AgentTrace for evaluation."""
        return AgentTrace(
            query=self.query,
            tools_called=[tc.tool_name for tc in self.tool_calls],
            output=self.final_output,
            tool_args=[tc.args for tc in self.tool_calls],
            tool_outputs=[tc.output for tc in self.tool_calls],
        )


class MCPTracer:
    """
    Trace capture for MCP servers.

    Wraps around an MCP server and records all tool calls as traces.
    Works in two modes:

    Mode 1: Manual recording (for testing/simulation)
        tracer.start_interaction("user query")
        tracer.record_call("tool_name", args={...}, output="...")
        tracer.end_interaction("final answer")

    Mode 2: Middleware (intercepts calls automatically)
        tracer = MCPTracer(server, auto_intercept=True)
        # All tool calls are recorded automatically

    Args:
        server: The FastMCP server instance (optional, used for tool extraction).
        auto_intercept: If True, monkey-patches server tools to record calls.
    """

    def __init__(self, server: Any = None, auto_intercept: bool = False, persist: str | None = None):
        self._server = server
        self._lock = threading.Lock()
        self._interactions: list[MCPInteraction] = []
        self._active: MCPInteraction | None = None
        self._tools: list[ToolDefinition] = []
        self._persist_path = persist  # If set, append traces to this file after every interaction

        # Extract tools if server provided
        if server:
            self._extract_tools(server)

        # Auto-intercept mode: wrap server tools
        if auto_intercept and server:
            self._intercept_tools(server)

    def _extract_tools(self, server: Any) -> None:
        """Extract tool definitions from the MCP server."""
        from ragprobe.evaluator.agent_eval import AgentEvaluator
        try:
            self._tools = AgentEvaluator.extract_tools_from_mcp(server)
        except ValueError:
            self._tools = []

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get extracted tool definitions."""
        return self._tools

    # =========================================================================
    # Mode 1: Manual Recording
    # =========================================================================

    def start_interaction(self, query: str) -> None:
        """
        Start recording a new user interaction.

        Call this when a user sends a query to your MCP agent.
        """
        with self._lock:
            # Finalize previous interaction if not closed
            if self._active:
                self._finalize_active("")
            self._active = MCPInteraction(query=query)

    def record_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        output: Any = None,
        duration_ms: float = 0.0,
        query: str | None = None,
    ) -> None:
        """
        Record a tool call within the current interaction.

        If no interaction is active and query is provided, starts one automatically.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments passed to the tool.
            output: What the tool returned.
            duration_ms: How long the tool call took.
            query: If no active interaction, start one with this query.
        """
        with self._lock:
            # Auto-start interaction if needed
            if self._active is None:
                if query:
                    self._active = MCPInteraction(query=query)
                else:
                    # No active interaction and no query — create unnamed
                    self._active = MCPInteraction(query="[untracked query]")

            self._active.tool_calls.append(MCPToolCall(
                tool_name=tool_name,
                args=args or {},
                output=str(output)[:500] if output else "",
                duration_ms=duration_ms,
            ))

    def end_interaction(self, final_output: str) -> None:
        """
        End the current interaction and store the trace.

        Call this when the agent produces its final response to the user.

        Args:
            final_output: The agent's final answer to the user.
        """
        with self._lock:
            self._finalize_active(final_output)

    def _finalize_active(self, final_output: str) -> None:
        """Convert active interaction to completed trace."""
        if self._active is None:
            return
        self._active.final_output = final_output
        self._active.end_time = datetime.now().isoformat()
        self._interactions.append(self._active)

        # Persist immediately if path configured
        if self._persist_path:
            self._append_to_persist(self._active)

        self._active = None

    def _append_to_persist(self, interaction: MCPInteraction) -> None:
        """Append a single interaction to the persist file (JSONL — one JSON per line)."""
        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        trace_data = {
            "query": interaction.query,
            "tool_calls": [
                {"tool_name": tc.tool_name, "args": tc.args, "output": tc.output, "duration_ms": tc.duration_ms}
                for tc in interaction.tool_calls
            ],
            "final_output": interaction.final_output,
            "start_time": interaction.start_time,
            "end_time": interaction.end_time,
        }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data) + "\n")

    # =========================================================================
    # Mode 2: Auto-Intercept (monkey-patch MCP tools)
    # =========================================================================

    def _intercept_tools(self, server: Any) -> None:
        """
        Intercept tool calls on a FastMCP 3.x server.

        FastMCP 3.x uses async call_tool(). We wrap it to record calls.
        """
        import asyncio

        if not hasattr(server, 'call_tool'):
            return

        original_call_tool = server.call_tool

        async def intercepted_call_tool(name: str, arguments: dict | None = None, **kwargs):
            import time as _time
            start = _time.time()
            result = await original_call_tool(name, arguments=arguments, **kwargs)
            duration = (_time.time() - start) * 1000

            self.record_call(
                tool_name=name,
                args=arguments or {},
                output=str(result)[:500] if result else "",
                duration_ms=duration,
            )
            return result

        server.call_tool = intercepted_call_tool

    # =========================================================================
    # Output
    # =========================================================================

    def get_traces(self) -> list[AgentTrace]:
        """
        Get all completed interactions as AgentTrace objects.

        Ready to pass directly to AgentEvaluator.evaluate_batch().
        """
        with self._lock:
            return [interaction.to_agent_trace() for interaction in self._interactions]

    def get_interactions(self) -> list[MCPInteraction]:
        """Get raw interaction objects (more detail than traces)."""
        with self._lock:
            return self._interactions.copy()

    @property
    def trace_count(self) -> int:
        return len(self._interactions)

    def clear(self) -> None:
        """Clear all recorded interactions."""
        with self._lock:
            self._interactions.clear()
            self._active = None

    def export(self, path: str | Path) -> None:
        """Export all traces to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exported_at": datetime.now().isoformat(),
            "trace_count": self.trace_count,
            "tools_available": [t.to_dict() for t in self._tools],
            "interactions": [
                {
                    "query": i.query,
                    "tool_calls": [
                        {"tool_name": tc.tool_name, "args": tc.args, "output": tc.output, "duration_ms": tc.duration_ms}
                        for tc in i.tool_calls
                    ],
                    "final_output": i.final_output,
                    "start_time": i.start_time,
                    "end_time": i.end_time,
                }
                for i in self._interactions
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_traces_from_file(cls, path: str | Path) -> list[AgentTrace]:
        """
        Load traces from a JSONL persist file (written by persist= parameter).

        Each line is one JSON interaction. Returns AgentTrace objects ready
        for AgentEvaluator.evaluate_batch().

        Usage:
            traces = MCPTracer.load_traces_from_file("traces/session.jsonl")
            results = evaluator.evaluate_batch(traces)
        """
        path = Path(path)
        if not path.exists():
            return []

        traces = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    traces.append(AgentTrace(
                        query=data.get("query", ""),
                        tools_called=[tc["tool_name"] for tc in data.get("tool_calls", [])],
                        output=data.get("final_output", ""),
                        tool_args=[tc.get("args", {}) for tc in data.get("tool_calls", [])],
                        tool_outputs=[tc.get("output", "") for tc in data.get("tool_calls", [])],
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

        return traces