"""
Trace Store
===========
Thread-safe storage for captured agent traces.
Used internally by the @trace_agent decorator to record tool calls.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ragprobe.evaluator.agent_eval import AgentTrace


@dataclass
class ToolCallRecord:
    """A single tool call captured during agent execution."""
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    timestamp: str = ""
    duration_ms: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ActiveTrace:
    """A trace currently being recorded (in-progress agent call)."""
    query: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    output: str = ""
    start_time: float = 0.0


class TraceStore:
    """
    Thread-safe store for agent traces.

    Manages the lifecycle of trace recording:
    - start_trace(query) — begin recording for a new query
    - record_tool_call(name, args, output) — add a tool call to current trace
    - end_trace(output) — finalize and store the trace
    - get_traces() — retrieve all completed traces
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._completed_traces: list[AgentTrace] = []
        self._active: ActiveTrace | None = None

    def start_trace(self, query: str) -> None:
        """Start recording a new trace for a query."""
        import time
        with self._lock:
            # If there's an active trace that wasn't closed, close it
            if self._active is not None:
                self._finalize_active("")

            self._active = ActiveTrace(query=query, start_time=time.time())

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        output: Any = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool call within the current active trace."""
        with self._lock:
            if self._active is None:
                return  # No active trace, skip silently

            self._active.tool_calls.append(ToolCallRecord(
                tool_name=tool_name,
                args=args or {},
                output=str(output)[:500] if output else "",
                duration_ms=duration_ms,
            ))

    def end_trace(self, output: str) -> None:
        """Finalize the current trace and store it."""
        with self._lock:
            self._finalize_active(output)

    def _finalize_active(self, output: str) -> None:
        """Internal: convert active trace to AgentTrace and store."""
        if self._active is None:
            return

        trace = AgentTrace(
            query=self._active.query,
            tools_called=[tc.tool_name for tc in self._active.tool_calls],
            output=output,
            tool_args=[tc.args for tc in self._active.tool_calls],
            tool_outputs=[tc.output for tc in self._active.tool_calls],
        )
        self._completed_traces.append(trace)
        self._active = None

    def get_traces(self) -> list[AgentTrace]:
        """Get all completed traces."""
        with self._lock:
            return self._completed_traces.copy()

    def clear(self) -> None:
        """Clear all stored traces."""
        with self._lock:
            self._completed_traces.clear()
            self._active = None

    @property
    def trace_count(self) -> int:
        """Number of completed traces."""
        return len(self._completed_traces)

    def export(self, path: str | Path) -> None:
        """Export all traces to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exported_at": datetime.now().isoformat(),
            "trace_count": self.trace_count,
            "traces": [t.to_dict() for t in self._completed_traces],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "TraceStore":
        """Load traces from a previously exported JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls()
        for t in data.get("traces", []):
            store._completed_traces.append(AgentTrace(
                query=t["query"],
                tools_called=t["tools_called"],
                output=t["output"],
                tool_args=t.get("tool_args", []),
                tool_outputs=t.get("tool_outputs", []),
                metadata=t.get("metadata", {}),
            ))
        return store
