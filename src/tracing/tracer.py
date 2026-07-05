"""
Hybrid RAG - Tracing module
"""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class Trace:
    """Represents a trace of a query execution."""

    def __init__(
        self,
        trace_id: str,
        query: str,
        timestamp: str,
    ):
        self.trace_id = trace_id
        self.query = query
        self.timestamp = timestamp
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Add a step to the trace."""
        self.steps.append(
            {
                "name": name,
                "input": input_data,
                "output": output_data,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "timestamp": self.timestamp,
            "steps": self.steps,
        }

    def to_json(self) -> str:
        """Convert trace to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BaseTracer(ABC):
    """Base class for tracers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def start_trace(self, query: str) -> Trace:
        """Start a new trace."""
        pass

    @abstractmethod
    async def end_trace(self, trace: Trace) -> None:
        """End and store a trace."""
        pass


class LoggingTracer(BaseTracer):
    """Tracer that logs to console/file."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = config.get("enabled", True)

    async def start_trace(self, query: str) -> Trace:
        """Start a new trace."""
        if not self.enabled:
            return Trace(str(uuid.uuid4()), query, datetime.utcnow().isoformat() + "Z")

        trace_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            query=query,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        return trace

    async def end_trace(self, trace: Trace) -> None:
        """End and store a trace."""
        if not self.enabled:
            return

        print(f"Trace completed: {trace.trace_id}")
        print(trace.to_json())
