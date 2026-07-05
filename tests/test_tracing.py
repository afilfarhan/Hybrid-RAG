"""
Tests for Hybrid RAG tracing
"""

import pytest
from src.tracing.tracer import Trace, LoggingTracer


def test_trace_creation():
    """Test trace creation."""
    trace = Trace(
        trace_id="test_trace_id",
        query="test query",
        timestamp="2024-01-01T00:00:00Z",
    )
    
    assert trace.trace_id == "test_trace_id"
    assert trace.query == "test query"
    assert len(trace.steps) == 0


def test_trace_add_step():
    """Test adding steps to trace."""
    trace = Trace(
        trace_id="test_trace_id",
        query="test query",
        timestamp="2024-01-01T00:00:00Z",
    )
    
    trace.add_step(
        name="retrieval",
        input_data={"query": "test"},
        output_data={"chunks": 5},
        duration_ms=100.0,
    )
    
    assert len(trace.steps) == 1
    assert trace.steps[0]["name"] == "retrieval"


def test_trace_to_dict():
    """Test converting trace to dictionary."""
    trace = Trace(
        trace_id="test_trace_id",
        query="test query",
        timestamp="2024-01-01T00:00:00Z",
    )
    
    trace.add_step(name="test", input_data={"a": 1})
    
    result = trace.to_dict()
    
    assert "trace_id" in result
    assert "query" in result
    assert "steps" in result
    assert len(result["steps"]) == 1


def test_trace_to_json():
    """Test converting trace to JSON."""
    trace = Trace(
        trace_id="test_trace_id",
        query="test query",
        timestamp="2024-01-01T00:00:00Z",
    )
    
    trace.add_step(name="test")
    
    json_str = trace.to_json()
    
    assert "test_trace_id" in json_str
    assert "test query" in json_str


@pytest.mark.asyncio
async def test_logging_tracer():
    """Test logging tracer."""
    tracer = LoggingTracer({"enabled": True})
    
    trace = await tracer.start_trace("test query")
    assert trace.query == "test query"
    
    # End trace just returns (no-op in logging tracer)
    await tracer.end_trace(trace)
