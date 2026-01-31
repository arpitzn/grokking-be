"""
Tool observability utility
Emits tool_call_started, tool_call_completed, tool_call_failed events
Streams to UI (SSE) and Langfuse (tracing)
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Global event queue for SSE streaming (will be populated by chat endpoint)
_event_queue = []


def emit_tool_event(event_type: str, payload: Dict[str, Any]) -> None:
    """
    Emits tool observability events.
    
    Event types:
    - tool_call_started: Before tool execution
    - tool_call_completed: After successful execution
    - tool_call_failed: After failed execution
    
    Payload includes:
    - tool_name: str
    - params: Dict (for started)
    - status: str (for completed)
    - error: str (for failed)
    
    Events are:
    1. Streamed to UI via SSE (for real-time visibility)
    2. Logged to Langfuse (for tracing and observability)
    """
    event = {
        "type": event_type,
        "payload": payload,
        "timestamp": None  # Will be set by streaming handler
    }
    
    # Add to event queue for SSE streaming
    _event_queue.append(event)
    
    # Log to Langfuse (via logger - Langfuse callback will pick this up)
    logger.info(f"Tool event: {event_type}", extra={
        "tool_event": True,
        "event_type": event_type,
        "payload": payload
    })


def get_pending_events() -> List[Dict[str, Any]]:
    """Get and clear pending events for SSE streaming"""
    global _event_queue
    events = _event_queue.copy()
    _event_queue.clear()
    return events


def stream_to_ui(event_type: str, payload: Dict[str, Any]) -> None:
    """Stream event to UI via SSE (called by emit_tool_event)"""
    # Implementation handled by emit_tool_event adding to queue
    # Chat endpoint will consume from queue
    pass


def log_to_langfuse(event_type: str, payload: Dict[str, Any]) -> None:
    """Log event to Langfuse for tracing (called by emit_tool_event)"""
    # Implementation handled by logger.info with extra metadata
    # Langfuse callback will pick up the metadata
    pass
