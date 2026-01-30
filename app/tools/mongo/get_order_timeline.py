"""
Tool: get_order_timeline
Fetches order timeline from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime
from typing import List

from app.models.evidence import OrderEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_order_timeline",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_order_timeline(order_id: str, include: List[str]) -> OrderEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches order timeline from MongoDB
    - Returns structured order events with ToolResult
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_order_timeline",
        "params": {"order_id": order_id, "include": include}
    })
    
    try:
        # Mock implementation for hackathon
        timeline_data = {
            "order_id": order_id,
            "status": "delivered",
            "created_at": "2026-01-30T10:00:00Z",
            "events": [
                {"timestamp": "2026-01-30T10:00:00Z", "event": "order_placed", "status": "pending"},
                {"timestamp": "2026-01-30T10:05:00Z", "event": "restaurant_confirmed", "status": "confirmed"},
                {"timestamp": "2026-01-30T10:20:00Z", "event": "picked_up", "status": "in_transit"},
                {"timestamp": "2026-01-30T10:45:00Z", "event": "delivered", "status": "delivered"}
            ],
            "estimated_delivery": "2026-01-30T10:40:00Z",
            "actual_delivery": "2026-01-30T10:45:00Z",
            "delivery_delay_minutes": 5
        }
        
        result = OrderEvidenceEnvelope(
            source="mongo",
            entity_refs=[order_id],
            freshness=datetime.utcnow(),
            confidence=0.95,
            data=timeline_data,
            gaps=[],
            provenance={"query": "get_order_timeline", "latency_ms": 50},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=timeline_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_order_timeline",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_order_timeline",
            "error": str(e)
        })
        
        return OrderEvidenceEnvelope(
            source="mongo",
            entity_refs=[order_id],
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["order_timeline_unavailable"],
            provenance={"query": "get_order_timeline", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
