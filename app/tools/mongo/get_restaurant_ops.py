"""
Tool: get_restaurant_ops
Fetches restaurant operations data from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime

from app.models.evidence import RestaurantEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_restaurant_ops",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_restaurant_ops(restaurant_id: str, time_window: str) -> RestaurantEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches restaurant operations data from MongoDB
    - Returns prep time, quality metrics, and operational status
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_restaurant_ops",
        "params": {"restaurant_id": restaurant_id, "time_window": time_window}
    })
    
    try:
        # Mock implementation for hackathon
        ops_data = {
            "restaurant_id": restaurant_id,
            "time_window": time_window,
            "avg_prep_time_minutes": 18,
            "on_time_rate": 0.92,
            "quality_rating": 4.3,
            "complaint_count": 3,
            "order_volume": 450,
            "is_open": True,
            "current_wait_time": 15
        }
        
        result = RestaurantEvidenceEnvelope(
            source="mongo",
            entity_refs=[restaurant_id],
            freshness=datetime.utcnow(),
            confidence=0.91,
            data=ops_data,
            gaps=[],
            provenance={"query": "get_restaurant_ops", "latency_ms": 40},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=ops_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_restaurant_ops",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_restaurant_ops",
            "error": str(e)
        })
        
        return RestaurantEvidenceEnvelope(
            source="mongo",
            entity_refs=[restaurant_id],
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["restaurant_ops_unavailable"],
            provenance={"query": "get_restaurant_ops", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
