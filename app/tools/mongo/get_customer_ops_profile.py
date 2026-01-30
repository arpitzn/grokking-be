"""
Tool: get_customer_ops_profile
Fetches customer operations profile from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime

from app.models.evidence import CustomerEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_customer_ops_profile",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_customer_ops_profile(customer_id: str) -> CustomerEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches customer operations profile from MongoDB
    - Returns customer history, preferences, and operational metrics
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_customer_ops_profile",
        "params": {"customer_id": customer_id}
    })
    
    try:
        # Mock implementation for hackathon
        profile_data = {
            "customer_id": customer_id,
            "total_orders": 45,
            "lifetime_value": 1250.50,
            "avg_order_value": 27.79,
            "refund_count": 2,
            "refund_rate": 0.044,
            "last_order_date": "2026-01-29T18:30:00Z",
            "preferred_cuisines": ["Italian", "Chinese", "Mexican"],
            "avg_rating_given": 4.5,
            "complaint_count": 1,
            "vip_status": False
        }
        
        result = CustomerEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
            freshness=datetime.utcnow(),
            confidence=0.92,
            data=profile_data,
            gaps=[],
            provenance={"query": "get_customer_ops_profile", "latency_ms": 45},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=profile_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_customer_ops_profile",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_customer_ops_profile",
            "error": str(e)
        })
        
        return CustomerEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["customer_profile_unavailable"],
            provenance={"query": "get_customer_ops_profile", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
