"""
Tool: get_order_timeline
Fetches order timeline from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

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
            freshness=datetime.now(timezone.utc),
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
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["order_timeline_unavailable"],
            provenance={"query": "get_order_timeline", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetOrderTimelineInput(BaseModel):
    """Input schema for get_order_timeline tool"""
    order_id: str = Field(description="Order ID to fetch timeline for")
    include: List[str] = Field(
        default=["events", "status", "timestamps"],
        description="Fields to include: events, status, timestamps"
    )


class GetOrderTimelineTool(BaseTool):
    """LangChain tool wrapper for get_order_timeline"""
    name: str = "get_order_timeline"
    description: str = "Fetches order timeline from MongoDB with events, status, and timestamps. Returns order events, delivery times, and status information."
    args_schema: Type[BaseModel] = GetOrderTimelineInput
    
    async def _arun(self, order_id: str, include: List[str]) -> str:
        """Async execution - returns JSON string of OrderEvidenceEnvelope"""
        result = await get_order_timeline(order_id, include)
        return result.model_dump_json()
    
    def _run(self, order_id: str, include: List[str]) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
