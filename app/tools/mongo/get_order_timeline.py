"""
Tool: get_order_timeline
Fetches order timeline from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import List, Type

from bson import Binary
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import OrderEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import uuid_to_binary, is_uuid_string, binary_to_uuid
from app.infra.mongo import get_mongodb_client

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
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert UUID string to Binary UUID if needed
        query_order_id = uuid_to_binary(order_id) if is_uuid_string(order_id) else order_id
        
        # Query order from MongoDB (try order_id field first, then _id)
        order_doc = await db.orders.find_one({"order_id": query_order_id})
        if not order_doc:
            # Try querying by _id directly
            order_doc = await db.orders.find_one({"_id": query_order_id})
        
        if not order_doc:
            # Order not found - return empty with gap
            return OrderEvidenceEnvelope(
                source="mongo",
                entity_refs=[order_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["order_timeline_unavailable"],
                provenance={"query": "get_order_timeline", "order_id": order_id},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Order not found")
            )
        
        # Transform MongoDB document to tool output format
        # Convert Binary UUIDs to strings for output
        order_id_val = order_doc.get("order_id")
        if isinstance(order_id_val, Binary):
            order_id_val = binary_to_uuid(order_id_val)
        
        user_id_val = order_doc.get("user_id")
        if isinstance(user_id_val, Binary):
            user_id_val = binary_to_uuid(user_id_val)
        
        timeline_data = {
            "order_id": order_id_val,
            "user_id": user_id_val,
            "status": order_doc.get("status"),
            "created_at": order_doc.get("created_at").isoformat() if order_doc.get("created_at") else None,
            "events": [
                {
                    "timestamp": event.get("timestamp").isoformat() if event.get("timestamp") else None,
                    "event": event.get("event"),
                    "status": event.get("status")
                }
                for event in order_doc.get("events", [])
            ],
            "estimated_delivery": order_doc.get("estimated_delivery").isoformat() if order_doc.get("estimated_delivery") else None,
            "actual_delivery": order_doc.get("actual_delivery").isoformat() if order_doc.get("actual_delivery") else None,
            "delivery_delay_minutes": order_doc.get("delivery_delay_minutes")
        }
        
        # Include refund and payment if present
        if order_doc.get("refund"):
            timeline_data["refund"] = order_doc.get("refund")
        if order_doc.get("payment"):
            timeline_data["payment"] = order_doc.get("payment")
        if order_doc.get("refund_status"):
            timeline_data["refund_status"] = order_doc.get("refund_status")
        
        result = OrderEvidenceEnvelope(
            source="mongo",
            entity_refs=[order_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.95 if order_doc.get("events") else 0.7,  # Lower confidence if no events
            data=timeline_data,
            gaps=[] if order_doc.get("events") else ["events_missing"],
            provenance={"query": "get_order_timeline", "order_id": order_id, "latency_ms": 50},
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
