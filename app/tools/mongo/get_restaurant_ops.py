"""
Tool: get_restaurant_ops
Fetches restaurant operations data from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone, timedelta
from typing import Type

from bson import Binary
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import RestaurantEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import uuid_to_binary, is_uuid_string, binary_to_uuid
from app.infra.mongo import get_mongodb_client

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_restaurant_ops",
    criticality=ToolCriticality.DECISION_CRITICAL
)


def parse_time_window(time_window: str) -> tuple[datetime, datetime]:
    """Parse time_window string to start/end datetime"""
    now = datetime.now(timezone.utc)
    
    if time_window.endswith("h"):
        hours = int(time_window[:-1])
        start_time = now - timedelta(hours=hours)
    elif time_window.endswith("d"):
        days = int(time_window[:-1])
        start_time = now - timedelta(days=days)
    else:
        # Default to 24h
        start_time = now - timedelta(hours=24)
    
    return start_time, now


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
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Parse time window
        start_time, end_time = parse_time_window(time_window)
        
        # Convert UUID string to Binary UUID if needed
        query_restaurant_id = uuid_to_binary(restaurant_id) if is_uuid_string(restaurant_id) else restaurant_id
        
        # Query restaurant_metrics_history collection
        metrics_doc = await db.restaurant_metrics_history.find_one(
            {
                "restaurant_id": query_restaurant_id,
                "time_window": time_window,
                "timestamp": {"$gte": start_time, "$lte": end_time}
            },
            sort=[("timestamp", -1)]
        )
        
        # Also get current status from restaurants collection (query by _id)
        restaurant_doc = await db.restaurants.find_one({"_id": query_restaurant_id})
        
        if not metrics_doc:
            # Metrics not found - return empty with gap
            return RestaurantEvidenceEnvelope(
                source="mongo",
                entity_refs=[restaurant_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["restaurant_metrics_unavailable"],
                provenance={"query": "get_restaurant_ops", "restaurant_id": restaurant_id, "time_window": time_window},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Restaurant metrics not found")
            )
        
        # Transform MongoDB document to tool output format
        ops_data = {
            "restaurant_id": metrics_doc.get("restaurant_id"),
            "time_window": metrics_doc.get("time_window"),
            "avg_prep_time_minutes": metrics_doc.get("avg_prep_time_minutes"),
            "on_time_rate": metrics_doc.get("on_time_rate"),
            "quality_rating": metrics_doc.get("quality_rating"),
            "support_ticket_count": metrics_doc.get("support_ticket_count"),  # Changed from complaint_count
            "order_volume": metrics_doc.get("order_volume"),
            "is_open": restaurant_doc.get("is_open", False) if restaurant_doc else metrics_doc.get("is_open", False),
            "current_wait_time": metrics_doc.get("current_wait_time")
        }
        
        result = RestaurantEvidenceEnvelope(
            source="mongo",
            entity_refs=[restaurant_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.91,
            data=ops_data,
            gaps=[],
            provenance={"query": "get_restaurant_ops", "restaurant_id": restaurant_id, "time_window": time_window, "latency_ms": 40},
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
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["restaurant_ops_unavailable"],
            provenance={"query": "get_restaurant_ops", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetRestaurantOpsInput(BaseModel):
    """Input schema for get_restaurant_ops tool"""
    restaurant_id: str = Field(description="Restaurant ID to fetch operations data for")
    time_window: str = Field(default="24h", description="Time window for operations data (e.g., '24h', '7d')")


class GetRestaurantOpsTool(BaseTool):
    """LangChain tool wrapper for get_restaurant_ops"""
    name: str = "get_restaurant_ops"
    description: str = "Fetches restaurant operations data from MongoDB. Returns prep time metrics, quality ratings, support ticket counts, order volume, and operational status."
    args_schema: Type[BaseModel] = GetRestaurantOpsInput
    
    async def _arun(self, restaurant_id: str, time_window: str) -> str:
        """Async execution - returns JSON string of RestaurantEvidenceEnvelope"""
        result = await get_restaurant_ops(restaurant_id, time_window)
        return result.model_dump_json()
    
    def _run(self, restaurant_id: str, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
