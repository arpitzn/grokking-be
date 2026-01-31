"""
Tool: get_restaurant_ops
Fetches restaurant operations data from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

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


# LangChain BaseTool wrapper
class GetRestaurantOpsInput(BaseModel):
    """Input schema for get_restaurant_ops tool"""
    restaurant_id: str = Field(description="Restaurant ID to fetch operations data for")
    time_window: str = Field(default="24h", description="Time window for operations data (e.g., '24h', '7d')")


class GetRestaurantOpsTool(BaseTool):
    """LangChain tool wrapper for get_restaurant_ops"""
    name: str = "get_restaurant_ops"
    description: str = "Fetches restaurant operations data from MongoDB. Returns prep time metrics, quality ratings, complaint counts, order volume, and operational status."
    args_schema: Type[BaseModel] = GetRestaurantOpsInput
    
    async def _arun(self, restaurant_id: str, time_window: str) -> dict:
        """Async execution - returns dict representation of RestaurantEvidenceEnvelope"""
        result = await get_restaurant_ops(restaurant_id, time_window)
        return result.dict()
    
    def _run(self, restaurant_id: str, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
