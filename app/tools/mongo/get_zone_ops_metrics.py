"""
Tool: get_zone_ops_metrics
Fetches zone operations metrics from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import ToolResult, ToolStatus, ZoneEvidenceEnvelope
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_zone_ops_metrics",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_zone_ops_metrics(zone_id: str, time_window: str) -> ZoneEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches zone operations metrics from MongoDB
    - Returns delivery performance, incident rates, and operational health
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_zone_ops_metrics",
        "params": {"zone_id": zone_id, "time_window": time_window}
    })
    
    try:
        # Mock implementation for hackathon
        metrics_data = {
            "zone_id": zone_id,
            "time_window": time_window,
            "total_orders": 1250,
            "avg_delivery_time_minutes": 32,
            "on_time_delivery_rate": 0.87,
            "incident_count": 15,
            "incident_rate": 0.012,
            "active_drivers": 45,
            "avg_restaurant_prep_time": 18,
            "weather_alert": False,
            "traffic_alert": True
        }
        
        result = ZoneEvidenceEnvelope(
            source="mongo",
            entity_refs=[zone_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.90,
            data=metrics_data,
            gaps=[],
            provenance={"query": "get_zone_ops_metrics", "latency_ms": 60},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=metrics_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_zone_ops_metrics",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_zone_ops_metrics",
            "error": str(e)
        })
        
        return ZoneEvidenceEnvelope(
            source="mongo",
            entity_refs=[zone_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["zone_metrics_unavailable"],
            provenance={"query": "get_zone_ops_metrics", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetZoneOpsMetricsInput(BaseModel):
    """Input schema for get_zone_ops_metrics tool"""
    zone_id: str = Field(description="Zone ID to fetch operations metrics for")
    time_window: str = Field(default="24h", description="Time window for metrics (e.g., '24h', '7d')")


class GetZoneOpsMetricsTool(BaseTool):
    """LangChain tool wrapper for get_zone_ops_metrics"""
    name: str = "get_zone_ops_metrics"
    description: str = "Fetches zone operations metrics from MongoDB. Returns delivery performance, incident rates, active drivers, and operational health indicators."
    args_schema: Type[BaseModel] = GetZoneOpsMetricsInput
    
    async def _arun(self, zone_id: str, time_window: str) -> str:
        """Async execution - returns JSON string of ZoneEvidenceEnvelope"""
        result = await get_zone_ops_metrics(zone_id, time_window)
        return result.model_dump_json()
    
    def _run(self, zone_id: str, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
