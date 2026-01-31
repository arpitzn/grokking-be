"""
Tool: get_zone_ops_metrics
Fetches zone operations metrics from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import ToolResult, ToolStatus, ZoneEvidenceEnvelope
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import uuid_to_binary, is_uuid_string
from app.infra.mongo import get_mongodb_client

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_zone_ops_metrics",
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


async def get_zone_ops_metrics(zone_id: str, time_window: str) -> ZoneEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches zone operations metrics from MongoDB
    - Returns delivery performance, support ticket rates, and operational health
    
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
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Parse time window
        start_time, end_time = parse_time_window(time_window)
        
        # Convert UUID string to Binary UUID if needed
        query_zone_id = uuid_to_binary(zone_id) if is_uuid_string(zone_id) else zone_id
        
        # Query zone_metrics_history collection
        metrics_doc = await db.zone_metrics_history.find_one(
            {
                "zone_id": query_zone_id,
                "time_window": time_window,
                "timestamp": {"$gte": start_time, "$lte": end_time}
            },
            sort=[("timestamp", -1)]
        )
        
        if not metrics_doc:
            # Metrics not found - return empty with gap
            return ZoneEvidenceEnvelope(
                source="mongo",
                entity_refs=[zone_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["zone_metrics_unavailable"],
                provenance={"query": "get_zone_ops_metrics", "zone_id": zone_id, "time_window": time_window},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Zone metrics not found")
            )
        
        # Transform MongoDB document to tool output format
        metrics_data = {
            "zone_id": metrics_doc.get("zone_id"),
            "time_window": metrics_doc.get("time_window"),
            "total_orders": metrics_doc.get("total_orders"),
            "avg_delivery_time_minutes": metrics_doc.get("avg_delivery_time_minutes"),
            "on_time_delivery_rate": metrics_doc.get("on_time_delivery_rate"),
            "support_ticket_count": metrics_doc.get("support_ticket_count"),  # Changed from incident_count
            "support_ticket_rate": metrics_doc.get("support_ticket_rate"),  # Changed from incident_rate
            "active_drivers": metrics_doc.get("active_drivers"),
            "avg_restaurant_prep_time": metrics_doc.get("avg_restaurant_prep_time"),
            "weather_alert": metrics_doc.get("weather_alert", False),
            "traffic_alert": metrics_doc.get("traffic_alert", False)
        }
        
        result = ZoneEvidenceEnvelope(
            source="mongo",
            entity_refs=[zone_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.90,
            data=metrics_data,
            gaps=[],
            provenance={"query": "get_zone_ops_metrics", "zone_id": zone_id, "time_window": time_window, "latency_ms": 60},
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
    description: str = "Fetches zone operations metrics from MongoDB. Returns delivery performance, support ticket rates, active drivers, and operational health indicators."
    args_schema: Type[BaseModel] = GetZoneOpsMetricsInput
    
    async def _arun(self, zone_id: str, time_window: str) -> str:
        """Async execution - returns JSON string of ZoneEvidenceEnvelope"""
        result = await get_zone_ops_metrics(zone_id, time_window)
        return result.model_dump_json()
    
    def _run(self, zone_id: str, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
