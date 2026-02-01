"""
Tool: get_zone_ops_metrics
Fetches zone operations metrics from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Type

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import ToolResult, ToolStatus, ZoneEvidenceEnvelope
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import string_to_mongo_id
from app.infra.mongo import get_mongodb_client
from app.infra.demo_constants import DEMO_ZONE_ID, DEMO_TIME_WINDOW

logger = logging.getLogger(__name__)

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


async def get_zone_ops_metrics() -> ZoneEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches zone operations metrics from MongoDB (uses hardcoded demo zone ID)
    - Returns delivery performance, support ticket rates, and operational health
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    # Use hardcoded demo values
    zone_id = DEMO_ZONE_ID
    time_window = DEMO_TIME_WINDOW
    
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_zone_ops_metrics")
    print(f"  zone_id: {zone_id}")
    print(f"  time_window: {time_window}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_zone_ops_metrics] Starting - zone_id={zone_id}, time_window={time_window}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_zone_ops_metrics",
        "params": {"zone_id": zone_id, "time_window": time_window}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        query_zone_id = string_to_mongo_id(zone_id)
        logger.debug(f"[get_zone_ops_metrics] Query zone_id (converted): {query_zone_id}, type: {type(query_zone_id).__name__}")
        
        # Query zone_metrics_history collection - get latest 10 documents
        query_filter = {"zone_id": query_zone_id}
        logger.debug(f"[get_zone_ops_metrics] MongoDB query filter: {query_filter}")
        
        metrics_docs = await db.zone_metrics_history.find(
            query_filter,
            sort=[("timestamp", -1)],
            limit=10
        ).to_list(length=10)
        
        logger.info(f"[get_zone_ops_metrics] Found {len(metrics_docs)} documents")
        if metrics_docs:
            logger.debug(f"[get_zone_ops_metrics] Latest doc timestamp: {metrics_docs[0].get('timestamp')}")
            logger.debug(f"[get_zone_ops_metrics] Latest doc sample: {dict(list(metrics_docs[0].items())[:5])}")
        
        # Use the most recent document
        metrics_doc = metrics_docs[0] if metrics_docs else None
        
        if not metrics_doc:
            logger.warning(f"[get_zone_ops_metrics] No metrics found for zone_id={zone_id}")
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
        # Convert ObjectId fields to strings for JSON serialization
        zone_id_value = metrics_doc.get("zone_id")
        zone_id_str = str(zone_id_value) if isinstance(zone_id_value, (ObjectId, Binary)) else zone_id_value
        
        metrics_data = {
            "zone_id": zone_id_str,
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
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_zone_ops_metrics - SUCCESS")
        print(f"  zone_id: {metrics_data.get('zone_id')}")
        print(f"  total_orders: {metrics_data.get('total_orders')}")
        print(f"  on_time_delivery_rate: {metrics_data.get('on_time_delivery_rate')}")
        print(f"  support_ticket_count: {metrics_data.get('support_ticket_count')}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_zone_ops_metrics] Error - zone_id={zone_id}, error={str(e)}", exc_info=True)
        
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
    """No input needed - uses hardcoded demo values"""
    pass


class GetZoneOpsMetricsTool(BaseTool):
    """LangChain tool wrapper for get_zone_ops_metrics"""
    name: str = "get_zone_ops_metrics"
    description: str = "Fetches zone operations metrics from MongoDB (uses hardcoded DEMO_ZONE_ID). Returns delivery performance, support ticket rates, active drivers, and operational health indicators."
    args_schema: Type[BaseModel] = GetZoneOpsMetricsInput
    
    async def _arun(self) -> str:
        """Async execution - returns JSON string of ZoneEvidenceEnvelope"""
        result = await get_zone_ops_metrics()
        return result.model_dump_json()
    
    def _run(self) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
