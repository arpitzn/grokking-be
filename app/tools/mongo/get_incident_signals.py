"""
Tool: get_incident_signals
Fetches incident signals from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import IncidentEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import uuid_to_binary, is_uuid_string
from app.infra.mongo import get_mongodb_client

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_incident_signals",
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


def build_incident_query(scope: Dict, start_time: datetime, end_time: datetime) -> Dict:
    """Build MongoDB query from scope dictionary"""
    query = {
        "timestamp": {
            "$gte": start_time,
            "$lte": end_time
        }
    }
    
    if scope.get("order_id"):
        order_id = scope["order_id"]
        query["order_id"] = uuid_to_binary(order_id) if is_uuid_string(order_id) else order_id
    if scope.get("customer_id"):
        customer_id = scope["customer_id"]
        query["user_id"] = uuid_to_binary(customer_id) if is_uuid_string(customer_id) else customer_id
    if scope.get("restaurant_id"):
        restaurant_id = scope["restaurant_id"]
        query["restaurant_id"] = uuid_to_binary(restaurant_id) if is_uuid_string(restaurant_id) else restaurant_id
    if scope.get("zone_id"):
        zone_id = scope["zone_id"]
        query["affected_zones"] = uuid_to_binary(zone_id) if is_uuid_string(zone_id) else zone_id
    if scope.get("scope"):
        query["scope"] = scope["scope"]
    
    return query


async def get_incident_signals(scope: Dict, time_window: str) -> IncidentEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches incident signals from MongoDB (support_tickets collection)
    - Returns relevant support tickets matching scope (order_id, customer_id, zone_id, restaurant_id)
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_incident_signals",
        "params": {"scope": scope, "time_window": time_window}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Parse time window
        start_time, end_time = parse_time_window(time_window)
        
        # Build query from scope
        query = build_incident_query(scope, start_time, end_time)
        
        # Query support_tickets collection (replaces incidents)
        cursor = db.support_tickets.find(query).sort("timestamp", -1)
        tickets = await cursor.to_list(length=100)  # Limit to 100 tickets
        
        # Transform tickets to incident-like structure
        incidents = []
        high_severity_count = 0
        
        for ticket in tickets:
            # Map severity: 1=Critical, 2=High, 3=Medium, 4=Low
            severity_str = "low"
            if ticket.get("severity") == 1:
                severity_str = "critical"
                high_severity_count += 1
            elif ticket.get("severity") == 2:
                severity_str = "high"
                high_severity_count += 1
            elif ticket.get("severity") == 3:
                severity_str = "medium"
            
            incident_data = {
                "ticket_id": ticket.get("ticket_id"),
                "ticket_type": ticket.get("ticket_type"),
                "issue_type": ticket.get("issue_type"),
                "subtype": ticket.get("subtype", {}),
                "severity": severity_str,
                "order_id": ticket.get("order_id"),
                "timestamp": ticket.get("timestamp").isoformat() if ticket.get("timestamp") else None,
                "status": ticket.get("status"),
                "resolution": ticket.get("resolution")
            }
            incidents.append(incident_data)
        
        signals_data = {
            "scope": scope,
            "time_window": time_window,
            "incidents": incidents,
            "total_count": len(incidents),
            "high_severity_count": high_severity_count
        }
        
        result = IncidentEvidenceEnvelope(
            source="mongo",
            entity_refs=[scope.get("order_id", scope.get("customer_id", ""))],
            freshness=datetime.now(timezone.utc),
            confidence=0.88 if incidents else 0.5,
            data=signals_data,
            gaps=[] if incidents else ["no_incidents_found"],
            provenance={"query": "get_incident_signals", "scope": scope, "time_window": time_window, "latency_ms": 55},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=signals_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_incident_signals",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_incident_signals",
            "error": str(e)
        })
        
        return IncidentEvidenceEnvelope(
            source="mongo",
            entity_refs=[],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["incident_signals_unavailable"],
            provenance={"query": "get_incident_signals", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetIncidentSignalsInput(BaseModel):
    """Input schema for get_incident_signals tool"""
    scope: Dict = Field(description="Scope dictionary with order_id, customer_id, zone_id, or restaurant_id")
    time_window: str = Field(default="24h", description="Time window for incident search (e.g., '24h', '7d')")


class GetIncidentSignalsTool(BaseTool):
    """LangChain tool wrapper for get_incident_signals"""
    name: str = "get_incident_signals"
    description: str = "Fetches incident signals from MongoDB (support_tickets collection). Returns relevant support tickets matching scope (order_id, customer_id, zone_id, restaurant_id) within the specified time window."
    args_schema: Type[BaseModel] = GetIncidentSignalsInput
    
    async def _arun(self, scope: Dict, time_window: str) -> str:
        """Async execution - returns JSON string of IncidentEvidenceEnvelope"""
        result = await get_incident_signals(scope, time_window)
        return result.model_dump_json()
    
    def _run(self, scope: Dict, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
