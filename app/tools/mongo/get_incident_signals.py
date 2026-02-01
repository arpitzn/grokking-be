"""
Tool: get_incident_signals
Fetches incident signals from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone
from typing import Type, Union

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import IncidentEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import string_to_mongo_id
from app.infra.mongo import get_mongodb_client

logger = logging.getLogger(__name__)


def safe_isoformat(value: Union[datetime, str, None]) -> Union[str, None]:
    """
    Safely convert datetime value to ISO format string.
    Handles both datetime objects and already-formatted strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value  # Already a string, return as-is
    return str(value)  # Fallback for other types

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_incident_signals",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_incident_signals(customer_id: str) -> IncidentEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches incident signals from MongoDB (support_tickets collection) for a customer
    - Returns relevant support tickets for the customer
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_incident_signals")
    print(f"  customer_id: {customer_id}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_incident_signals] Starting - customer_id={customer_id}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_incident_signals",
        "params": {"customer_id": customer_id}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        query_user_id = string_to_mongo_id(customer_id)
        logger.debug(f"[get_incident_signals] Query user_id (converted): {query_user_id}, type: {type(query_user_id).__name__}")
        
        # Query support_tickets collection by user_id (customer_id)
        query_filter = {"user_id": query_user_id}
        logger.debug(f"[get_incident_signals] MongoDB query filter: {query_filter}")
        
        tickets = await db.support_tickets.find(
            query_filter
        ).sort("timestamp", -1).limit(10).to_list(length=10)
        
        logger.info(f"[get_incident_signals] Found {len(tickets)} tickets")
        
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
            
            # Convert ObjectId fields to strings
            ticket_id_val = ticket.get("ticket_id")
            if isinstance(ticket_id_val, (ObjectId, Binary)):
                ticket_id_val = str(ticket_id_val)
            
            order_id_val = ticket.get("order_id")
            if isinstance(order_id_val, (ObjectId, Binary)):
                order_id_val = str(order_id_val)
            
            user_id_val = ticket.get("user_id")
            if isinstance(user_id_val, (ObjectId, Binary)):
                user_id_val = str(user_id_val)
            
            incident_data = {
                "ticket_id": ticket_id_val,
                "ticket_type": ticket.get("ticket_type"),
                "issue_type": ticket.get("issue_type"),
                "subtype": ticket.get("subtype", {}),
                "severity": severity_str,
                "order_id": order_id_val,
                "user_id": user_id_val,
                "timestamp": safe_isoformat(ticket.get("timestamp")),
                "status": ticket.get("status"),
                "resolution": ticket.get("resolution")
            }
            incidents.append(incident_data)
        
        signals_data = {
            "incidents": incidents,
            "total_count": len(incidents),
            "high_severity_count": high_severity_count
        }
        
        logger.info(f"[get_incident_signals] Success - total_count={signals_data.get('total_count')}, "
                   f"high_severity_count={signals_data.get('high_severity_count')}")
        if incidents:
            logger.debug(f"[get_incident_signals] Sample incidents: {incidents[:2]}")
        else:
            logger.debug(f"[get_incident_signals] No incidents found")
        
        result = IncidentEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.88 if incidents else 0.5,
            data=signals_data,
            gaps=[] if incidents else ["no_incidents_found"],
            provenance={"query": "get_incident_signals", "customer_id": customer_id, "latency_ms": 55},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=signals_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_incident_signals",
            "status": "success"
        })
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_incident_signals - SUCCESS")
        print(f"  total_count: {signals_data.get('total_count')}")
        print(f"  high_severity_count: {signals_data.get('high_severity_count')}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_incident_signals] Error - customer_id={customer_id}, error={str(e)}", exc_info=True)
        
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_incident_signals",
            "error": str(e)
        })
        
        return IncidentEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
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
    customer_id: str = Field(description="Customer ID to fetch incident signals for")


class GetIncidentSignalsTool(BaseTool):
    """LangChain tool wrapper for get_incident_signals"""
    name: str = "get_incident_signals"
    description: str = "Fetches incident signals from MongoDB (support_tickets collection) for a customer. Returns relevant support tickets for the customer."
    args_schema: Type[BaseModel] = GetIncidentSignalsInput
    
    async def _arun(self, customer_id: str) -> str:
        """Async execution - returns JSON string of IncidentEvidenceEnvelope"""
        result = await get_incident_signals(customer_id)
        return result.model_dump_json()
    
    def _run(self, customer_id: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
