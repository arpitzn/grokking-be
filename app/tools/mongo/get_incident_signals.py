"""
Tool: get_incident_signals
Fetches incident signals from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime
from typing import Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import IncidentEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_incident_signals",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_incident_signals(scope: Dict, time_window: str) -> IncidentEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches incident signals from MongoDB
    - Returns relevant incidents matching scope (order_id, customer_id, zone_id, restaurant_id)
    
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
        # Mock implementation for hackathon
        signals_data = {
            "scope": scope,
            "time_window": time_window,
            "incidents": [
                {
                    "incident_id": "inc_001",
                    "type": "delivery_delay",
                    "severity": "medium",
                    "order_id": scope.get("order_id"),
                    "timestamp": "2026-01-30T10:50:00Z",
                    "status": "resolved",
                    "resolution": "refund_issued"
                }
            ],
            "total_count": 1,
            "high_severity_count": 0
        }
        
        result = IncidentEvidenceEnvelope(
            source="mongo",
            entity_refs=[scope.get("order_id", scope.get("customer_id", ""))],
            freshness=datetime.utcnow(),
            confidence=0.88,
            data=signals_data,
            gaps=[],
            provenance={"query": "get_incident_signals", "latency_ms": 55},
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
            freshness=datetime.utcnow(),
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
    description: str = "Fetches incident signals from MongoDB. Returns relevant incidents matching scope (order_id, customer_id, zone_id, restaurant_id) within the specified time window."
    args_schema: Type[BaseModel] = GetIncidentSignalsInput
    
    async def _arun(self, scope: Dict, time_window: str) -> dict:
        """Async execution - returns dict representation of IncidentEvidenceEnvelope"""
        result = await get_incident_signals(scope, time_window)
        return result.dict()
    
    def _run(self, scope: Dict, time_window: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
