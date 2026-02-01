"""
Tool: get_case_context
Fetches aggregated case context from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import CaseEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.infra.mongo import get_mongodb_client

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_case_context",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_case_context(case_id: str) -> CaseEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches aggregated case context from MongoDB
    - Returns consolidated view of case history and related data
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "get_case_context",
        "params": {"case_id": case_id}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert UUID string to Binary UUID if needed
        query_case_id = uuid_to_binary(case_id) if is_uuid_string(case_id) else case_id
        
        # Query support_tickets collection (replaces cases)
        # Try ticket_id first, then _id, then conversation_id
        ticket_doc = await db.support_tickets.find_one({"ticket_id": query_case_id})
        if not ticket_doc:
            ticket_doc = await db.support_tickets.find_one({"_id": query_case_id})
        if not ticket_doc:
            ticket_doc = await db.support_tickets.find_one({"conversation_id": case_id})
        
        if not ticket_doc:
            # Ticket not found - return empty with gap
            return CaseEvidenceEnvelope(
                source="mongo",
                entity_refs=[case_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["case_context_unavailable"],
                provenance={"query": "get_case_context", "case_id": case_id},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Support ticket not found")
            )
        
        # Transform MongoDB document to case-like structure
        ticket_id_val = ticket_doc.get("ticket_id")
        if isinstance(ticket_id_val, Binary):
            ticket_id_val = binary_to_uuid(ticket_id_val)
        
        context_data = {
            "ticket_id": ticket_id_val,
            "case_id": ticket_doc.get("ticket_id"),  # Keep case_id for backward compatibility
            "conversation_id": ticket_doc.get("conversation_id"),
            "user_id": ticket_doc.get("user_id"),
            "ticket_type": ticket_doc.get("ticket_type"),
            "issue_type": ticket_doc.get("issue_type"),
            "subtype": ticket_doc.get("subtype", {}),
            "severity": ticket_doc.get("severity"),
            "created_at": ticket_doc.get("created_at").isoformat() if ticket_doc.get("created_at") else None,
            "status": ticket_doc.get("status"),
            "related_orders": ticket_doc.get("related_orders", []),
            "related_tickets": ticket_doc.get("related_tickets", []),  # Changed from related_cases
            "agent_notes": ticket_doc.get("agent_notes", []),
            "resolution_history": ticket_doc.get("resolution_history", []),
            "resolution": ticket_doc.get("resolution")
        }
        
        result = CaseEvidenceEnvelope(
            source="mongo",
            entity_refs=[case_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.93,
            data=context_data,
            gaps=[],
            provenance={"query": "get_case_context", "case_id": case_id, "latency_ms": 35},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=context_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_case_context",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_case_context",
            "error": str(e)
        })
        
        return CaseEvidenceEnvelope(
            source="mongo",
            entity_refs=[case_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["case_context_unavailable"],
            provenance={"query": "get_case_context", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetCaseContextInput(BaseModel):
    """Input schema for get_case_context tool"""
    case_id: str = Field(description="Case ID (conversation_id) to fetch aggregated context for")


class GetCaseContextTool(BaseTool):
    """LangChain tool wrapper for get_case_context"""
    name: str = "get_case_context"
    description: str = "Fetches aggregated case context from MongoDB (support_tickets collection). Returns consolidated view of support ticket history, related orders, agent notes, and resolution history."
    args_schema: Type[BaseModel] = GetCaseContextInput
    
    async def _arun(self, case_id: str) -> str:
        """Async execution - returns JSON string of CaseEvidenceEnvelope"""
        result = await get_case_context(case_id)
        return result.model_dump_json()
    
    def _run(self, case_id: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
