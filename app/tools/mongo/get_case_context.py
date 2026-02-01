"""
Tool: get_case_context
Fetches aggregated case context from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone
from typing import Type, Union

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import CaseEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import string_to_mongo_id, binary_to_uuid
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
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_case_context")
    print(f"  case_id: {case_id}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_case_context] Starting - case_id={case_id}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_case_context",
        "params": {"case_id": case_id}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        # Note: case_id might be ticket_id (ObjectId), _id (ObjectId), or conversation_id (string)
        query_case_id = string_to_mongo_id(case_id) if len(case_id) == 24 else case_id
        logger.debug(f"[get_case_context] Query case_id (converted): {query_case_id}, type: {type(query_case_id).__name__}")
        
        # Query support_tickets collection (replaces cases)
        # Try ticket_id first, then _id, then conversation_id
        logger.debug(f"[get_case_context] Trying ticket_id lookup")
        ticket_doc = await db.support_tickets.find_one({"ticket_id": query_case_id})
        if not ticket_doc:
            logger.debug(f"[get_case_context] Trying _id lookup")
            ticket_doc = await db.support_tickets.find_one({"_id": query_case_id})
        if not ticket_doc:
            logger.debug(f"[get_case_context] Trying conversation_id lookup")
            ticket_doc = await db.support_tickets.find_one({"conversation_id": case_id})
        
        if not ticket_doc:
            logger.warning(f"[get_case_context] Support ticket not found - case_id={case_id}")
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
        # Convert ObjectIds and Binary UUIDs to strings for JSON serialization
        from bson import ObjectId
        
        ticket_id_val = ticket_doc.get("ticket_id")
        if isinstance(ticket_id_val, Binary):
            ticket_id_val = binary_to_uuid(ticket_id_val)
        elif isinstance(ticket_id_val, ObjectId):
            ticket_id_val = str(ticket_id_val)
        
        user_id_val = ticket_doc.get("user_id")
        if isinstance(user_id_val, (Binary, ObjectId)):
            user_id_val = str(user_id_val)
        
        # Convert arrays of ObjectIds to strings
        related_orders = ticket_doc.get("related_orders", [])
        related_orders_str = [str(oid) if isinstance(oid, (ObjectId, Binary)) else oid for oid in related_orders]
        
        related_tickets = ticket_doc.get("related_tickets", [])
        related_tickets_str = [str(oid) if isinstance(oid, (ObjectId, Binary)) else oid for oid in related_tickets]
        
        context_data = {
            "ticket_id": ticket_id_val,
            "case_id": ticket_id_val,  # Keep case_id for backward compatibility
            "conversation_id": ticket_doc.get("conversation_id"),
            "user_id": user_id_val,
            "ticket_type": ticket_doc.get("ticket_type"),
            "issue_type": ticket_doc.get("issue_type"),
            "subtype": ticket_doc.get("subtype", {}),
            "severity": ticket_doc.get("severity"),
            "created_at": safe_isoformat(ticket_doc.get("created_at")),
            "status": ticket_doc.get("status"),
            "related_orders": related_orders_str,
            "related_tickets": related_tickets_str,
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
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_case_context - SUCCESS")
        print(f"  ticket_id: {context_data.get('ticket_id')}")
        print(f"  issue_type: {context_data.get('issue_type')}")
        print(f"  status: {context_data.get('status')}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_case_context] Error - case_id={case_id}, error={str(e)}", exc_info=True)
        
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
