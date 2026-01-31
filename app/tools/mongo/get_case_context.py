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
        # Mock implementation for hackathon
        context_data = {
            "case_id": case_id,
            "created_at": "2026-01-30T10:00:00Z",
            "status": "open",
            "priority": "medium",
            "related_orders": ["order_123", "order_124"],
            "related_cases": [],
            "agent_notes": [],
            "resolution_history": []
        }
        
        result = CaseEvidenceEnvelope(
            source="mongo",
            entity_refs=[case_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.93,
            data=context_data,
            gaps=[],
            provenance={"query": "get_case_context", "latency_ms": 35},
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
    description: str = "Fetches aggregated case context from MongoDB. Returns consolidated view of case history, related orders, agent notes, and resolution history."
    args_schema: Type[BaseModel] = GetCaseContextInput
    
    async def _arun(self, case_id: str) -> str:
        """Async execution - returns JSON string of CaseEvidenceEnvelope"""
        result = await get_case_context(case_id)
        return result.model_dump_json()
    
    def _run(self, case_id: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
