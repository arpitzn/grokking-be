"""
Tool: lookup_policy
Looks up specific policy document by ID in Elasticsearch

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import PolicyEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="lookup_policy",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def lookup_policy(doc_id: str, section_id: Optional[str] = None) -> PolicyEvidenceEnvelope:
    """
    Tool Responsibility:
    - Looks up specific policy document by ID in Elasticsearch
    - Optionally retrieves specific section if section_id provided
    - Returns full policy document with ToolResult
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "lookup_policy",
        "params": {"doc_id": doc_id, "section_id": section_id}
    })
    
    try:
        # Mock implementation for hackathon
        policy_data = {
            "policy_id": doc_id,
            "title": "Refund Policy - Food Quality Issues",
            "full_content": "Customers are eligible for a full refund if food quality issues are reported within 2 hours of delivery. The refund will be processed within 24-48 hours...",
            "document_type": "policy",
            "section": section_id or "full",
            "effective_date": "2025-01-01",
            "last_updated": "2025-06-15"
        }
        
        result = PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[doc_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.95,
            data=policy_data,
            gaps=[],
            provenance={"query": "lookup_policy", "doc_id": doc_id, "section_id": section_id},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=policy_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "lookup_policy",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "lookup_policy",
            "error": str(e)
        })
        
        return PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[doc_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["policy_lookup_failed"],
            provenance={"query": "lookup_policy", "doc_id": doc_id, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class LookupPolicyInput(BaseModel):
    """Input schema for lookup_policy tool"""
    doc_id: str = Field(description="Policy document ID to lookup (e.g., 'POL-REFUND-001')")
    section_id: Optional[str] = Field(default=None, description="Optional section ID within the policy document")


class LookupPolicyTool(BaseTool):
    """LangChain tool wrapper for lookup_policy"""
    name: str = "lookup_policy"
    description: str = "Looks up specific policy document by ID in Elasticsearch. Returns full policy content or specific section if section_id is provided."
    args_schema: Type[BaseModel] = LookupPolicyInput
    
    async def _arun(self, doc_id: str, section_id: Optional[str] = None) -> str:
        """Async execution - returns JSON string of PolicyEvidenceEnvelope"""
        result = await lookup_policy(doc_id, section_id)
        return result.model_dump_json()
    
    def _run(self, doc_id: str, section_id: Optional[str] = None) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
