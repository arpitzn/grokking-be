"""
Tool: lookup_policy
Looks up specific policy document by ID in Elasticsearch

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime
from typing import Optional

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
            freshness=datetime.utcnow(),
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
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["policy_lookup_failed"],
            provenance={"query": "lookup_policy", "doc_id": doc_id, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
