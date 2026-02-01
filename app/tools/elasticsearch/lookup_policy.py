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


async def lookup_policy(doc_id: str) -> PolicyEvidenceEnvelope:
    """
    Tool Responsibility:
    - Looks up specific policy document by file_id in Elasticsearch
    - Returns full policy document with concatenated content from all chunks
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "lookup_policy",
        "params": {"doc_id": doc_id}
    })
    
    try:
        from app.infra.elasticsearch import get_elasticsearch_client
        
        es_client = await get_elasticsearch_client()
        policy_doc = await es_client.lookup_policy_by_file_id(doc_id)
        
        if not policy_doc:
            return PolicyEvidenceEnvelope(
                source="elasticsearch",
                entity_refs=[doc_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["policy_not_found"],
                provenance={"query": "lookup_policy", "doc_id": doc_id},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Policy not found")
            )
        
        policy_data = {
            "file_id": policy_doc.get("file_id"),
            "filename": policy_doc.get("filename", ""),
            "content": policy_doc.get("content", ""),
            "category": policy_doc.get("category"),
            "priority": policy_doc.get("priority"),
            "issue_type": policy_doc.get("issue_type", [])
        }
        
        result = PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[doc_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.95,
            data=policy_data,
            gaps=[],
            provenance={"query": "lookup_policy", "doc_id": doc_id},
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
    doc_id: str = Field(description="Policy document file_id to lookup")


class LookupPolicyTool(BaseTool):
    """LangChain tool wrapper for lookup_policy"""
    name: str = "lookup_policy"
    description: str = "Looks up specific policy document by file_id in Elasticsearch. Returns full policy content with all chunks concatenated."
    args_schema: Type[BaseModel] = LookupPolicyInput
    
    async def _arun(self, doc_id: str) -> str:
        """Async execution - returns JSON string of PolicyEvidenceEnvelope"""
        result = await lookup_policy(doc_id)
        return result.model_dump_json()
    
    def _run(self, doc_id: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
