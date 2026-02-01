"""
Tool: search_policies
Searches policy documents in Elasticsearch

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import PolicyEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="search_policies",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def search_policies(query: str, filters: Dict, top_k: int) -> PolicyEvidenceEnvelope:
    """
    Tool Responsibility:
    - Searches policy documents in Elasticsearch
    - Returns relevant policy chunks with citations and ToolResult
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "search_policies",
        "params": {"query": query, "filters": filters, "top_k": top_k}
    })
    
    try:
        from app.infra.elasticsearch import get_elasticsearch_client
        
        es_client = await get_elasticsearch_client()
        results = await es_client.search_policies(query, filters, top_k)
        
        # Transform results to match expected format
        policy_results = []
        for r in results:
            policy_results.append({
                "file_id": r.get("file_id"),
                "filename": r.get("filename", ""),
                "content": r.get("content", ""),
                "category": r.get("category"),
                "priority": r.get("priority"),
                "issue_type": r.get("issue_type", []),
                "score": r.get("score", 0.0)
            })
        
        policy_data = {
            "query": query,
            "results": policy_results,
            "total_results": len(policy_results)
        }
        
        result = PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[],
            freshness=datetime.now(timezone.utc),
            confidence=0.90 if policy_results else 0.0,
            data=policy_data,
            gaps=[] if policy_results else ["no_policies_found"],
            provenance={"query": query, "top_k": top_k, "filters": filters},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=policy_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "search_policies",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "search_policies",
            "error": str(e)
        })
        
        return PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["policy_search_failed"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class SearchPoliciesInput(BaseModel):
    """Input schema for search_policies tool"""
    query: str = Field(
        description="Search query. Example: 'refund policy for late deliveries'"
    )
    filters: Dict = Field(
        default_factory=dict,
        description=(
            "Optional filters:\n"
            "- priority: 'high', 'medium', 'low'\n"
            "- category: 'policy', 'guideline'\n"
            "- issue_type: ['refund', 'delivery']\n"
            "Example: {'priority': 'high'}"
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results (1-20, default: 5)"
    )


class SearchPoliciesTool(BaseTool):
    """LangChain tool wrapper for search_policies"""
    name: str = "search_policies"
    description: str = "Searches policy documents in Elasticsearch. Returns relevant policies, SOPs, and SLAs matching the query and filters."
    args_schema: Type[BaseModel] = SearchPoliciesInput
    
    async def _arun(self, query: str, filters: Dict, top_k: int) -> str:
        """Async execution - returns JSON string of PolicyEvidenceEnvelope"""
        result = await search_policies(query, filters, top_k)
        return result.model_dump_json()
    
    def _run(self, query: str, filters: Dict, top_k: int) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
