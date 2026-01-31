"""
Tool: search_policies
Searches policy documents in Elasticsearch

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime
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
        # Mock implementation for hackathon
        policy_data = {
            "query": query,
            "results": [
                {
                    "policy_id": "POL-REFUND-001",
                    "title": "Refund Policy - Food Quality Issues",
                    "content": "Customers are eligible for a full refund if food quality issues are reported within 2 hours of delivery...",
                    "relevance_score": 0.92,
                    "document_type": "policy",
                    "section": "refunds",
                    "effective_date": "2025-01-01"
                },
                {
                    "policy_id": "POL-REFUND-002",
                    "title": "Refund Policy - Delivery Delays",
                    "content": "For delivery delays exceeding 45 minutes, customers may request a partial refund...",
                    "relevance_score": 0.85,
                    "document_type": "policy",
                    "section": "refunds",
                    "effective_date": "2025-01-01"
                }
            ],
            "total_results": 2
        }
        
        result = PolicyEvidenceEnvelope(
            source="elasticsearch",
            entity_refs=[],
            freshness=datetime.utcnow(),
            confidence=0.90,
            data=policy_data,
            gaps=[],
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
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["policy_search_failed"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class SearchPoliciesInput(BaseModel):
    """Input schema for search_policies tool"""
    query: str = Field(description="Search query string for policy documents")
    filters: Dict = Field(default={}, description="Filters dictionary (e.g., {'issue_type': 'refund'})")
    top_k: int = Field(default=5, description="Number of top results to return")


class SearchPoliciesTool(BaseTool):
    """LangChain tool wrapper for search_policies"""
    name: str = "search_policies"
    description: str = "Searches policy documents in Elasticsearch. Returns relevant policies, SOPs, and SLAs matching the query and filters."
    args_schema: Type[BaseModel] = SearchPoliciesInput
    
    async def _arun(self, query: str, filters: Dict, top_k: int) -> dict:
        """Async execution - returns dict representation of PolicyEvidenceEnvelope"""
        result = await search_policies(query, filters, top_k)
        return result.dict()
    
    def _run(self, query: str, filters: Dict, top_k: int) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
