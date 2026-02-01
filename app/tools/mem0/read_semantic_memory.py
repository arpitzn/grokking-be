"""
Tool: read_semantic_memory
Reads semantic memories (learned patterns)

Criticality: non-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import MemoryEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="read_semantic_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def read_semantic_memory(query: str, top_k: int) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Reads semantic memories (learned patterns) - app-scoped
    - Returns relevant learned insights and patterns
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue with partial results
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "read_semantic_memory",
        "params": {"query": query, "top_k": top_k}
    })
    
    try:
        # Real Mem0 call - app-scoped semantic memory
        from app.infra.mem0 import get_mem0_client
        
        mem0_client = await get_mem0_client()
        results = await mem0_client.search_memory(
            query=query,
            memory_type="semantic",
            user_id=None,  # App-scoped
            limit=top_k
        )
        
        # Transform to expected format
        patterns = []
        for idx, item in enumerate(results):
            patterns.append({
                "pattern_id": item.get("id", f"pat_{idx}"),
                "insight": item.get("memory", ""),
                "confidence": item.get("metadata", {}).get("confidence", 0.8),
                "metadata": item.get("metadata", {}),
                "structured_attributes": item.get("structured_attributes", {}),
                "app_id": item.get("app_id")
            })
        
        memory_data = {
            "query": query,
            "patterns": patterns,
            "total_found": len(patterns)
        }
        
        result = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.80 if patterns else 0.0,
            data=memory_data,
            gaps=[] if patterns else ["no_semantic_memories"],
            provenance={"query": query, "top_k": top_k, "memory_type": "semantic"},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=memory_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "read_semantic_memory",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "read_semantic_memory",
            "error": str(e)
        })
        
        return MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["semantic_memory_unavailable"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class ReadSemanticMemoryInput(BaseModel):
    """Input schema for read_semantic_memory tool"""
    query: str = Field(description="Query string to find relevant learned patterns and insights")
    top_k: int = Field(default=5, description="Number of top patterns to return")


class ReadSemanticMemoryTool(BaseTool):
    """LangChain tool wrapper for read_semantic_memory"""
    name: str = "read_semantic_memory"
    description: str = "Reads semantic memories (learned patterns) from Mem0. Returns relevant learned insights and patterns (app-wide)."
    args_schema: Type[BaseModel] = ReadSemanticMemoryInput
    
    async def _arun(self, query: str, top_k: int=5) -> str:
        """Async execution - returns JSON string of MemoryEvidenceEnvelope"""
        result = await read_semantic_memory(query, top_k)
        return result.model_dump_json()
    
    def _run(self, query: str, top_k: int=5) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
