"""
Tool: read_procedural_memory
Reads procedural memories (heuristics, what works)

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
    name="read_procedural_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def read_procedural_memory(query: str, top_k: int) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Reads procedural memories (heuristics, what works) - app-scoped
    - Returns effective actions and best practices
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue with partial results
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "read_procedural_memory",
        "params": {"query": query, "top_k": top_k}
    })
    
    try:
        # Real Mem0 call - app-scoped procedural memory
        from app.infra.mem0 import get_mem0_client
        
        mem0_client = await get_mem0_client()
        results = await mem0_client.search_memory(
            query=query,
            memory_type="procedural",
            user_id=None,  # App-scoped
            limit=top_k
        )
        
        # Transform to expected format
        heuristics = []
        for idx, item in enumerate(results):
            heuristics.append({
                "heuristic_id": item.get("id", f"heur_{idx}"),
                "action": item.get("memory", ""),
                "confidence": item.get("metadata", {}).get("confidence", 0.9),
                "metadata": item.get("metadata", {}),
                "structured_attributes": item.get("structured_attributes", {}),
                "app_id": item.get("app_id")
            })
        
        memory_data = {
            "query": query,
            "heuristics": heuristics,
            "total_found": len(heuristics)
        }
        
        result = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.90 if heuristics else 0.0,
            data=memory_data,
            gaps=[] if heuristics else ["no_procedural_memories"],
            provenance={"query": query, "top_k": top_k, "memory_type": "procedural"},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=memory_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "read_procedural_memory",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "read_procedural_memory",
            "error": str(e)
        })
        
        return MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["procedural_memory_unavailable"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class ReadProceduralMemoryInput(BaseModel):
    """Input schema for read_procedural_memory tool"""
    query: str = Field(description="Query to find relevant heuristics and best practices")
    top_k: int = Field(default=5, description="Number of top heuristics to return")


class ReadProceduralMemoryTool(BaseTool):
    """LangChain tool wrapper for read_procedural_memory"""
    name: str = "read_procedural_memory"
    description: str = "Reads procedural memories (heuristics, what works) from Mem0. Returns effective actions and best practices (app-wide)."
    args_schema: Type[BaseModel] = ReadProceduralMemoryInput
    
    async def _arun(self, query: str, top_k: int) -> str:
        """Async execution - returns JSON string of MemoryEvidenceEnvelope"""
        result = await read_procedural_memory(query, top_k)
        return result.model_dump_json()
    
    def _run(self, query: str, top_k: int) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
