"""
Tool: read_semantic_memory
Reads semantic memories (learned patterns)

Criticality: non-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime

from app.models.evidence import MemoryEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="read_semantic_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def read_semantic_memory(user_id: str, query: str, top_k: int) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Reads semantic memories (learned patterns)
    - Returns relevant learned insights and patterns
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue with partial results
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "read_semantic_memory",
        "params": {"user_id": user_id, "query": query, "top_k": top_k}
    })
    
    try:
        # Mock implementation for hackathon
        memory_data = {
            "user_id": user_id,
            "query": query,
            "patterns": [
                {
                    "pattern_id": "pat_001",
                    "insight": "Customer prefers quick resolution for quality issues",
                    "confidence": 0.82,
                    "learned_from": 3
                },
                {
                    "pattern_id": "pat_002",
                    "insight": "Customer values clear communication about refund timelines",
                    "confidence": 0.75,
                    "learned_from": 2
                }
            ],
            "total_found": 2
        }
        
        result = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id],
            freshness=datetime.utcnow(),
            confidence=0.80,
            data=memory_data,
            gaps=[],
            provenance={"query": query, "top_k": top_k},
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
            entity_refs=[user_id],
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["semantic_memory_unavailable"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
