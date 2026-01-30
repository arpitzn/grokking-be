"""
Tool: read_episodic_memory
Reads episodic memories (past incidents/cases)

Criticality: non-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime

from app.models.evidence import MemoryEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="read_episodic_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def read_episodic_memory(user_id: str, query: str, top_k: int) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Reads episodic memories (past incidents/cases)
    - Returns semantically similar past experiences
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue with partial results
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "read_episodic_memory",
        "params": {"user_id": user_id, "query": query, "top_k": top_k}
    })
    
    try:
        # Mock implementation for hackathon
        memory_data = {
            "user_id": user_id,
            "query": query,
            "memories": [
                {
                    "memory_id": "mem_001",
                    "content": "Customer previously reported food quality issue with Italian restaurant. Refund was issued.",
                    "timestamp": "2026-01-15T14:30:00Z",
                    "similarity_score": 0.88
                },
                {
                    "memory_id": "mem_002",
                    "content": "Customer had delivery delay complaint last month. Partial refund provided.",
                    "timestamp": "2026-01-10T09:15:00Z",
                    "similarity_score": 0.75
                }
            ],
            "total_found": 2
        }
        
        result = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id],
            freshness=datetime.utcnow(),
            confidence=0.85,
            data=memory_data,
            gaps=[],
            provenance={"query": query, "top_k": top_k},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=memory_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "read_episodic_memory",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "read_episodic_memory",
            "error": str(e)
        })
        
        return MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id],
            freshness=datetime.utcnow(),
            confidence=0.0,
            data={},
            gaps=["episodic_memory_unavailable"],
            provenance={"query": query, "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
