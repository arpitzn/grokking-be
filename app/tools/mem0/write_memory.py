"""
Tool: write_memory
Writes to Mem0 (episodic or semantic memory)

Criticality: non-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Dict

from app.models.evidence import MemoryEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="write_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def write_memory(user_id: str, memory_type: str, content: Dict) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Writes to Mem0 (episodic or semantic memory)
    - Stores case outcomes, learnings, and patterns
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue without blocking (async write)
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "write_memory",
        "params": {"user_id": user_id, "memory_type": memory_type}
    })
    
    try:
        # Mock implementation for hackathon (fire-and-forget async write)
        # In real implementation, this would call Mem0 API
        
        write_result = {
            "user_id": user_id,
            "memory_type": memory_type,
            "memory_id": f"mem_{datetime.now(timezone.utc).timestamp()}",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "status": "success"
        }
        
        result = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.90,
            data=write_result,
            gaps=[],
            provenance={"operation": "write_memory", "memory_type": memory_type},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=write_result)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "write_memory",
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        emit_tool_event("tool_call_failed", {
            "tool_name": "write_memory",
            "error": str(e)
        })
        
        # Non-critical failure - return success with degraded confidence
        return MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["memory_write_failed"],
            provenance={"operation": "write_memory", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
