"""
Tool: write_memory
Writes to Mem0 with proper classification (episodic/semantic/procedural)

Criticality: non-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Literal

from app.models.evidence import MemoryEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event

# Tool specification
TOOL_SPEC = ToolSpec(
    name="write_memory",
    criticality=ToolCriticality.NON_CRITICAL
)


async def write_memory(
    content: str,
    memory_type: Literal["episodic", "semantic", "procedural"],
    user_id: Optional[str] = None,
    additional_metadata: Optional[Dict] = None
) -> MemoryEvidenceEnvelope:
    """
    Tool Responsibility:
    - Writes to Mem0 with proper classification
    - Stores episodic (user-scoped), semantic (app-scoped), or procedural (app-scoped) memories
    
    Args:
        content: Declarative memory statement (not "remember this")
        memory_type: episodic, semantic, or procedural
        user_id: Required for episodic (user-scoped), None for semantic/procedural (app-scoped)
        additional_metadata: Optional extra metadata
    
    Criticality: non-critical (declared in TOOL_SPEC)
    Failure handling: Continue without blocking (async write)
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    emit_tool_event("tool_call_started", {
        "tool_name": "write_memory",
        "params": {
            "memory_type": memory_type,
            "user_id": user_id,
            "scope": "user" if user_id else "application"
        }
    })
    
    try:
        # Real Mem0 call
        from app.infra.mem0 import get_mem0_client
        
        mem0_client = await get_mem0_client()
        
        # Write memory with classification
        result = await mem0_client.add_memory(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            additional_metadata=additional_metadata
        )
        
        write_result = {
            "memory_id": result.get("id") if result else None,
            "memory_type": memory_type,
            "scope": "user" if user_id else "application",
            "user_id": user_id,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "status": "success" if result else "failed"
        }
        
        envelope = MemoryEvidenceEnvelope(
            source="mem0",
            entity_refs=[user_id] if user_id else ["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.90 if result else 0.0,
            data=write_result,
            gaps=[] if result else ["memory_write_failed"],
            provenance={
                "operation": "write_memory",
                "memory_type": memory_type,
                "scope": "user" if user_id else "application"
            },
            tool_result=ToolResult(
                status=ToolStatus.SUCCESS if result else ToolStatus.FAILED,
                data=write_result
            )
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
            entity_refs=[user_id] if user_id else ["app_wide"],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["memory_write_failed"],
            provenance={"operation": "write_memory", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )
