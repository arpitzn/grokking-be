"""
Agent Responsibility:
- Calls Mem0 tools for episodic/semantic memory
- Records execution status in retrieval_status
- Populates evidence.memory[]
- Does NOT reason or generate responses
"""

from datetime import datetime, timezone
from typing import Dict, Any

from app.agent.state import AgentState
from app.models.tool_spec import ToolCriticality
from app.tools.mem0.read_episodic_memory import read_episodic_memory, TOOL_SPEC as READ_EPISODIC_SPEC
from app.tools.mem0.read_semantic_memory import read_semantic_memory, TOOL_SPEC as READ_SEMANTIC_SPEC


# Tool registry for Mem0 tools
TOOL_REGISTRY = {
    "read_episodic_memory": (read_episodic_memory, READ_EPISODIC_SPEC),
    "read_semantic_memory": (read_semantic_memory, READ_SEMANTIC_SPEC),
}


async def memory_retrieval_node(state: AgentState) -> AgentState:
    """
    Memory retrieval node: Calls Mem0 tools.
    
    Input: plan.tool_selection, case (customer_id), intent
    Output: evidence.memory[], retrieval_status.memory
    """
    plan = state.get("plan", {})
    tool_selection = plan.get("tool_selection", [])
    case = state.get("case", {})
    intent = state.get("intent", {})
    
    # Filter to only Mem0 tools
    memory_tools = [tool for tool in tool_selection if tool in TOOL_REGISTRY]
    
    # Initialize evidence and status
    if "evidence" not in state:
        state["evidence"] = {}
    if "memory" not in state["evidence"]:
        state["evidence"]["memory"] = []
    
    if "retrieval_status" not in state:
        state["retrieval_status"] = {}
    
    failed_tools = []
    
    # Build query from intent
    issue_type = intent.get("issue_type", "other")
    query = f"{issue_type} issue customer support"
    user_id = case.get("customer_id", case.get("user_id", ""))
    
    # Call each selected Mem0 tool
    for tool_name in memory_tools:
        if tool_name not in TOOL_REGISTRY:
            continue
        
        tool_func, tool_spec = TOOL_REGISTRY[tool_name]
        
        try:
            # Prepare tool arguments
            if tool_name == "read_episodic_memory":
                result = await tool_func(
                    user_id=user_id,
                    query=query,
                    top_k=5
                )
            elif tool_name == "read_semantic_memory":
                result = await tool_func(
                    user_id=user_id,
                    query=query,
                    top_k=5
                )
            else:
                continue
            
            # Handle failures based on ToolSpec criticality (Mem0 tools are non-critical)
            if result.tool_result.status.value == "failed":
                # Non-critical: Continue with partial results
                state["evidence"]["memory"].append(result.dict())
            else:
                # Success - add to evidence
                state["evidence"]["memory"].append(result.dict())
        
        except Exception as e:
            # Non-critical failure - continue
            failed_envelope = {
                "source": "mem0",
                "entity_refs": [user_id],
                "freshness": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.0,
                "data": {},
                "gaps": [f"{tool_name}_exception"],
                "provenance": {"tool": tool_name, "error": str(e)},
                "tool_result": {"status": "failed", "error": str(e)}
            }
            state["evidence"]["memory"].append(failed_envelope)
    
    # Update retrieval status
    state["retrieval_status"]["memory"] = {
        "completed": True,
        "failed_tools": failed_tools,
        "successful_tools": [tool for tool in memory_tools if tool not in failed_tools]
    }
    
    return state
