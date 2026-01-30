"""
Agent Responsibility:
- Calls MongoDB read-model tools based on planner selection
- Handles tool failures based on declared criticality
- Records execution status in retrieval_status
- Populates evidence.mongo[]
- Does NOT reason or generate responses
- Agents may escalate but never downgrade tool criticality
"""

from typing import Dict, Any, List

from app.agent.state import AgentState
from app.models.tool_spec import ToolCriticality
from app.tools.mongo.get_case_context import get_case_context, TOOL_SPEC as CASE_CONTEXT_SPEC
from app.tools.mongo.get_customer_ops_profile import get_customer_ops_profile, TOOL_SPEC as CUSTOMER_PROFILE_SPEC
from app.tools.mongo.get_incident_signals import get_incident_signals, TOOL_SPEC as INCIDENT_SIGNALS_SPEC
from app.tools.mongo.get_order_timeline import get_order_timeline, TOOL_SPEC as ORDER_TIMELINE_SPEC
from app.tools.mongo.get_restaurant_ops import get_restaurant_ops, TOOL_SPEC as RESTAURANT_OPS_SPEC
from app.tools.mongo.get_zone_ops_metrics import get_zone_ops_metrics, TOOL_SPEC as ZONE_METRICS_SPEC


# Tool registry for easy lookup
TOOL_REGISTRY = {
    "get_order_timeline": (get_order_timeline, ORDER_TIMELINE_SPEC),
    "get_customer_ops_profile": (get_customer_ops_profile, CUSTOMER_PROFILE_SPEC),
    "get_zone_ops_metrics": (get_zone_ops_metrics, ZONE_METRICS_SPEC),
    "get_incident_signals": (get_incident_signals, INCIDENT_SIGNALS_SPEC),
    "get_restaurant_ops": (get_restaurant_ops, RESTAURANT_OPS_SPEC),
    "get_case_context": (get_case_context, CASE_CONTEXT_SPEC),
}


async def mongo_retrieval_node(state: AgentState) -> AgentState:
    """
    Mongo retrieval node: Calls MongoDB tools based on planner selection.
    
    Input: plan.tool_selection, case entities
    Output: evidence.mongo[], retrieval_status.mongo
    """
    plan = state.get("plan", {})
    tool_selection = plan.get("tool_selection", [])
    case = state.get("case", {})
    
    # Filter to only MongoDB tools
    mongo_tools = [tool for tool in tool_selection if tool in TOOL_REGISTRY]
    
    # Initialize evidence and status
    if "evidence" not in state:
        state["evidence"] = {}
    if "evidence" not in state["evidence"]:
        state["evidence"]["mongo"] = []
    
    if "retrieval_status" not in state:
        state["retrieval_status"] = {}
    
    failed_tools = []
    
    # Call each selected MongoDB tool
    for tool_name in mongo_tools:
        if tool_name not in TOOL_REGISTRY:
            continue
        
        tool_func, tool_spec = TOOL_REGISTRY[tool_name]
        
        try:
            # Prepare tool arguments based on tool name and case entities
            if tool_name == "get_order_timeline":
                result = await tool_func(
                    order_id=case.get("order_id", ""),
                    include=["events", "status", "timestamps"]
                )
            elif tool_name == "get_customer_ops_profile":
                result = await tool_func(customer_id=case.get("customer_id", ""))
            elif tool_name == "get_zone_ops_metrics":
                result = await tool_func(
                    zone_id=case.get("zone_id", ""),
                    time_window="24h"
                )
            elif tool_name == "get_incident_signals":
                result = await tool_func(
                    scope={
                        "order_id": case.get("order_id"),
                        "customer_id": case.get("customer_id"),
                        "zone_id": case.get("zone_id")
                    },
                    time_window="24h"
                )
            elif tool_name == "get_restaurant_ops":
                result = await tool_func(
                    restaurant_id=case.get("restaurant_id", ""),
                    time_window="24h"
                )
            elif tool_name == "get_case_context":
                result = await tool_func(case_id=case.get("conversation_id", ""))
            else:
                continue
            
            # Handle failures based on ToolSpec criticality
            if result.tool_result.status.value == "failed":
                # Check criticality
                if tool_spec.criticality == ToolCriticality.SAFETY_CRITICAL:
                    # Block execution immediately
                    failed_tools.append(tool_name)
                    # Don't add to evidence, but record failure
                elif tool_spec.criticality == ToolCriticality.DECISION_CRITICAL:
                    # Record failure, continue (Guardrails will decide escalation)
                    failed_tools.append(tool_name)
                    state["evidence"]["mongo"].append(result.dict())
                elif tool_spec.criticality == ToolCriticality.NON_CRITICAL:
                    # Continue with partial results
                    state["evidence"]["mongo"].append(result.dict())
            else:
                # Success - add to evidence
                state["evidence"]["mongo"].append(result.dict())
        
        except Exception as e:
            # Exception handling based on criticality
            if tool_spec.criticality == ToolCriticality.SAFETY_CRITICAL:
                failed_tools.append(tool_name)
            else:
                failed_tools.append(tool_name)
                # Add empty evidence envelope for failed tool
                from app.models.evidence import ToolResult, ToolStatus
                from datetime import datetime
                failed_envelope = {
                    "source": "mongo",
                    "entity_refs": [],
                    "freshness": datetime.utcnow().isoformat(),
                    "confidence": 0.0,
                    "data": {},
                    "gaps": [f"{tool_name}_exception"],
                    "provenance": {"tool": tool_name, "error": str(e)},
                    "tool_result": {"status": "failed", "error": str(e)}
                }
                state["evidence"]["mongo"].append(failed_envelope)
    
    # Update retrieval status
    state["retrieval_status"]["mongo"] = {
        "completed": True,
        "failed_tools": failed_tools,
        "successful_tools": [tool for tool in mongo_tools if tool not in failed_tools]
    }
    
    return state
