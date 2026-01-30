"""
Agent Responsibility:
- Decides which retrieval tools to activate (explicit tool selection)
- Creates retrieval plan with tool names
- Outputs ADVISORY initial_route recommendation (auto | human)
- Does NOT set confidence thresholds (Guardrails Agent owns this)
- Does NOT make FINAL routing decision (Guardrails Agent owns this)
- Does NOT fetch data or generate final responses

IMPORTANT: Planner depends on Intent Classification output.
Intent Classification is a mandatory pre-planning signal.
"""

from typing import List, Dict, Any

from app.agent.state import AgentState


def select_tools_for_intent(intent: Dict[str, Any], case: Dict[str, Any]) -> List[str]:
    """
    Select explicit tools based on intent and case context.
    
    Returns list of tool names (e.g., ["get_order_timeline", "search_policies"])
    """
    tool_selection = []
    issue_type = intent.get("issue_type", "other")
    case_entities = case
    
    # Base tool selection on issue_type
    if issue_type == "refund":
        if case_entities.get("order_id"):
            tool_selection.append("get_order_timeline")
        if case_entities.get("customer_id"):
            tool_selection.append("get_customer_ops_profile")
        tool_selection.append("search_policies")  # Need refund policy
        tool_selection.append("read_episodic_memory")  # Past refund cases
    
    elif issue_type == "delivery_delay":
        if case_entities.get("order_id"):
            tool_selection.append("get_order_timeline")
        if case_entities.get("zone_id"):
            tool_selection.append("get_zone_ops_metrics")
        tool_selection.append("search_policies")  # Need delivery SLA policy
        tool_selection.append("read_episodic_memory")
    
    elif issue_type == "quality":
        if case_entities.get("order_id"):
            tool_selection.append("get_order_timeline")
        if case_entities.get("restaurant_id"):
            tool_selection.append("get_restaurant_ops")
        tool_selection.append("search_policies")  # Need quality policy
        tool_selection.append("read_episodic_memory")
    
    elif issue_type == "safety":
        tool_selection.append("get_incident_signals")
        tool_selection.append("search_policies")  # Need safety policy
        tool_selection.append("read_episodic_memory")
    
    else:
        # Default: minimal tool selection
        if case_entities.get("order_id"):
            tool_selection.append("get_order_timeline")
        tool_selection.append("search_policies")
    
    return tool_selection


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: Selects tools and creates retrieval plan.
    
    Input: intent slice (from Intent Classification - MANDATORY), case slice
    Output: plan slice (tool_selection, retrieval_strategy, context, initial_route ADVISORY)
    """
    # Intent Classification output (mandatory signal)
    intent = state.get("intent", {})
    issue_type = intent.get("issue_type", "other")
    severity = intent.get("severity", "low")
    SLA_risk = intent.get("SLA_risk", False)
    safety_flags = intent.get("safety_flags", [])
    
    case = state.get("case", {})
    
    # Select explicit tools based on intent
    tool_selection = select_tools_for_intent(intent, case)
    
    # Determine advisory initial route based on intent signals
    # This is ADVISORY - Guardrails has final authority
    if safety_flags or severity == "high":
        initial_route = "human"  # Advisory: recommend escalation
    elif SLA_risk:
        initial_route = "human"  # Advisory: recommend escalation
    else:
        initial_route = "auto"  # Advisory: recommend auto-response
    
    # Populate state["plan"] with:
    state["plan"] = {
        "tool_selection": tool_selection,  # List[str] - explicit tool names
        "retrieval_strategy": "parallel",  # parallel | sequential
        "context": {
            "issue_type": issue_type,
            "severity": severity,
            "entities": case
        },
        "initial_route": initial_route  # ADVISORY routing (auto | human)
    }
    
    # Add CoT trace entry
    if "cot_trace" not in state:
        state["cot_trace"] = []
    state["cot_trace"].append({
        "phase": "planning",
        "content": f"Selected tools: {tool_selection}, Advisory route: {initial_route}"
    })
    
    # Guardrails Agent has FINAL authority to override initial_route
    return state
