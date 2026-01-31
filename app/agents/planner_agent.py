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

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_cheap_model


class PlanningOutput(BaseModel):
    """Structured output for planner LLM"""
    agents_to_activate: List[Literal["mongo_retrieval", "policy_rag", "memory_retrieval"]] = Field(
        description="List of agent names to activate in parallel"
    )
    tool_selection: List[str] = Field(
        description="List of tool names for reference (e.g., ['get_order_timeline', 'search_policies'])"
    )
    retrieval_strategy: Literal["parallel", "sequential"] = Field(
        default="parallel",
        description="How to execute agents: parallel or sequential"
    )
    initial_route: Literal["auto", "human"] = Field(
        description="Advisory routing decision: auto for auto-response, human for escalation"
    )
    reasoning: str = Field(
        description="Brief explanation of agent and tool selection"
    )


async def planner_node(state: AgentState) -> AgentState:
    """
    Agentic planner: Uses LLM to decide which tools to call
    
    Input: intent slice (from Intent Classification - MANDATORY), case slice, conversation_history
    Output: plan slice (tool_selection, retrieval_strategy, context, initial_route ADVISORY)
    """
    intent = state.get("intent", {})
    case = state.get("case", {})
    conversation_history = state.get("conversation_history", [])
    turn_number = state.get("turn_number", 1)
    
    # Build context for LLM planner
    available_tools = """
    Available tools:
    1. get_order_timeline - Fetch order events, status, timestamps
    2. get_customer_ops_profile - Get customer history, refund count, VIP status
    3. get_zone_ops_metrics - Get zone-level delivery metrics
    4. get_incident_signals - Check for active incidents
    5. get_restaurant_ops - Get restaurant operational data
    6. get_case_context - Get case metadata
    7. search_policies - Search policy documents (refund, SLA, quality)
    8. lookup_policy - Get specific policy by ID
    9. read_episodic_memory - Search past conversations
    10. read_semantic_memory - Get user preferences and facts
    """
    
    # Add conversation context if multi-turn
    history_context = ""
    if turn_number > 1 and conversation_history:
        history_context = f"\n\nConversation history (last {len(conversation_history)} messages):\n"
        for msg in conversation_history[-3:]:  # Last 3 for context
            history_context += f"{msg['role']}: {msg['content'][:100]}...\n"
    
    prompt = f"""You are a planning agent for a food delivery support system.

Current query: {case.get('raw_text', '')}
Turn number: {turn_number}

Intent classification:
- Issue type: {intent.get('issue_type', 'unknown')}
- Severity: {intent.get('severity', 'low')}
- SLA risk: {intent.get('SLA_risk', False)}
- Safety flags: {intent.get('safety_flags', [])}

Extracted entities:
- Order ID: {case.get('order_id', 'none')}
- Customer ID: {case.get('customer_id', 'none')}
- Zone ID: {case.get('zone_id', 'none')}
- Restaurant ID: {case.get('restaurant_id', 'none')}
{history_context}

{available_tools}

Based on the query, intent, and entities, decide:
1. Which tools should be called to gather relevant information?
2. Should this be handled automatically or escalated to human?

Guidelines:
- For refund requests: get order timeline, customer profile, refund policy
- For delivery delays: get order timeline, zone metrics, delivery SLA policy
- For quality issues: get order timeline, restaurant ops, quality policy
- For safety concerns: get incident signals, safety policy, escalate to human
- Always search episodic memory to check past similar cases
- If high severity or SLA risk, recommend human escalation
"""
    
    # Call LLM with structured output
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_cheap_model(),
        schema=PlanningOutput,
        temperature=0.3
    )
    
    messages = [
        {"role": "system", "content": "You are a planning agent. Analyze the query and decide which tools to use."},
        {"role": "user", "content": prompt}
    ]
    
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Parse structured output
    planning_output: PlanningOutput = response
    
    # Populate state["plan"]
    state["plan"] = {
        "agents_to_activate": planning_output.agents_to_activate,
        "tool_selection": planning_output.tool_selection,
        "retrieval_strategy": planning_output.retrieval_strategy,
        "context": {
            "issue_type": intent.get("issue_type"),
            "severity": intent.get("severity"),
            "entities": case
        },
        "initial_route": planning_output.initial_route
    }
    
    # Emit phase event
    emit_phase_event(
        state,
        "planning",
        f"Selected {len(planning_output.agents_to_activate)} retrieval agents",
        metadata={
            "agents": planning_output.agents_to_activate,
            "tools": planning_output.tool_selection,
            "route": planning_output.initial_route,
            "evidence_count": 0  # Will be updated by retrieval
        }
    )
    
    # Guardrails Agent has FINAL authority to override initial_route
    return state
