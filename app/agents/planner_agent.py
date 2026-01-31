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
from app.infra.prompts import get_prompts


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
    
    # Add conversation context if multi-turn
    history_context = ""
    if turn_number > 1 and conversation_history:
        history_context = f"\n\nConversation history (last {len(conversation_history)} messages):\n"
        for msg in conversation_history[-3:]:  # Last 3 for context
            history_context += f"{msg['role']}: {msg['content'][:100]}...\n"
    
    # Get prompts from centralized prompts module
    system_prompt, user_prompt = get_prompts(
        "planner_agent",
        {
            "raw_text": case.get('raw_text', ''),
            "turn_number": str(turn_number),
            "issue_type": intent.get('issue_type', 'unknown'),
            "severity": intent.get('severity', 'low'),
            "sla_risk": str(intent.get('SLA_risk', False)),
            "safety_flags": str(intent.get('safety_flags', [])),
            "order_id": case.get('order_id', 'none'),
            "customer_id": case.get('customer_id', 'none'),
            "zone_id": case.get('zone_id', 'none'),
            "restaurant_id": case.get('restaurant_id', 'none'),
            "history_context": history_context
        }
    )
    
    # Call LLM with structured output
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_cheap_model(),
        schema=PlanningOutput,
        temperature=0.3
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
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
