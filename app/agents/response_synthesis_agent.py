"""
Agent Responsibility:
- Generates final response for auto-routed cases
- Synthesizes evidence naturally
- Includes citations and explanations
- Does NOT handle human escalation
"""

from typing import Dict, Any

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_expensive_model
from app.infra.prompts import get_prompts


async def response_synthesis_node(state: AgentState) -> AgentState:
    """
    Response synthesis node: Generates final response for auto-routed cases.
    
    Input: analysis, evidence, intent, case
    Output: final_response
    """
    analysis = state.get("analysis", {})
    evidence = state.get("evidence", {})
    intent = state.get("intent", {})
    case = state.get("case", {})
    
    # Handle greetings with simple response
    issue_type = intent.get("issue_type")
    if issue_type == "greeting":
        greeting_response = (
            "Hello! I'm your food delivery support assistant. "
            "How can I help you today?"
        )
        emit_phase_event(state, "generating", "Generated greeting response")
        return {
            "final_response": greeting_response,
            "messages": []
        }
    
    # Get top hypothesis and action
    hypotheses = analysis.get("hypotheses", [])
    action_candidates = analysis.get("action_candidates", [])
    
    top_hypothesis = hypotheses[0] if hypotheses else {"hypothesis": "Unable to determine", "confidence": 0.0}
    top_action = action_candidates[0] if action_candidates else {"action": "investigate", "rationale": "Need more information"}
    
    # Build execution messages from working memory + current turn
    messages = []
    
    # Add working memory for multi-turn context
    working_memory = state.get("working_memory", [])
    for msg in working_memory:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Get prompts from centralized prompts module for current turn
    system_prompt, user_prompt = get_prompts(
        "response_synthesis_agent",
        {
            "raw_text": case.get('raw_text', ''),
            "issue_type": intent.get('issue_type', 'unknown'),
            "top_hypothesis": top_hypothesis.get('hypothesis', ''),
            "hypothesis_confidence": f"{top_hypothesis.get('confidence', 0.0):.2f}",
            "top_action": top_action.get('action', ''),
            "action_rationale": top_action.get('rationale', '')
        }
    )
    
    # Add current turn prompts
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    # Use expensive model for response synthesis
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=get_expensive_model(),
        temperature=0.3
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Extract response content
    final_response = response.content if hasattr(response, 'content') else str(response)
    
    # Emit phase event
    emit_phase_event(state, "generating", "Composing final response")
    
    # WRITE to state.messages for observability and multi-turn continuity
    # This is safe because synthesis runs sequentially (after reasoning)
    return {
        "final_response": final_response,
        "messages": lc_messages  # Write to state.messages (single-writer)
    }
