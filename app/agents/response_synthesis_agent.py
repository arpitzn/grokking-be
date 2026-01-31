"""
Agent Responsibility:
- Generates final response for auto-routed cases
- Synthesizes evidence naturally
- Includes citations and explanations
- Does NOT handle human escalation
"""

from typing import Dict, Any

from app.agent.state import AgentState
from app.infra.llm import get_llm_service, get_expensive_model


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
    
    # Get top hypothesis and action
    hypotheses = analysis.get("hypotheses", [])
    action_candidates = analysis.get("action_candidates", [])
    
    top_hypothesis = hypotheses[0] if hypotheses else {"hypothesis": "Unable to determine", "confidence": 0.0}
    top_action = action_candidates[0] if action_candidates else {"action": "investigate", "rationale": "Need more information"}
    
    # Build prompt for response synthesis
    prompt = f"""You are a customer support agent for a food delivery platform.

Customer Query: {case.get('raw_text', '')}
Issue Type: {intent.get('issue_type', 'unknown')}

Top Hypothesis: {top_hypothesis.get('hypothesis', '')} (Confidence: {top_hypothesis.get('confidence', 0.0):.2f})
Recommended Action: {top_action.get('action', '')}
Rationale: {top_action.get('rationale', '')}

Generate a friendly, helpful response to the customer that:
1. Acknowledges their concern
2. Explains the situation based on the evidence
3. Proposes the recommended action
4. Provides next steps

Be empathetic, clear, and professional. Keep it concise (2-3 paragraphs).
"""
    
    messages = [
        {"role": "system", "content": "You are a helpful customer support agent for a food delivery platform. Be empathetic, clear, and professional."},
        {"role": "user", "content": prompt}
    ]
    
    # Use expensive model for response synthesis
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=get_expensive_model(),
        temperature=0.7
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Extract response content
    final_response = response.content if hasattr(response, 'content') else str(response)
    
    # Populate final_response
    state["final_response"] = final_response
    
    # Add CoT trace entry
    turn_number = state.get("turn_number", 1)
    if "cot_trace" not in state:
        state["cot_trace"] = []
    state["cot_trace"].append({
        "phase": "response_synthesis",
        "turn": turn_number,
        "content": f"[Turn {turn_number}] Generated response for {intent.get('issue_type', 'unknown')} issue"
    })
    
    return state
