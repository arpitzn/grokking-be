"""
Agent Responsibility:
- Generates final response for auto-routed cases using LLM reasoning
- Synthesizes evidence naturally
- Includes citations and explanations
- Handles all response types agentically (greetings, clarifications, issue responses)
- Does NOT handle human escalation
"""

from typing import Dict, Any

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_expensive_model, get_cheap_model
from app.infra.prompts import get_prompts
from app.infra.guardrails import get_guardrails_manager
from app.infra.guardrails_messages import get_i_dont_know_message


async def response_synthesis_node(state: AgentState) -> AgentState:
    """
    Response synthesis node: Generates final response for auto-routed cases using LLM.
    Uses agentic reasoning for ALL response types including greetings and clarifications.
    
    Input: analysis, evidence, intent, case
    Output: final_response
    """
    analysis = state.get("analysis", {})
    evidence = state.get("evidence", {})
    intent = state.get("intent", {})
    case = state.get("case", {})
    
    issue_type = intent.get("issue_type")
    needs_more_data = analysis.get("needs_more_data", False)
    gaps = analysis.get("gaps", [])
    
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
            "persona": case.get('persona', 'customer'),
            "raw_text": case.get('raw_text', ''),
            "issue_type": issue_type or 'unknown',
            "needs_more_data": str(needs_more_data),
            "gaps": str(gaps) if gaps else "None",
            "top_hypothesis": top_hypothesis.get('hypothesis', ''),
            "hypothesis_confidence": f"{top_hypothesis.get('confidence', 0.0):.2f}",
            "top_action": top_action.get('action', ''),
            "action_rationale": top_action.get('rationale', '')
        }
    )
    
    # Add current turn prompts
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    # Model selection: Use cheap model for simple conversational queries, expensive for complex issues
    severity = intent.get("severity", "low")
    if severity == "low" and issue_type in ["greeting", "acknowledgment", "question", "clarification_request"]:
        model_name = get_cheap_model()
        temperature = 0.7  # Friendly and natural for conversational responses
    else:
        model_name = get_expensive_model()
        temperature = 0.3  # More careful for complex issues
    
    # Use LLM for ALL response synthesis (agentic behavior)
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=model_name,
        temperature=temperature
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Extract response content
    final_response = response.content if hasattr(response, 'content') else str(response)
    
    # Emit phase event
    emit_phase_event(state, "generating", "Composing final response")
    
    # ============================================================================
    # OUTPUT GUARDRAILS: Validate response before streaming
    # ============================================================================
    guardrails = get_guardrails_manager()
    
    # Validate output through guardrails
    output_result = await guardrails.validate_output(
        response=final_response,
        context={
            "user_id": state.get("case", {}).get("user_id", "unknown"),
            "conversation_id": state.get("case", {}).get("conversation_id"),
            "persona": intent.get("persona"),
            "issue_type": issue_type
        }
    )
    
    # Use validated/modified response (never block, always empathetic)
    final_response = output_result.message
    
    # ============================================================================
    # CONFIDENCE-BASED "I DON'T KNOW" HANDLING
    # ============================================================================
    confidence = state.get("confidence_scores", {}).get("overall", 
                state.get("analysis", {}).get("confidence", 1.0))
    
    if confidence < 0.4:
        # Very low confidence - generate "I don't know" response with escalation offer
        final_response = get_i_dont_know_message() + "\n\n" + \
            "Would you like me to:\n" + \
            "1. Escalate to a senior agent (response in 15 min)\n" + \
            "2. Have someone call you back\n\n" + \
            "Which would you prefer?"
        # Note: Escalation flag will be set by guardrails_agent if needed
    elif confidence < 0.6:
        # Low confidence - partial response + offer escalation
        final_response += "\n\nHowever, for a definitive answer on this specific situation, " + \
            "let me check with our team. Would you like me to escalate this " + \
            "to get you a confirmed response?"
    
    # ============================================================================
    # HALLUCINATION CHECK: Only for RAG responses
    # ============================================================================
    rag_context = _extract_rag_context(state.get("evidence", {}))
    if rag_context:
        hallucination_result = await guardrails.check_hallucination(
            response=final_response,
            rag_context=rag_context,
            user_id=state.get("case", {}).get("user_id", "unknown")
        )
        if hallucination_result.detected:
            final_response += hallucination_result.warning_message
    
    # WRITE to state.messages for observability and multi-turn continuity
    # This is safe because synthesis runs sequentially (after reasoning)
    return {
        "final_response": final_response,
        "messages": lc_messages  # Write to state.messages (single-writer)
    }


def _extract_rag_context(evidence: Dict[str, Any]) -> str:
    """
    Extract RAG context from evidence state for hallucination checking.
    
    Args:
        evidence: Evidence dictionary with mongo, policy, memory keys
        
    Returns:
        Combined RAG context string, or empty string if no RAG context available
    """
    rag_parts = []
    
    # Extract from policy evidence (most relevant for RAG)
    policy_evidence = evidence.get("policy", [])
    for ev in policy_evidence:
        if isinstance(ev, dict):
            # Extract text content from evidence envelope
            content = ev.get("data", {}).get("content") or ev.get("content") or ev.get("text")
            if content:
                rag_parts.append(str(content))
    
    # Extract from memory evidence
    memory_evidence = evidence.get("memory", [])
    for ev in memory_evidence:
        if isinstance(ev, dict):
            content = ev.get("data", {}).get("content") or ev.get("content") or ev.get("text")
            if content:
                rag_parts.append(str(content))
    
    # Extract from mongo evidence (if it contains retrieved knowledge)
    mongo_evidence = evidence.get("mongo", [])
    for ev in mongo_evidence:
        if isinstance(ev, dict):
            # Only include if it's from knowledge retrieval, not just order data
            source = ev.get("provenance", {}).get("source", "")
            if "knowledge" in source.lower() or "rag" in source.lower():
                content = ev.get("data", {}).get("content") or ev.get("content")
                if content:
                    rag_parts.append(str(content))
    
    return "\n\n".join(rag_parts) if rag_parts else ""
