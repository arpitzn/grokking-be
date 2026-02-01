"""
Agent Responsibility:
- Classifies issue type using structured LLM output
- Determines severity, SLA risk, safety flags
- Provides confidence scoring
- Does NOT plan retrieval or generate responses

IMPORTANT: Intent Classification is a MANDATORY pre-planning signal.
Planner depends on intent classification output to make informed decisions.
This agent MUST execute before Planner.
"""

from typing import List, Literal
from pydantic import BaseModel, Field

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_cheap_model
from app.infra.prompts import get_prompts


class IntentOutput(BaseModel):
    """Structured output for intent classification agent"""
    issue_type: Literal["refund", "delivery_delay", "quality", "safety", "account", "greeting", "question", "acknowledgment", "clarification_request", "other"] = Field(
        ...,
        description="Primary issue category"
    )
    severity: Literal["low", "medium", "high"] = Field(
        ...,
        description="Issue severity level"
    )
    SLA_risk: bool = Field(
        ...,
        description="True if this might violate SLA commitments"
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description="List of safety concerns (e.g., ['food_safety', 'driver_safety'])"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0.0 to 1.0)"
    )


async def intent_classification_node(state: AgentState) -> AgentState:
    """
    Intent classification node: Classifies issue type, severity, SLA risk.
    
    Input: case slice (raw_text, normalized_text, entities)
    Output: intent slice (issue_type, severity, SLA_risk, safety_flags, confidence)
    """
    case = state.get("case", {})
    raw_text = case.get("raw_text", "")
    normalized_text = case.get("normalized_text", raw_text)
    order_id = case.get("order_id")
    working_memory = state.get("working_memory", [])
    
    # Add conversation context if multi-turn
    history_context = ""
    # Filter out system messages (summaries), keep user/assistant only
    conversation_messages = [m for m in working_memory if m.get("role") != "system"]
    if len(conversation_messages) > 0:
        history_context = f"\n\nConversation history (last {len(conversation_messages)} messages):\n"
        for msg in conversation_messages[-3:]:  # Last 3 for context
            history_context += f"{msg['role']}: {msg['content'][:100]}...\n"
    
    # Get prompts from centralized prompts module
    system_prompt, user_prompt = get_prompts(
        "intent_classification_agent",
        {
            "normalized_text": normalized_text,
            "order_id": order_id or "Not mentioned",
            "history_context": history_context
        }
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use structured output with cheap model
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_cheap_model(),
        schema=IntentOutput,
        temperature=0  # Low temperature for consistent classification
    )
    
    lc_messages = llm_service.convert_messages(messages)
    response: IntentOutput = await llm.ainvoke(lc_messages)
    
    # Populate intent slice
    state["intent"] = {
        "issue_type": response.issue_type,
        "severity": response.severity,
        "SLA_risk": response.SLA_risk,
        "safety_flags": response.safety_flags,
        "confidence": response.confidence
    }
    
    # Update confidence tracking
    if "confidence_scores" not in state:
        state["confidence_scores"] = {}
    state["confidence_scores"]["intent_classification"] = response.confidence
    
    # Emit phase event
    emit_phase_event(
        state,
        "intent_classification",
        f"{response.issue_type} (severity: {response.severity}, confidence: {response.confidence:.2f})",
        metadata={
            "safety_flags": response.safety_flags,
            "SLA_risk": response.SLA_risk,
            "confidence": response.confidence
        }
    )
    
    return state
