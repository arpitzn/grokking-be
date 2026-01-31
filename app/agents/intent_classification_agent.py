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


class IntentOutput(BaseModel):
    """Structured output for intent classification agent"""
    issue_type: Literal["refund", "delivery_delay", "quality", "safety", "account", "other"] = Field(
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
    reasoning: str = Field(
        ...,
        description="Brief explanation of classification decision"
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
    
    # Build prompt for intent classification
    prompt = f"""Classify this food delivery support query.

Query: "{normalized_text}"
Order ID: {order_id or "Not mentioned"}

Classify the query:
1. issue_type: Choose ONE from ["refund", "delivery_delay", "quality", "safety", "account", "other"]
2. severity: Choose ONE from ["low", "medium", "high"]
   - high: Urgent issues, safety concerns, angry customers, SLA violations
   - medium: Standard complaints, delays, quality issues
   - low: Simple questions, account updates, general inquiries
3. SLA_risk: true if this might violate service level agreements (e.g., long delays, repeated issues)
4. safety_flags: List any safety concerns (e.g., ["food_safety"], ["driver_behavior"], or empty list)
5. reasoning: Brief explanation of your classification
6. confidence: Your confidence in this classification (0.0 to 1.0)

Examples:
- "My order is 2 hours late and I want a refund" → issue_type: "refund", severity: "high", SLA_risk: true
- "Food was cold" → issue_type: "quality", severity: "medium", SLA_risk: false
- "How do I update my address?" → issue_type: "account", severity: "low", SLA_risk: false
- "Driver was rude and driving dangerously" → issue_type: "safety", severity: "high", safety_flags: ["driver_behavior"]
"""
    
    messages = [
        {
            "role": "system", 
            "content": "You are an intent classification system for food delivery support. Classify accurately and provide confidence scores."
        },
        {"role": "user", "content": prompt}
    ]
    
    # Use structured output with cheap model
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_cheap_model(),
        schema=IntentOutput,
        temperature=0.3  # Low temperature for consistent classification
    )
    
    lc_messages = llm_service.convert_messages(messages)
    response: IntentOutput = await llm.ainvoke(lc_messages)
    
    # Populate intent slice
    state["intent"] = {
        "issue_type": response.issue_type,
        "severity": response.severity,
        "SLA_risk": response.SLA_risk,
        "safety_flags": response.safety_flags,
        "reasoning": response.reasoning,
        "confidence": response.confidence  # Add confidence
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
