"""
Agent Responsibility:
- Classifies issue type (refund, delivery_delay, quality, etc.)
- Determines severity (low, medium, high)
- Assesses SLA risk
- Sets safety flags
- Does NOT plan retrieval or generate responses

IMPORTANT: Intent Classification is a MANDATORY pre-planning signal.
Planner depends on intent classification output to make informed decisions.
This agent MUST execute before Planner.
"""

import json
from typing import Dict, Any

from app.agent.state import AgentState
from app.infra.llm import get_llm_service, get_cheap_model


async def intent_classification_node(state: AgentState) -> AgentState:
    """
    Intent classification node: Classifies issue type, severity, SLA risk.
    
    Input: case slice (raw_text, entities)
    Output: intent slice (issue_type, severity, SLA_risk, safety_flags, reasoning)
    """
    case = state.get("case", {})
    raw_text = case.get("raw_text", "")
    
    # Build prompt for intent classification
    prompt = f"""Classify the following customer query for a food delivery support system.

Query: {raw_text}

Extract and classify:
1. issue_type: One of ["refund", "delivery_delay", "quality", "safety", "account", "other"]
2. severity: One of ["low", "medium", "high"]
3. SLA_risk: boolean (true if this might violate SLA commitments)
4. safety_flags: List of safety concerns (empty list if none)
5. reasoning: Brief explanation of classification

Respond ONLY with valid JSON:
{{
    "issue_type": "refund",
    "severity": "medium",
    "SLA_risk": true,
    "safety_flags": [],
    "reasoning": "Customer requesting refund for delayed delivery"
}}
"""
    
    messages = [
        {"role": "system", "content": "You are an intent classification system for food delivery support. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    # Use cheap model for classification
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=get_cheap_model(),
        temperature=0.3  # Lower temperature for more deterministic classification
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Parse JSON response
    try:
        content = response.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        intent_data = json.loads(content)
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback to default classification
        intent_data = {
            "issue_type": "other",
            "severity": "low",
            "SLA_risk": False,
            "safety_flags": [],
            "reasoning": f"Classification failed, using defaults. Error: {str(e)}"
        }
    
    # Populate intent slice
    state["intent"] = {
        "issue_type": intent_data.get("issue_type", "other"),
        "severity": intent_data.get("severity", "low"),
        "SLA_risk": intent_data.get("SLA_risk", False),
        "safety_flags": intent_data.get("safety_flags", []),
        "reasoning": intent_data.get("reasoning", "")
    }
    
    # Add CoT trace entry
    if "cot_trace" not in state:
        state["cot_trace"] = []
    state["cot_trace"].append({
        "phase": "intent_classification",
        "content": f"Classified as {state['intent']['issue_type']} (severity: {state['intent']['severity']}, SLA_risk: {state['intent']['SLA_risk']})"
    })
    
    return state
