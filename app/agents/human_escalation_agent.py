"""
Agent Responsibility:
- Creates handover packet for human agents
- Calls POST /escalations endpoint
- Includes all evidence and analysis
- Does NOT generate responses
"""

import httpx
from typing import Dict, Any

from app.agent.state import AgentState, emit_phase_event
from app.infra.config import settings


async def human_escalation_node(state: AgentState) -> AgentState:
    """
    Human escalation node: Creates handover packet and calls escalation endpoint.
    
    Input: case, intent, evidence, analysis, guardrails
    Output: handover_packet
    """
    case = state.get("case", {})
    intent = state.get("intent", {})
    evidence = state.get("evidence", {})
    analysis = state.get("analysis", {})
    guardrails = state.get("guardrails", {})
    
    # Create handover packet
    handover_packet = {
        "case_id": case.get("conversation_id", ""),
        "customer_id": case.get("customer_id", ""),
        "order_id": case.get("order_id"),
        "issue_type": intent.get("issue_type", "unknown"),
        "severity": intent.get("severity", "low"),
        "SLA_risk": intent.get("SLA_risk", False),
        "safety_flags": intent.get("safety_flags", []),
        "evidence_summary": {
            "mongo_count": len(evidence.get("mongo", [])),
            "policy_count": len(evidence.get("policy", [])),
            "memory_count": len(evidence.get("memory", []))
        },
        "analysis": {
            "confidence": analysis.get("confidence", 0.0),
            "top_hypothesis": analysis.get("hypotheses", [{}])[0] if analysis.get("hypotheses") else {},
            "gaps": analysis.get("gaps", [])
        },
        "guardrails": {
            "compliance_passed": guardrails.get("compliance_result", {}).get("passed", False),
            "safety_passed": guardrails.get("safety_result", {}).get("passed", False),
            "critical_failures": guardrails.get("critical_failures", []),
            "routing_reason": "Low confidence" if not guardrails.get("confidence_gate_passed") else "Other"
        },
        "raw_text": case.get("raw_text", ""),
        "events": state.get("events", [])  # Changed from cot_trace
    }
    
    # Call POST /escalations endpoint
    try:
        # Get base URL from settings (default to localhost)
        base_url = getattr(settings, "api_base_url", "http://localhost:8000")
        escalation_url = f"{base_url}/api/escalations"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                escalation_url,
                json=handover_packet
            )
            response.raise_for_status()
            escalation_result = response.json()
            handover_packet["escalation_id"] = escalation_result.get("escalation_id")
    except Exception as e:
        # Log error but don't fail - handover packet is still created
        handover_packet["escalation_error"] = str(e)
    
    # Populate handover_packet
    state["handover_packet"] = handover_packet
    
    # Emit phase event
    emit_phase_event(state, "human_escalation", f"Escalating {intent.get('issue_type', 'unknown')} issue to human agent")
    
    return state
