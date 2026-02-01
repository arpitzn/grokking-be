"""
Agent Responsibility:
- Creates escalated ticket in MongoDB support_tickets collection
- Ticket is queryable via GET /api/escalated-tickets/{user_id}
- Includes all evidence and analysis
- Does NOT generate responses
"""

import logging
from datetime import datetime
from typing import Dict, Any

from app.agent.state import AgentState, emit_phase_event
from app.infra.mongo import get_mongodb_client

logger = logging.getLogger(__name__)


async def human_escalation_node(state: AgentState) -> AgentState:
    """
    Human escalation node: Creates ticket in MongoDB support_tickets collection.
    
    Input: case, intent, evidence, analysis, guardrails
    Output: handover_packet with ticket_id
    """
    case = state.get("case", {})
    intent = state.get("intent", {})
    evidence = state.get("evidence", {})
    analysis = state.get("analysis", {})
    guardrails = state.get("guardrails", {})
    
    # Determine severity: 1=Critical (SLA risk or critical failures), 2=High (other escalations)
    has_critical_failures = bool(guardrails.get("critical_failures"))
    has_sla_risk = intent.get("SLA_risk", False)
    severity = 1 if (has_sla_risk or has_critical_failures) else 2
    
    # Determine routing reason
    routing_reason = guardrails.get("guardrails", {}).get("routing_reason", "Low confidence")
    if not guardrails.get("confidence_gate_passed"):
        routing_reason = "Low confidence"
    
    # Create ticket document matching escalated_tickets.py schema
    ticket_doc = {
        "ticket_type": "complaint",  # Required for escalated_tickets query
        "severity": severity,  # 1=Critical, 2=High
        "issue_type": intent.get("issue_type", "unknown"),
        "subtype": intent.get("subtype"),
        "scope": "user",  # Default to user scope
        "user_id": case.get("user_id"),
        "order_id": case.get("order_id"),
        "restaurant_id": case.get("restaurant_id"),
        "affected_zones": [],
        "affected_city": None,
        "title": f"{intent.get('issue_type', 'Issue')} - Case {case.get('conversation_id', '')}",
        "description": case.get("raw_text", ""),
        "status": "open",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "timestamp": datetime.utcnow(),
        "related_orders": [case.get("order_id")] if case.get("order_id") else [],
        "related_tickets": [],
        "agent_notes": [
            {
                "note": f"Auto-escalated: {routing_reason}",
                "created_at": datetime.utcnow(),
                "created_by": "system"
            }
        ],
        "resolution_history": [],
        "resolution": None,
        # Additional metadata for agent context
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
            "routing_reason": routing_reason
        },
        "safety_flags": intent.get("safety_flags", []),
        "SLA_risk": has_sla_risk,
        "events": state.get("events", [])
    }
    
    # Write to MongoDB
    handover_packet = {}
    try:
        db = await get_mongodb_client()
        result = await db.support_tickets.insert_one(ticket_doc)
        ticket_id = str(result.inserted_id)
        
        logger.info(
            f"Escalated ticket created: {ticket_id}",
            extra={
                "ticket_id": ticket_id,
                "user_id": case.get("user_id"),
                "issue_type": intent.get("issue_type"),
                "severity": severity,
                "SLA_risk": has_sla_risk
            }
        )
        
        handover_packet = {
            "ticket_id": ticket_id,
            "status": "created",
            "user_id": case.get("user_id"),
            "issue_type": intent.get("issue_type"),
            "severity": severity,
            "routing_reason": routing_reason
        }
    except Exception as e:
        logger.error(f"Failed to create escalated ticket: {str(e)}", exc_info=True)
        handover_packet = {
            "status": "failed",
            "error": str(e),
            "user_id": case.get("user_id"),
            "issue_type": intent.get("issue_type")
        }
    
    # Populate handover_packet in state
    state["handover_packet"] = handover_packet
    
    # Emit phase event
    ticket_id = handover_packet.get("ticket_id", "unknown")
    emit_phase_event(
        state, 
        "human_escalation", 
        f"Created escalated ticket {ticket_id} for {intent.get('issue_type', 'unknown')} issue"
    )
    
    return state
