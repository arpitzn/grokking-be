"""
Agent Responsibility:
- SINGLE AUTHORITY for confidence threshold enforcement
- SINGLE AUTHORITY for auto vs human routing decision
- Applies deterministic compliance checks
- Validates content safety (via middleware + node)
- Evaluates tool failure criticality and makes escalation decision
- Does NOT generate responses

IMPORTANT: This is the ONLY agent that:
1. Enforces confidence gating (checks analysis.confidence against hardcoded thresholds)
2. Makes final routing decision (auto vs human)
3. Can trigger escalation based on tool failures
"""

from typing import Dict, Any, List

from app.agent.state import AgentState, emit_phase_event
from app.models.tool_spec import ToolCriticality

# Hardcoded confidence thresholds (owned by Guardrails Agent)
CONFIDENCE_THRESHOLD_AUTO = 0.85  # Auto-route if confidence >= 0.85
CONFIDENCE_THRESHOLD_HUMAN = 0.85  # Escalate if confidence < 0.85


def run_compliance_checks(state: AgentState) -> Dict[str, Any]:
    """
    Run deterministic compliance checks.
    Returns: {"passed": bool, "checks": List[Dict]}
    """
    intent = state.get("intent", {})
    case = state.get("case", {})
    evidence = state.get("evidence", {})
    
    checks = []
    
    # Check 1: Refund eligibility (if refund issue)
    if intent.get("issue_type") == "refund":
        order_evidence = evidence.get("mongo", [])
        # Simple check: if order exists and is delivered, eligible
        eligible = any(
            ev.get("data", {}).get("status") == "delivered"
            for ev in order_evidence
            if ev.get("source") == "mongo"
        )
        checks.append({
            "check": "refund_eligibility",
            "passed": eligible,
            "details": "Order must be delivered for refund eligibility"
        })
    
    # Check 2: Policy compliance (if policy evidence exists)
    policy_evidence = evidence.get("policy", [])
    policy_compliant = len(policy_evidence) > 0
    checks.append({
        "check": "policy_compliance",
        "passed": policy_compliant,
        "details": "Policy evidence must be available"
    })
    
    # Overall compliance result
    passed = all(check["passed"] for check in checks)
    
    return {
        "passed": passed,
        "checks": checks
    }


def validate_content_safety(state: AgentState) -> Dict[str, Any]:
    """
    Validate content safety (simplified for hackathon).
    In production, this would use NeMo Guardrails middleware.
    """
    intent = state.get("intent", {})
    safety_flags = intent.get("safety_flags", [])
    
    # If safety flags exist, fail safety check
    passed = len(safety_flags) == 0
    
    return {
        "passed": passed,
        "safety_flags": safety_flags
    }


def evaluate_tool_failures(state: AgentState) -> List[str]:
    """
    Evaluates tool failures from evidence (not retrieval_status).
    Returns list of critical failures that require escalation.
    """
    critical_failures = []
    evidence = state.get("evidence", {})
    
    # Check evidence for failed tools
    for source in ["mongo", "policy", "memory"]:
        for ev in evidence.get(source, []):
            status = ev.get("tool_result", {}).get("status", "unknown")
            if status == "failed":
                tool_name = ev.get("provenance", {}).get("tool", "unknown")
                # Check if tool is critical (order, customer, policy)
                if any(keyword in tool_name for keyword in ["order", "customer", "policy"]):
                    critical_failures.append(tool_name)
    
    return critical_failures


async def guardrails_node(state: AgentState) -> AgentState:
    """
    Guardrails node: SINGLE AUTHORITY for confidence gating and routing.
    
    Input: analysis, plan, evidence
    Output: guardrails (compliance_result, routing_decision FINAL, confidence_gate_result)
    """
    # 1. Run deterministic compliance checks
    compliance_result = run_compliance_checks(state)
    
    # 2. Validate content safety
    safety_result = validate_content_safety(state)
    
    # 3. Evaluate tool failure criticality
    critical_failures = evaluate_tool_failures(state)
    
    # 4. CONFIDENCE GATING (single authority)
    # Use overall confidence from confidence_scores if available, otherwise fall back to analysis confidence
    confidence_scores = state.get("confidence_scores", {})
    overall_confidence = confidence_scores.get("overall", state.get("analysis", {}).get("confidence", 0.0))
    analysis_confidence = state.get("analysis", {}).get("confidence", 0.0)
    
    # Use the lower threshold (0.7) for routing decisions as per plan
    CONFIDENCE_THRESHOLD_ROUTING = 0.7
    confidence_gate_passed = overall_confidence >= CONFIDENCE_THRESHOLD_ROUTING
    
    # Check if reasoning agent needs more data
    analysis = state.get("analysis", {})
    needs_more_data = analysis.get("needs_more_data", False)
    
    # 5. ROUTING DECISION (single authority, may override Planner's advisory)
    initial_route = state.get("plan", {}).get("initial_route", "auto")  # Planner's advisory
    
    # Guardrails has FINAL authority and may override
    if not compliance_result["passed"]:
        routing_decision = "human"  # Compliance failure (override)
        routing_reason = "Compliance check failed"
    elif not safety_result["passed"]:
        routing_decision = "human"  # Safety failure (override)
        routing_reason = "Safety check failed"
    elif critical_failures:
        routing_decision = "human"  # Critical tool failure (override)
        routing_reason = f"Critical tool failures: {', '.join(critical_failures)}"
    elif needs_more_data:
        routing_decision = "human"  # Needs more data (override)
        routing_reason = "Reasoning agent indicates more data needed"
    elif not confidence_gate_passed:
        routing_decision = "human"  # Low confidence (override)
        routing_reason = f"Low confidence: {overall_confidence:.2f} < {CONFIDENCE_THRESHOLD_ROUTING}"
    else:
        routing_decision = initial_route  # Accept Planner's advisory if all checks passed
        routing_reason = f"All checks passed (confidence: {overall_confidence:.2f})"
    
    # Populate state["guardrails"]
    state["guardrails"] = {
        "compliance_result": compliance_result,
        "safety_result": safety_result,
        "critical_failures": critical_failures,
        "confidence_gate_passed": confidence_gate_passed,
        "routing_decision": routing_decision,  # FINAL routing decision
        "routing_reason": routing_reason,  # Reason for routing decision
        "overall_confidence": overall_confidence,  # Overall confidence score used
        "needs_more_data": needs_more_data,  # Flag from reasoning agent
    }
    
    # Emit phase event
    emit_phase_event(
        state,
        "guardrails",
        f"Routing decision: {routing_decision} ({routing_reason})",
        metadata={
            "compliance": compliance_result,
            "confidence": overall_confidence,
            "needs_more_data": needs_more_data,
            "confidence_threshold": CONFIDENCE_THRESHOLD_ROUTING
        }
    )
    
    return state
