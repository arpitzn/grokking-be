"""
Agent Responsibility:
- SINGLE AUTHORITY for confidence threshold enforcement
- SINGLE AUTHORITY for auto vs human routing decision
- Applies deterministic compliance checks
- Validates content safety (via middleware + node)
- Evaluates tool failure criticality and makes escalation decision
- Uses LLM reasoning to make intelligent routing decisions
- Does NOT generate responses

IMPORTANT: This is the ONLY agent that:
1. Uses LLM reasoning to make routing decisions (agentic behavior)
2. Makes final routing decision (auto vs human)
3. Can trigger escalation based on tool failures
"""

from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from app.agent.state import AgentState, emit_phase_event
from app.models.tool_spec import ToolCriticality


class GuardrailsDecision(BaseModel):
    """Structured output for guardrails routing decision"""
    routing_decision: Literal["auto", "human"] = Field(
        ...,
        description="Final routing decision: auto for automated response, human for escalation"
    )
    routing_reason: str = Field(
        ...,
        description="Clear explanation of why this routing decision was made"
    )
    risk_assessment: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Overall risk level of handling this case automatically"
    )
    confidence_in_decision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident the guardrails agent is in this routing decision"
    )
    key_factors: List[str] = Field(
        ...,
        description="Top 3-5 factors that influenced this decision"
    )

# Note: Confidence thresholds are now determined agentically by the LLM
# These constants are kept for backward compatibility but are no longer used
CONFIDENCE_THRESHOLD_AUTO = 0.85  # Legacy: Auto-route if confidence >= 0.85
CONFIDENCE_THRESHOLD_HUMAN = 0.85  # Legacy: Escalate if confidence < 0.85


def run_compliance_checks(state: AgentState) -> Dict[str, Any]:
    """
    Run deterministic compliance checks.
    Returns: {"passed": bool, "checks": List[Dict]}
    """
    intent = state.get("intent", {})
    case = state.get("case", {})
    evidence = state.get("evidence", {})
    plan = state.get("plan", {})
    
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
    
    # Check 2: Policy compliance (only if policy retrieval was attempted)
    # Policy evidence is only required if policy_rag was in the plan's agents_to_activate
    # For greetings and simple intents that skip retrieval, this check should pass automatically
    agents_to_activate = plan.get("agents_to_activate", [])
    policy_retrieval_attempted = "policy_rag" in agents_to_activate
    
    if policy_retrieval_attempted:
        # Policy retrieval was attempted, so we need policy evidence
        policy_evidence = evidence.get("policy", [])
        policy_compliant = len(policy_evidence) > 0
        checks.append({
            "check": "policy_compliance",
            "passed": policy_compliant,
            "details": "Policy evidence must be available when policy retrieval was attempted"
        })
    else:
        # Policy retrieval was not attempted (e.g., greeting, simple query)
        # This is expected and compliant, so we skip the check or mark it as passed
        checks.append({
            "check": "policy_compliance",
            "passed": True,
            "details": "Policy retrieval not required for this intent type"
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
    Agentic Guardrails node: Uses LLM to make intelligent routing decisions.
    
    Input: analysis, plan, evidence, intent
    Output: guardrails (routing_decision, routing_reason, risk_assessment)
    """
    # 1. Run deterministic compliance checks (keep these - they're facts)
    compliance_result = run_compliance_checks(state)
    
    # 2. Validate content safety (keep this - it's a fact)
    safety_result = validate_content_safety(state)
    
    # 3. Evaluate tool failure criticality (keep this - it's a fact)
    critical_failures = evaluate_tool_failures(state)
    
    # 4. Gather context for LLM reasoning
    intent = state.get("intent", {})
    analysis = state.get("analysis", {})
    plan = state.get("plan", {})
    confidence_scores = state.get("confidence_scores", {})
    
    overall_confidence = confidence_scores.get("overall", analysis.get("confidence", 0.0))
    reasoning_confidence = confidence_scores.get("reasoning", analysis.get("confidence", 0.0))
    
    hypotheses = analysis.get("hypotheses", [])
    action_candidates = analysis.get("action_candidates", [])
    top_hypothesis = hypotheses[0] if hypotheses else {}
    top_action = action_candidates[0] if action_candidates else {}
    
    # 5. Build prompt context
    from app.infra.prompts import get_prompts
    from app.infra.llm import get_llm_service, get_expensive_model
    
    system_prompt, user_prompt = get_prompts(
        "guardrails_agent",
        {
            "issue_type": intent.get("issue_type", "unknown"),
            "severity": intent.get("severity", "low"),
            "overall_confidence": f"{overall_confidence:.2f}",
            "reasoning_confidence": f"{reasoning_confidence:.2f}",
            "evidence_quality": analysis.get("evidence_quality", "unknown"),
            "needs_more_data": str(analysis.get("needs_more_data", False)),
            "safety_flags": str(safety_result.get("safety_flags", [])),
            "compliance_checks": str(compliance_result.get("checks", [])),
            "critical_failures": str(critical_failures),
            "top_hypothesis": top_hypothesis.get("hypothesis", "None"),
            "hypothesis_confidence": f"{top_hypothesis.get('confidence', 0.0):.2f}",
            "recommended_action": top_action.get("action", "unknown"),
            "gaps": str(analysis.get("gaps", [])),
            "planner_advisory": plan.get("initial_route", "auto")
        }
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 6. Call LLM with structured output
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_expensive_model(),
        schema=GuardrailsDecision,
        temperature=0  # Deterministic for safety decisions
    )
    
    lc_messages = llm_service.convert_messages(messages)
    decision: GuardrailsDecision = await llm.ainvoke(lc_messages)
    
    # 7. Populate state with LLM decision
    state["guardrails"] = {
        "compliance_result": compliance_result,
        "safety_result": safety_result,
        "critical_failures": critical_failures,
        "routing_decision": decision.routing_decision,  # FINAL routing decision
        "routing_reason": decision.routing_reason,  # Reason for routing decision
        "risk_assessment": decision.risk_assessment,  # Risk level assessment
        "confidence_in_decision": decision.confidence_in_decision,  # Confidence in decision
        "key_factors": decision.key_factors,  # Key factors influencing decision
        "overall_confidence": overall_confidence,  # Overall confidence score used
        "needs_more_data": analysis.get("needs_more_data", False),  # Flag from reasoning agent
        "severity": intent.get("severity", "low"),  # Store severity for observability
    }
    
    # 8. Emit phase event
    emit_phase_event(
        state,
        "guardrails",
        f"Routing decision: {decision.routing_decision} ({decision.routing_reason})",
        metadata={
            "risk_assessment": decision.risk_assessment,
            "confidence_in_decision": decision.confidence_in_decision,
            "key_factors": decision.key_factors,
            "overall_confidence": overall_confidence,
            "compliance": compliance_result,
            "safety_flags": safety_result.get("safety_flags", [])
        }
    )
    
    return state
