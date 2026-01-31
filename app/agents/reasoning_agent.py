"""
Agent Responsibility:
- Fuses evidence from all sources
- Generates top N hypotheses with confidence scores
- Creates action candidates
- Identifies knowledge gaps
- Self-reflects on evidence quality and conflicts
- Does NOT make routing decisions or generate final responses
"""

import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_expensive_model, get_cheap_model
from app.infra.prompts import get_prompts


class Hypothesis(BaseModel):
    """Single hypothesis with evidence"""
    hypothesis: str = Field(..., description="The hypothesis statement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence: List[str] = Field(..., description="Evidence sources supporting this hypothesis")


class ActionCandidate(BaseModel):
    """Possible action with rationale"""
    action: str = Field(..., description="Action to take (e.g., 'issue_refund', 'escalate_to_human')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this action")
    rationale: str = Field(..., description="Why this action is recommended")


class ReasoningOutput(BaseModel):
    """Structured output for reasoning agent with self-reflection"""
    hypotheses: List[Hypothesis] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="Top 3-5 hypotheses ranked by confidence"
    )
    action_candidates: List[ActionCandidate] = Field(
        ...,
        description="Recommended actions with rationale"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in analysis"
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Missing information that would improve analysis"
    )
    
    # Self-reflection fields
    evidence_quality: Literal["high", "medium", "low"] = Field(
        ...,
        description="Assessment of evidence quality and completeness"
    )
    conflicting_evidence: List[str] = Field(
        default_factory=list,
        description="Any contradictions or conflicts in the evidence"
    )
    needs_more_data: bool = Field(
        ...,
        description="True if more information is needed for confident decision"
    )


def _format_evidence(evidence_list: List[Dict[str, Any]]) -> str:
    """Format evidence list for prompt"""
    if not evidence_list:
        return "(No evidence found)"
    
    return json.dumps(evidence_list, indent=2)


async def reasoning_node(state: AgentState) -> AgentState:
    """
    Reasoning node: Fuses evidence, generates hypotheses, and self-reflects.
    
    Input: evidence (mongo[], policy[], memory[]), intent, case
    Output: analysis (hypotheses[], action_candidates[], confidence, gaps, self-reflection)
    """
    evidence = state.get("evidence", {})
    intent = state.get("intent", {})
    case = state.get("case", {})
    
    # Collect all evidence (even if empty for simple queries)
    mongo_evidence = evidence.get("mongo", [])
    policy_evidence = evidence.get("policy", [])
    memory_evidence = evidence.get("memory", [])
    
    # Count evidence items
    total_evidence = len(mongo_evidence) + len(policy_evidence) + len(memory_evidence)
    
    # Build execution messages from working memory + current turn
    messages = []
    
    # Add working memory for multi-turn context
    working_memory = state.get("working_memory", [])
    for msg in working_memory:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Get prompts from centralized prompts module for current turn
    system_prompt, user_prompt = get_prompts(
        "reasoning_agent",
        {
            "issue_type": intent.get('issue_type', 'unknown'),
            "severity": intent.get('severity', 'low'),
            "sla_risk": str(intent.get('SLA_risk', False)),
            "order_id": case.get('order_id', 'N/A'),
            "customer_id": case.get('customer_id', 'N/A'),
            "mongo_count": str(len(mongo_evidence)),
            "policy_count": str(len(policy_evidence)),
            "memory_count": str(len(memory_evidence)),
            "mongo_evidence": _format_evidence(mongo_evidence),
            "policy_evidence": _format_evidence(policy_evidence),
            "memory_evidence": _format_evidence(memory_evidence)
        }
    )
    
    # Add current turn prompts
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    # Model selection: Use cheap model for simple conversational queries, expensive for complex issues
    severity = intent.get('severity', 'low')
    issue_type = intent.get('issue_type', 'unknown')
    
    if severity == "low" and issue_type in ["greeting", "question", "acknowledgment", "clarification_request"]:
        model_name = get_cheap_model()
        temperature = 0.1  # More deterministic for simple cases
    else:
        model_name = get_expensive_model()
        temperature = 0.3  # More creative for complex cases
    
    # Use LLM reasoning for ALL cases (agentic behavior)
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=model_name,
        schema=ReasoningOutput,
        temperature=temperature
    )
    
    lc_messages = llm_service.convert_messages(messages)
    response: ReasoningOutput = await llm.ainvoke(lc_messages)
    
    # Populate analysis slice with self-reflection
    state["analysis"] = {
        "hypotheses": [h.model_dump() for h in response.hypotheses],
        "action_candidates": [a.model_dump() for a in response.action_candidates],
        "confidence": response.confidence,
        "gaps": response.gaps,
        # Self-reflection fields
        "evidence_quality": response.evidence_quality,
        "conflicting_evidence": response.conflicting_evidence,
        "needs_more_data": response.needs_more_data
    }
    
    # Update confidence tracking
    if "confidence_scores" not in state:
        state["confidence_scores"] = {}
    state["confidence_scores"]["reasoning"] = response.confidence
    
    # Calculate overall confidence (weighted average)
    ingestion_conf = state.get("confidence_scores", {}).get("ingestion", 1.0)
    intent_conf = state.get("confidence_scores", {}).get("intent_classification", 1.0)
    reasoning_conf = response.confidence
    
    # Weight: ingestion 20%, intent 30%, reasoning 50%
    overall_confidence = (ingestion_conf * 0.2) + (intent_conf * 0.3) + (reasoning_conf * 0.5)
    state["confidence_scores"]["overall"] = overall_confidence
    
    # Emit phase event with self-reflection summary
    quality_emoji = {"high": "✓", "medium": "~", "low": "⚠"}[response.evidence_quality]
    emit_phase_event(
        state,
        "reasoning",
        f"Generated {len(response.hypotheses)} hypotheses | Evidence: {quality_emoji} {response.evidence_quality} | Confidence: {response.confidence:.2f}",
        metadata={
            "hypothesis_count": len(response.hypotheses),
            "confidence": response.confidence,
            "evidence_quality": response.evidence_quality,
            "needs_more_data": response.needs_more_data,
            "conflicts": len(response.conflicting_evidence),
            "overall_confidence": overall_confidence
        }
    )
    
    # WRITE to state.messages for observability and multi-turn continuity
    # This is safe because reasoning runs sequentially (after parallel retrieval)
    return {
        "analysis": state["analysis"],
        "confidence_scores": state["confidence_scores"],
        "messages": lc_messages  # Write to state.messages (single-writer)
    }
