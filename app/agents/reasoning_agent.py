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
from app.infra.llm import get_llm_service, get_expensive_model


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
    recommended_next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested next steps (e.g., 'fetch_restaurant_logs', 'escalate_to_human')"
    )
    reasoning_trace: str = Field(
        ...,
        description="Step-by-step reasoning process"
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
    
    # Collect all evidence
    mongo_evidence = evidence.get("mongo", [])
    policy_evidence = evidence.get("policy", [])
    memory_evidence = evidence.get("memory", [])
    
    # Count evidence items
    total_evidence = len(mongo_evidence) + len(policy_evidence) + len(memory_evidence)
    
    # Build prompt for reasoning with self-reflection
    prompt = f"""You are a reasoning agent for a food delivery support system. Analyze the evidence and provide structured analysis with self-reflection.

Case Context:
- Issue Type: {intent.get('issue_type', 'unknown')}
- Severity: {intent.get('severity', 'low')}
- SLA Risk: {intent.get('SLA_risk', False)}
- Order ID: {case.get('order_id', 'N/A')}
- Customer ID: {case.get('customer_id', 'N/A')}

Evidence from MongoDB ({len(mongo_evidence)} items):
{_format_evidence(mongo_evidence)}

Evidence from Policies ({len(policy_evidence)} items):
{_format_evidence(policy_evidence)}

Evidence from Memory ({len(memory_evidence)} items):
{_format_evidence(memory_evidence)}

Analyze the evidence and provide:

1. **hypotheses**: Top 3-5 hypotheses about what happened, ranked by confidence
   - Each hypothesis should have: hypothesis text, confidence (0-1), evidence sources

2. **action_candidates**: Recommended actions with confidence and rationale
   - Examples: "issue_refund", "apologize_and_explain", "escalate_to_human", "request_more_info"

3. **confidence**: Overall confidence in your analysis (0-1)

4. **gaps**: What information is missing that would improve your analysis?

5. **Self-Reflection** (CRITICAL):
   - evidence_quality: Rate the quality ("high", "medium", "low")
     * high: Complete, consistent, from multiple sources
     * medium: Partial coverage, some gaps
     * low: Sparse, contradictory, or unreliable
   
   - conflicting_evidence: List any contradictions you found
     * Example: "Order timeline shows delivered, but customer says not received"
   
   - needs_more_data: Do you need more information? (true/false)
     * true if: Low evidence quality, high gaps, conflicting data
     * false if: Sufficient evidence for confident decision
   
   - recommended_next_steps: What should happen next?
     * Examples: ["escalate_to_human"], ["fetch_delivery_photos"], ["auto_respond"]
   
   - reasoning_trace: Explain your step-by-step reasoning process

Guidelines:
- Be honest about uncertainty - low confidence is better than false confidence
- Flag conflicts explicitly - don't ignore contradictions
- If evidence is weak, recommend escalation or more data gathering
- Consider policy compliance in your action recommendations
"""
    
    messages = [
        {
            "role": "system", 
            "content": "You are a reasoning agent with self-reflection capabilities. Analyze evidence critically and honestly assess your confidence and limitations."
        },
        {"role": "user", "content": prompt}
    ]
    
    # Use expensive model for reasoning with structured output
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_expensive_model(),
        schema=ReasoningOutput,
        temperature=0.5  # Moderate temperature for creative reasoning
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
        "needs_more_data": response.needs_more_data,
        "recommended_next_steps": response.recommended_next_steps,
        "reasoning_trace": response.reasoning_trace
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
    
    return state
