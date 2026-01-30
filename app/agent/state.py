"""Agent state definition for LangGraph - Food Delivery Domain"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """
    Comprehensive state schema for food delivery agentic operations co-pilot.
    
    State slices:
    - Input: case (persona, channel, order_id, customer_id, zone_id, raw_text, locale)
    - Interpretation: intent (issue_type, severity, SLA_risk, safety_flags), plan (tool_selection, initial_route)
    - Evidence: evidence (mongo[], policy[], memory[]), retrieval_status (completion tracking)
    - Decision: analysis (hypotheses[], action_candidates[], confidence, gaps), guardrails (compliance, routing_decision FINAL)
    - Outputs: final_response, handover_packet
    - Working Memory: working_memory (last 10 messages), conversation_summary (async summaries)
    - Observability: trace_events, cot_trace
    """
    
    # Input slice
    case: Dict[str, Any]  # persona, channel, order_id, customer_id, zone_id, raw_text, locale, conversation_id, user_id
    
    # Interpretation slice
    intent: Dict[str, Any]  # issue_type, severity, SLA_risk, safety_flags, reasoning
    plan: Dict[str, Any]  # retrieval_plan, tool_selection, initial_route (advisory)
    
    # Evidence slice
    evidence: Dict[str, List[Dict]]  # mongo[], policy[], memory[]
    retrieval_status: Dict[str, Any]  # tracks completion/failure of parallel retrievals
    
    # Decision slice
    analysis: Dict[str, Any]  # hypotheses[], action_candidates[], confidence, gaps
    guardrails: Dict[str, Any]  # compliance_result, routing_decision (auto/human - FINAL), confidence_gate_result
    
    # Outputs
    final_response: str
    handover_packet: Optional[Dict[str, Any]]
    
    # Working Memory Management
    working_memory: List[Dict[str, str]]  # Last N=10 messages (trimmed + summarized view)
    conversation_summary: Optional[str]  # Periodic async summary (stored in MongoDB)
    
    # Observability
    trace_events: List[Dict[str, Any]]
    cot_trace: List[Dict[str, str]]  # Hybrid: simple by default, rich when needed
