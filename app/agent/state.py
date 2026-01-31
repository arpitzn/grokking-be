"""Agent state definition for LangGraph - Food Delivery Domain"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """
    Deep merge dicts for concurrent updates from parallel nodes.
    
    Used by parallel retrieval subgraphs to safely merge evidence and status updates.
    """
    result = {**left}
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_dicts(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # Concatenate lists
            result[key] = result[key] + value
        else:
            # Override with new value
            result[key] = value
    return result


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
    
    # Input slice - PARALLEL UPDATES (subgraphs return full state)
    case: Annotated[Dict[str, Any], merge_dicts]  # persona, channel, order_id, customer_id, zone_id, raw_text, locale, conversation_id, user_id
    
    # Interpretation slice - PARALLEL UPDATES (subgraphs return full state)
    intent: Annotated[Dict[str, Any], merge_dicts]  # issue_type, severity, SLA_risk, safety_flags, reasoning
    plan: Annotated[Dict[str, Any], merge_dicts]  # retrieval_plan, tool_selection, initial_route (advisory)
    
    # Evidence slice - PARALLEL UPDATES from 3 retrieval subgraphs
    evidence: Annotated[Dict[str, List[Dict]], merge_dicts]  # mongo[], policy[], memory[]
    retrieval_status: Annotated[Dict[str, Any], merge_dicts]  # tracks completion/failure of parallel retrievals
    
    # Decision slice
    analysis: Dict[str, Any]  # hypotheses[], action_candidates[], confidence, gaps
    guardrails: Dict[str, Any]  # compliance_result, routing_decision (auto/human - FINAL), confidence_gate_result
    
    # Outputs
    final_response: str
    handover_packet: Optional[Dict[str, Any]]
    
    # Working Memory Management
    working_memory: List[Dict[str, str]]  # Last N=10 messages (trimmed + summarized view)
    conversation_summary: Optional[str]  # Periodic async summary (stored in MongoDB)
    
    # Multi-turn conversation support
    conversation_history: List[Dict[str, str]]  # Last 5 messages (verbatim)
    turn_number: int  # Current turn in conversation
    
    # Observability - PARALLEL UPDATES from all nodes
    trace_events: Annotated[List[Dict[str, Any]], add]
    cot_trace: Annotated[List[Dict[str, str]], add]  # Hybrid: simple by default, rich when needed
    
    # Internal subgraph state - used by parallel retrieval subgraphs for LLM iterations
    messages: Annotated[List, add]  # LangChain messages for agentic tool calling
