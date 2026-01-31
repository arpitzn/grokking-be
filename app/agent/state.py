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


def take_right(left: Any, right: Any) -> Any:
    """
    Reducer that takes the rightmost (latest) value.
    Used for fields that should be overwritten, not merged, when updated concurrently.
    """
    return right


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
    
    # Decision slice - may be updated concurrently if reasoning runs multiple times
    analysis: Annotated[Dict[str, Any], take_right]  # hypotheses[], action_candidates[], confidence, gaps
    guardrails: Annotated[Dict[str, Any], take_right]  # compliance_result, routing_decision (auto/human - FINAL), confidence_gate_result
    
    # Outputs - may be updated by response_synthesis or human_escalation
    final_response: Annotated[str, take_right]
    handover_packet: Annotated[Optional[Dict[str, Any]], take_right]
    
    # Working Memory Management
    working_memory: Annotated[List[Dict[str, str]], take_right]  # Last N=10 messages (trimmed + summarized view)
    conversation_summary: Annotated[Optional[str], take_right]  # Periodic async summary (stored in MongoDB)
    
    # Multi-turn conversation support
    conversation_history: Annotated[List[Dict[str, str]], take_right]  # Last 5 messages (verbatim)
    turn_number: Annotated[int, take_right]  # Current turn in conversation
    
    # Observability - PARALLEL UPDATES from all nodes
    trace_events: Annotated[List[Dict[str, Any]], add]
    cot_trace: Annotated[List[Dict[str, Any]], add]  # turn can be int, phase/content are str
    
    # Internal subgraph state - used by parallel retrieval subgraphs for LLM iterations
    messages: Annotated[List, add]  # LangChain messages for agentic tool calling
