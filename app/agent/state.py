"""Agent state definition for LangGraph - Food Delivery Domain"""

from enum import Enum
from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# Forward reference to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import CaseRequest


class EventClass(str, Enum):
    """Event classification for UI filtering"""
    USER = "user"              # Final output, escalations, status
    EXPLAINABILITY = "explainability"  # CoT summaries, evidence, hypotheses
    DEBUG = "debug"            # Tool calls, internal state changes


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


# Subgraph state schemas with input/output isolation
# Each retrieval subgraph has private messages for tool-calling (not returned to parent)

class MongoRetrievalInputState(TypedDict):
    """Input schema for mongo retrieval subgraph"""
    case: Dict[str, Any]  # Read-only: order_id, customer_id, etc.
    intent: Dict[str, Any]  # Read-only: issue_type, severity, SLA_risk
    plan: Dict[str, Any]  # Read-only: retrieval_instructions, agents_to_activate
    evidence: Dict[str, List[Dict]]  # Shared evidence accumulator


class MongoRetrievalOutputState(TypedDict):
    """Output schema - only returns evidence"""
    evidence: Dict[str, List[Dict]]  # Only writes to evidence.mongo


class MongoRetrievalState(MongoRetrievalInputState, MongoRetrievalOutputState):
    """Internal state with private messages for tool calling"""
    messages: Annotated[List, add]  # Private to this subgraph (not in output schema)
    events: Annotated[List[Dict[str, Any]], add]  # For phase events


class PolicyRetrievalInputState(TypedDict):
    """Input schema for policy retrieval subgraph"""
    case: Dict[str, Any]  # Read-only: order_id, customer_id, etc.
    intent: Dict[str, Any]  # Read-only: issue_type, severity, SLA_risk
    plan: Dict[str, Any]  # Read-only: retrieval_instructions, agents_to_activate
    evidence: Dict[str, List[Dict]]  # Shared evidence accumulator


class PolicyRetrievalOutputState(TypedDict):
    """Output schema - only returns evidence"""
    evidence: Dict[str, List[Dict]]  # Only writes to evidence.policy


class PolicyRetrievalState(PolicyRetrievalInputState, PolicyRetrievalOutputState):
    """Internal state with private messages for tool calling"""
    messages: Annotated[List, add]  # Private to this subgraph (not in output schema)
    events: Annotated[List[Dict[str, Any]], add]  # For phase events


class MemoryRetrievalInputState(TypedDict):
    """Input schema for memory retrieval subgraph"""
    case: Dict[str, Any]  # Read-only: order_id, customer_id, etc.
    intent: Dict[str, Any]  # Read-only: issue_type, severity, SLA_risk
    plan: Dict[str, Any]  # Read-only: retrieval_instructions, agents_to_activate
    evidence: Dict[str, List[Dict]]  # Shared evidence accumulator


class MemoryRetrievalOutputState(TypedDict):
    """Output schema - only returns evidence"""
    evidence: Dict[str, List[Dict]]  # Only writes to evidence.memory


class MemoryRetrievalState(MemoryRetrievalInputState, MemoryRetrievalOutputState):
    """Internal state with private messages for tool calling"""
    messages: Annotated[List, add]  # Private to this subgraph (not in output schema)
    events: Annotated[List[Dict[str, Any]], add]  # For phase events


class AgentState(TypedDict):
    """
    Comprehensive state schema for food delivery agentic operations co-pilot.
    
    State slices:
    - Input: case (persona, channel, order_id, customer_id, zone_id, raw_text, locale)
    - Interpretation: intent (issue_type, severity, SLA_risk, safety_flags), plan (agents_to_activate, initial_route)
    - Evidence: evidence (mongo[], policy[], memory[])
    - Decision: analysis (hypotheses[], action_candidates[], confidence, gaps), guardrails (compliance, routing_decision FINAL)
    - Outputs: final_response, handover_packet
    - Working Memory: working_memory (summary + recent messages)
    - Observability: events (unified event stream), phase_status (phase completion tracking)
    """
    
    # Input slice - PARALLEL UPDATES (subgraphs return full state)
    case: Annotated[Dict[str, Any], merge_dicts]  # persona, channel, order_id, user_id, zone_id, raw_text, locale, conversation_id
    
    # Interpretation slice - PARALLEL UPDATES (subgraphs return full state)
    intent: Annotated[Dict[str, Any], merge_dicts]  # issue_type, severity, SLA_risk, safety_flags
    plan: Annotated[Dict[str, Any], merge_dicts]  # agents_to_activate, initial_route (advisory)
    
    # Evidence slice - PARALLEL UPDATES from 3 retrieval subgraphs
    evidence: Annotated[Dict[str, List[Dict]], merge_dicts]  # mongo[], policy[], memory[]
    
    # Decision slice - may be updated concurrently if reasoning runs multiple times
    analysis: Annotated[Dict[str, Any], take_right]  # hypotheses[], action_candidates[], confidence, gaps, self-reflection fields
    guardrails: Annotated[Dict[str, Any], take_right]  # compliance_result, routing_decision (auto/human - FINAL), confidence_gate_result
    
    # Confidence tracking - PARALLEL UPDATES from all agents
    confidence_scores: Annotated[Dict[str, float], merge_dicts]  # Per-agent confidence scores (ingestion, intent_classification, reasoning, overall)
    
    # Outputs - may be updated by response_synthesis or human_escalation
    final_response: Annotated[str, take_right]
    handover_packet: Annotated[Optional[Dict[str, Any]], take_right]
    
    # Working Memory Management (single source of truth)
    working_memory: Annotated[List[Dict[str, str]], take_right]  # Summary + recent messages
    
    # Observability - PARALLEL UPDATES from all nodes
    events: Annotated[List[Dict[str, Any]], add]  # Unified event stream (replaces trace_events and cot_trace)
    phase_status: Annotated[Dict[str, str], take_right]  # Track phase completion for summary generation
    
    # Per-turn execution buffer - SINGLE WRITER (only reasoning/synthesis nodes write)
    # Retrieval subgraphs have private messages (not in parent state)
    # Multi-turn continuity preserved via working_memory
    messages: Annotated[List, take_right]  # LangChain messages for reasoning/synthesis (per-turn buffer)


def emit_phase_event(
    state: AgentState, 
    phase: str, 
    content: str,
    event_class: str = "explainability",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Centralized event emission - prevents code duplication across agents.
    
    Args:
        state: Current agent state
        phase: Phase name (ingestion, planning, searching, reasoning, etc.)
        content: Human-readable summary of what happened
        event_class: Classification (user, explainability, debug)
        metadata: Optional additional data (tool counts, evidence counts, etc.)
    
    Example:
        emit_phase_event(state, "planning", 
            "Selected 3 retrieval agents for evidence gathering",
            metadata={"agents": ["mongo", "policy", "memory"]})
    """
    # Derive turn number from working memory length (approximate)
    working_memory = state.get("working_memory", [])
    conversation_messages = [m for m in working_memory if m.get("role") != "system"]
    turn = (len(conversation_messages) // 2) + 1 if conversation_messages else 1
    
    if "events" not in state:
        state["events"] = []
    
    event = {
        "phase": phase,
        "turn": turn,
        "content": content,
        "class": event_class,
        "timestamp": None  # Set by streamer
    }
    
    if metadata:
        event["metadata"] = metadata
    
    state["events"].append(event)
    
    # Track phase completion for summary generation
    if "phase_status" not in state:
        state["phase_status"] = {}
    state["phase_status"][phase] = "completed"


def create_initial_state(
    request: "CaseRequest", 
    conversation_id: str,
    working_memory: Optional[List[Dict[str, str]]] = None
) -> AgentState:
    """
    Create initial AgentState with working memory.
    
    Args:
        request: CaseRequest containing user message and metadata
        conversation_id: Conversation identifier
        working_memory: Pre-built working memory (summary + recent messages)
        
    Returns:
        Initialized AgentState with conversation context
    """
    return {
        "case": {
            "persona": request.persona or "customer",
            "channel": request.channel or "web",
            "raw_text": request.message,
            "user_id": request.user_id,
            "conversation_id": conversation_id,
            "order_id": None,
            "zone_id": None,
            "restaurant_id": None,
            "locale": "en-US",
        },
        "intent": {},
        "plan": {},
        "evidence": {},
        "analysis": {},
        "guardrails": {},
        "final_response": "",
        "handover_packet": None,
        "working_memory": working_memory or [],
        "events": [],
        "phase_status": {},
        "confidence_scores": {},  # Initialize confidence tracking
        "messages": [],  # Internal subgraph state for LLM iterations
    }
