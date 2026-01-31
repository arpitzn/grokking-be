"""
Agent Responsibility:
- Normalizes incoming case input using LLM-based entity extraction
- Extracts entities (order_id, customer_id, zone_id) with confidence scoring
- Populates case slice in state
- Does NOT classify intent or retrieve data
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from app.agent.state import AgentState, emit_phase_event
from app.infra.llm import get_llm_service, get_cheap_model
from app.infra.prompts import get_prompts


class IngestionOutput(BaseModel):
    """Structured output for LLM-based entity extraction"""
    order_id: Optional[str] = Field(
        None, 
        description="Extracted order ID (e.g., 'order_12345', '12345', 'ORDER-123')"
    )
    customer_id: str = Field(
        ..., 
        description="Customer identifier (defaults to user_id if not found)"
    )
    zone_id: Optional[str] = Field(
        None, 
        description="Delivery zone ID if mentioned"
    )
    restaurant_id: Optional[str] = Field(
        None, 
        description="Restaurant ID if mentioned"
    )
    normalized_query: str = Field(
        ..., 
        description="Cleaned, normalized version of the user query"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence in entity extraction (0.0 to 1.0)"
    )


async def ingestion_node(state: AgentState) -> AgentState:
    """
    Ingestion node: Normalizes input and extracts entities using LLM.
    
    Input: Raw user message, user_id, conversation_id
    Output: Populated case slice with extracted entities + confidence
    """
    # Get raw input from state
    raw_text = state.get("case", {}).get("raw_text", "")
    user_id = state.get("case", {}).get("user_id", "")
    conversation_id = state.get("case", {}).get("conversation_id", "")
    persona = state.get("case", {}).get("persona", "customer")
    channel = state.get("case", {}).get("channel", "web")
    
    # Get prompts from centralized prompts module
    system_prompt, user_prompt = get_prompts(
        "ingestion_agent",
        {
            "raw_text": raw_text,
            "user_id": user_id
        }
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Use structured output with cheap model
    llm_service = get_llm_service()
    llm = llm_service.get_structured_output_llm_instance(
        model_name=get_cheap_model(),
        schema=IngestionOutput,
        temperature=0  # Low temperature for deterministic extraction
    )
    
    lc_messages = llm_service.convert_messages(messages)
    response: IngestionOutput = await llm.ainvoke(lc_messages)
    
    # Populate case slice with extracted entities
    state["case"] = {
        "persona": persona,
        "channel": channel,
        "order_id": response.order_id,
        "customer_id": response.customer_id or user_id,
        "zone_id": response.zone_id,
        "restaurant_id": response.restaurant_id,
        "raw_text": raw_text,
        "normalized_text": response.normalized_query,  # Add normalized version
        "locale": "en-US",
        "conversation_id": conversation_id,
        "user_id": user_id
    }
    
    # Initialize confidence tracking
    if "confidence_scores" not in state:
        state["confidence_scores"] = {}
    state["confidence_scores"]["ingestion"] = response.confidence
    
    # Derive entities_found from extracted fields for logging
    entities_found = []
    if response.order_id:
        entities_found.append("order_id")
    if response.zone_id:
        entities_found.append("zone_id")
    if response.restaurant_id:
        entities_found.append("restaurant_id")
    
    # Emit phase event with entity details
    entities_str = ", ".join(entities_found) if entities_found else "none"
    emit_phase_event(
        state, 
        "ingestion", 
        f"Extracted entities: {entities_str} (confidence: {response.confidence:.2f})",
        metadata={
            "entities": entities_found,
            "confidence": response.confidence,
            "order_id": response.order_id
        }
    )
    
    return state
