"""
Agent Responsibility:
- Normalizes incoming case input
- Extracts entities (order_id, customer_id, zone_id)
- Populates case slice in state
- Does NOT classify intent or retrieve data
"""

import re
from typing import Dict, Any

from app.agent.state import AgentState


async def ingestion_node(state: AgentState) -> AgentState:
    """
    Ingestion node: Normalizes input and extracts entities.
    
    Input: Raw user message, user_id, conversation_id
    Output: Populated case slice with extracted entities
    """
    # Get raw input from state (will be populated by API endpoint)
    raw_text = state.get("case", {}).get("raw_text", "")
    user_id = state.get("case", {}).get("user_id", "")
    conversation_id = state.get("case", {}).get("conversation_id", "")
    
    # Extract entities using regex patterns (simple extraction for hackathon)
    order_id = None
    customer_id = user_id  # Default to user_id
    zone_id = None
    restaurant_id = None
    
    # Extract order ID (pattern: order_123, ORDER-123, order #123)
    order_patterns = [
        r'order[_\s#-]?(\w+)',
        r'ORDER[_\s#-]?(\w+)',
        r'order\s+#?(\d+)',
    ]
    for pattern in order_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            order_id = match.group(1) if match.lastindex else match.group(0)
            break
    
    # Extract zone ID (pattern: zone_123, zone-123)
    zone_patterns = [
        r'zone[_\s-]?(\w+)',
        r'ZONE[_\s-]?(\w+)',
    ]
    for pattern in zone_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            zone_id = match.group(1) if match.lastindex else match.group(0)
            break
    
    # Extract restaurant ID (pattern: restaurant_123, rest_123)
    restaurant_patterns = [
        r'restaurant[_\s-]?(\w+)',
        r'rest[_\s-]?(\w+)',
    ]
    for pattern in restaurant_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            restaurant_id = match.group(1) if match.lastindex else match.group(0)
            break
    
    # Populate case slice
    state["case"] = {
        "persona": "customer",  # Default persona
        "channel": "web",  # Default channel
        "order_id": order_id,
        "customer_id": customer_id,
        "zone_id": zone_id,
        "restaurant_id": restaurant_id,
        "raw_text": raw_text,
        "locale": "en-US",  # Default locale
        "conversation_id": conversation_id,
        "user_id": user_id
    }
    
    # Add CoT trace entry
    turn_number = state.get("turn_number", 1)
    if "cot_trace" not in state:
        state["cot_trace"] = []
    state["cot_trace"].append({
        "phase": "ingestion",
        "turn": turn_number,
        "content": f"[Turn {turn_number}] Extracted entities: order_id={order_id}, zone_id={zone_id}, restaurant_id={restaurant_id}"
    })
    
    return state
