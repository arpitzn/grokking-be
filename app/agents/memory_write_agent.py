"""
Agent Responsibility:
- Writes to Mem0 asynchronously (non-blocking)
- Writes episodic and semantic memories
- Triggers periodic summarization (every 10 messages)
- Does NOT block response generation

IMPORTANT: Memory operations are always async and never block.
"""

import asyncio
from typing import Dict, Any

from app.agent.state import AgentState
from app.tools.mem0.write_memory import write_memory


async def summarize_conversation_async(conversation_id: str) -> None:
    """
    Asynchronous summarization (fire-and-forget):
    1. Fetch messages since last summary
    2. Generate summary using cheap LLM
    3. Store in MongoDB summaries collection
    4. Never blocks response generation
    """
    # Mock implementation for hackathon
    # In production, this would:
    # 1. Fetch messages from MongoDB
    # 2. Call LLM for summarization
    # 3. Store summary in MongoDB
    pass


async def memory_write_node(state: AgentState) -> AgentState:
    """
    Memory write node: Async writes to Mem0 and triggers summarization.
    
    Input: case, intent, final_response or handover_packet
    Output: Non-blocking memory updates
    """
    case = state.get("case", {})
    intent = state.get("intent", {})
    final_response = state.get("final_response", "")
    handover_packet = state.get("handover_packet")
    
    user_id = case.get("customer_id", case.get("user_id", ""))
    
    # 1. Write to Mem0 (episodic + semantic) - async, non-blocking
    try:
        # Episodic memory: Store the case outcome
        episodic_content = {
            "case": case,
            "intent": intent,
            "outcome": final_response or (handover_packet.get("escalation_id") if handover_packet else "pending")
        }
        
        # Fire-and-forget async write
        asyncio.create_task(write_memory(
            user_id=user_id,
            memory_type="episodic",
            content=episodic_content
        ))
        
        # Semantic memory: Store learned patterns (if any)
        if final_response:
            semantic_content = {
                "issue_type": intent.get("issue_type"),
                "resolution": "auto_response",
                "pattern": f"Customer {intent.get('issue_type')} issue resolved via auto-response"
            }
            asyncio.create_task(write_memory(
                user_id=user_id,
                memory_type="semantic",
                content=semantic_content
            ))
    
    except Exception as e:
        # Non-blocking failure - log but don't raise
        pass
    
    # 2. Check if summarization needed (every 10 messages)
    working_memory = state.get("working_memory", [])
    message_count = len(working_memory)
    
    if message_count >= 10:
        # Trigger async summarization (fire-and-forget)
        conversation_id = case.get("conversation_id", "")
        asyncio.create_task(summarize_conversation_async(conversation_id))
    
    # Non-blocking, parallel execution
    return state
