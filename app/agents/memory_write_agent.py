"""
Agent Responsibility:
- Writes to Mem0 asynchronously (non-blocking)
- Writes episodic and semantic memories
- Triggers periodic summarization (every 10 messages)
- Does NOT block response generation

IMPORTANT: Memory operations are always async and never block.
"""

import asyncio
import logging
from typing import Dict, Any

from app.agent.state import AgentState
from app.tools.mem0.write_memory import write_memory
from app.utils.memory_builder import MemoryBuilder

logger = logging.getLogger(__name__)


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
    Memory write node: Async writes to Mem0 with parallel execution and rich context.
    
    Input: case, intent, final_response, evidence, analysis, conversation_history
    Output: Non-blocking memory updates executed in parallel
    """
    # Extract rich context from state
    case = state.get("case", {})
    intent = state.get("intent", {})
    evidence = state.get("evidence", {})
    analysis = state.get("analysis", {})
    working_memory = state.get("working_memory", [])
    guardrails = state.get("guardrails", {})
    final_response = state.get("final_response", "")
    handover_packet = state.get("handover_packet")
    
    user_id = case.get("customer_id", case.get("user_id", ""))
    outcome = final_response or (handover_packet.get("escalation_id") if handover_packet else "pending")
    
    # Collect all memory write tasks for parallel execution
    memory_tasks = []
    
    # 1. Write episodic memories (user-scoped) - with rich context
    try:
        episodic_memories = await MemoryBuilder.build_episodic_user_memory(
            case=case,
            intent=intent,
            outcome=outcome,
            evidence=evidence,
            analysis=analysis,
            conversation_history=working_memory  # Keep param name for memory_builder compatibility
        )
        
        for memory_content in episodic_memories:
            memory_tasks.append(write_memory(
                content=memory_content,
                memory_type="episodic",
                user_id=user_id  # User-scoped
            ))
    except Exception as e:
        logger.error(f"Episodic memory build failed: {e}")
    
    # 2. Write semantic memories (app-scoped) - with rich context
    try:
        zone_id = case.get("zone_id")
        restaurant_id = case.get("restaurant_id")
        issue_type = intent.get("issue_type", "unknown")
        
        # Get incident signals from evidence if available
        incident_signals = []
        if "mongo" in evidence:
            for item in evidence.get("mongo", []):
                if "incident_signals" in item.get("data", {}):
                    incident_signals.extend(item["data"]["incident_signals"])
        
        if zone_id or restaurant_id:
            semantic_memories = await MemoryBuilder.build_semantic_app_memory(
                zone_id=zone_id,
                restaurant_id=restaurant_id,
                incident_signals=incident_signals,
                issue_type=issue_type,
                evidence=evidence,
                analysis=analysis,
                intent=intent,
                case=case
            )
            
            for memory_content in semantic_memories:
                memory_tasks.append(write_memory(
                    content=memory_content,
                    memory_type="semantic",
                    user_id=None  # App-scoped
                ))
    except Exception as e:
        logger.error(f"Semantic memory build failed: {e}")
    
    # 3. Write procedural memories (app-scoped) - with rich context
    try:
        if final_response:
            issue_type = intent.get("issue_type", "unknown")
            resolution_action = "auto-response"
            confidence = 0.85  # Based on successful resolution
            
            procedural_memories = await MemoryBuilder.build_procedural_app_memory(
                issue_type=issue_type,
                resolution_action=resolution_action,
                confidence=confidence,
                analysis=analysis,
                evidence=evidence,
                intent=intent,
                guardrails=guardrails,
                final_response=final_response
            )
            
            for memory_content in procedural_memories:
                memory_tasks.append(write_memory(
                    content=memory_content,
                    memory_type="procedural",
                    user_id=None  # App-scoped
                ))
    except Exception as e:
        logger.error(f"Procedural memory build failed: {e}")
    
    # Execute all memory writes in parallel (fire-and-forget)
    if memory_tasks:
        # Wrap gather in a coroutine function since create_task expects a coroutine
        async def _run_memory_tasks():
            await asyncio.gather(*memory_tasks, return_exceptions=True)
        
        asyncio.create_task(_run_memory_tasks())
        logger.info(f"Fired {len(memory_tasks)} memory write tasks in parallel")
    
    # 2. Trigger summarization if needed (every 10 messages)
    conversation_id = case.get("conversation_id", "")
    if conversation_id:
        from app.services.summarization import trigger_summarization_if_needed
        # Fire-and-forget async summarization
        asyncio.create_task(trigger_summarization_if_needed(conversation_id))
    
    # Non-blocking, parallel execution
    return state
