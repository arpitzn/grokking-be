"""Semantic memory service for Mem0 integration"""
from app.infra.mem0 import get_mem0_client
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


async def async_write_to_mem0(
    user_id: str,
    conversation_id: str,
    messages: List[Dict[str, str]]
) -> None:
    """
    Write facts/preferences to Mem0 asynchronously (fire-and-forget)
    
    Extracts facts, preferences, constraints from recent messages
    """
    try:
        mem0_client = await get_mem0_client()
        
        # Extract facts from recent assistant messages
        # This is simplified - in production, use LLM to extract structured facts
        for msg in messages[-5:]:  # Last 5 messages
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Simple extraction - in production, use LLM for better extraction
                if len(content) > 50:  # Only extract from substantial messages
                    await mem0_client.add_memory(
                        user_id=user_id,
                        memory=content[:200],  # Truncate for simplicity
                        metadata={"conversation_id": conversation_id}
                    )
        
        logger.info(f"Wrote semantic memories for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to write to Mem0: {e}")
        # Fail silently - non-blocking operation


async def read_from_mem0(user_id: str, query: str) -> List[Dict[str, Any]]:
    """
    Read relevant memories from Mem0
    
    Returns:
        List of memory dicts
    """
    try:
        mem0_client = await get_mem0_client()
        results = await mem0_client.search_memories(user_id=user_id, query=query, limit=5)
        return results
    except Exception as e:
        logger.error(f"Failed to read from Mem0: {e}")
        return []
