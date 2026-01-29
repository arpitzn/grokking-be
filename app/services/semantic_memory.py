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
    Write user-assistant interactions to Mem0 asynchronously (fire-and-forget)
    
    Extracts structured message pairs from recent messages and stores them in Mem0.
    Uses SDK's add() method with proper message pair format.
    """
    try:
        mem0_client = await get_mem0_client()
        
        # Extract user-assistant message pairs from recent messages
        # Look for pairs where user message is followed by assistant response
        message_pairs = []
        i = 0
        while i < len(messages) - 1:
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            # Check if we have a user-assistant pair
            if current_msg.get("role") == "user" and next_msg.get("role") == "assistant":
                pair = [
                    {"role": "user", "content": current_msg.get("content", "")},
                    {"role": "assistant", "content": next_msg.get("content", "")}
                ]
                # Only add substantial pairs (both messages have content)
                if pair[0]["content"] and pair[1]["content"]:
                    message_pairs.append(pair)
                i += 2  # Skip both messages
            else:
                i += 1
        
        # Store each pair in Mem0
        for pair in message_pairs[-3:]:  # Store last 3 pairs to avoid overwhelming
            try:
                await mem0_client.add_interaction(user_id=user_id, messages=pair)
            except Exception as e:
                logger.warning(f"Failed to store message pair in Mem0: {e}")
        
        if message_pairs:
            logger.info(f"Wrote {len(message_pairs)} semantic memory pairs for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to write to Mem0: {e}")
        # Fail silently - non-blocking operation


async def read_from_mem0(user_id: str, query: str) -> List[Dict[str, Any]]:
    """
    Read relevant memories from Mem0 using semantic search
    
    Args:
        user_id: User identifier
        query: Search query for semantic retrieval
    
    Returns:
        List of memory dicts with 'memory' field
    """
    try:
        mem0_client = await get_mem0_client()
        results = await mem0_client.search(query=query, user_id=user_id, limit=5)
        return results
    except Exception as e:
        logger.error(f"Failed to read from Mem0: {e}")
        return []
