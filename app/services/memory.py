"""Memory service for working memory assembly"""
from app.infra.mongo import get_mongodb_client
from app.services.semantic_memory import read_from_mem0
from app.services.conversation import get_messages
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


async def build_working_memory(
    conversation_id: str,
    user_id: str,
    include_mem0: bool = True
) -> List[Dict[str, str]]:
    """
    Build working memory from summary + recent messages + optional Mem0
    
    Returns:
        List of message dicts in format: [{"role": "user", "content": "..."}, ...]
    """
    db = await get_mongodb_client()
    
    # 1. Fetch latest summary
    summary_doc = await db.summaries.find_one(
        {"conversation_id": conversation_id},
        sort=[("last_summarized_at", -1)]
    )
    
    working_memory = []
    
    # 2. Add system message with summary if available
    if summary_doc:
        summary_text = summary_doc.get("summary", "")
        if summary_text:
            working_memory.append({
                "role": "system",
                "content": f"Previous conversation summary: {summary_text}"
            })
    
    # 3. Fetch last 10 messages
    messages = await get_messages(conversation_id, limit=10, offset=0)
    
    for msg in messages:
        working_memory.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 4. Optionally query Mem0 for user preferences
    if include_mem0:
        try:
            mem0_results = await read_from_mem0(user_id, "user preferences")
            if mem0_results:
                preferences_text = "\n".join([m.get("memory", "") for m in mem0_results[:3]])
                if preferences_text:
                    working_memory.insert(1, {
                        "role": "system",
                        "content": f"User preferences: {preferences_text}"
                    })
        except Exception as e:
            logger.warning(f"Mem0 query failed: {e}. Continuing without Mem0 context.")
    
    logger.info(f"Built working memory with {len(working_memory)} messages for conversation {conversation_id}")
    return working_memory


async def should_summarize(conversation_id: str) -> bool:
    """
    Check if conversation should be summarized
    
    Trigger: Every 10 messages since last summary
    """
    db = await get_mongodb_client()
    
    # Get latest summary
    summary_doc = await db.summaries.find_one(
        {"conversation_id": conversation_id},
        sort=[("last_summarized_at", -1)]
    )
    
    if not summary_doc:
        # No summary yet - check if we have 10+ messages
        message_count = await db.messages.count_documents({"conversation_id": conversation_id})
        return message_count >= 10
    
    # Get message count since last summary
    last_summary_time = summary_doc["last_summarized_at"]
    message_count_since = await db.messages.count_documents({
        "conversation_id": conversation_id,
        "created_at": {"$gt": last_summary_time}
    })
    
    return message_count_since >= 10
