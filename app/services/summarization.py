"""Summarization service for conversation summaries"""
from app.infra.mongo import get_mongodb_client
from app.infra.llm import get_llm_service, get_cheap_model
from app.services.conversation import get_messages
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


async def summarize_conversation(conversation_id: str) -> str:
    """
    Summarize a conversation using GPT-3.5-turbo
    
    Steps:
    1. Fetch messages since last summary (or all messages if no summary)
    2. Call GPT-3.5-turbo with summarization prompt
    3. Store summary in summaries collection
    """
    db = await get_mongodb_client()
    llm_service = get_llm_service()
    
    # Get latest summary to find messages since then
    summary_doc = await db.summaries.find_one(
        {"conversation_id": conversation_id},
        sort=[("last_summarized_at", -1)]
    )
    
    if summary_doc:
        # Get messages since last summary
        last_summary_time = summary_doc["last_summarized_at"]
        # Note: This is simplified - in production, filter by timestamp
        messages = await get_messages(conversation_id, limit=100, offset=0)
        # Filter messages after last summary (simplified - would need timestamp comparison)
        messages_to_summarize = messages[-10:]  # Last 10 messages
    else:
        # No summary yet - summarize all messages
        messages_to_summarize = await get_messages(conversation_id, limit=100, offset=0)
    
    if not messages_to_summarize:
        logger.warning(f"No messages to summarize for conversation {conversation_id}")
        return ""
    
    # Build conversation text for summarization
    conversation_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in messages_to_summarize
    ])
    
    # Create summarization prompt
    prompt = f"""Summarize the following conversation concisely, focusing on key topics, decisions, and important information:

{conversation_text}

Summary:"""
    
    # Call LLM
    messages = [{"role": "user", "content": prompt}]
    llm = llm_service.get_llm_instance(
        model_name=get_cheap_model(),
        temperature=0.3,
        max_completion_tokens=500
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    summary_text = response.content.strip()
    
    # Store summary
    message_count = await db.messages.count_documents({"conversation_id": conversation_id})
    
    summary_doc = {
        "conversation_id": conversation_id,
        "summary": summary_text,
        "last_summarized_at": datetime.utcnow(),
        "message_count_at_summary": message_count,
        "version": (summary_doc.get("version", 0) + 1) if summary_doc else 1
    }
    
    # Upsert summary
    await db.summaries.update_one(
        {"conversation_id": conversation_id},
        {"$set": summary_doc},
        upsert=True
    )
    
    logger.info(f"Created summary for conversation {conversation_id}")
    return summary_text


async def trigger_summarization_if_needed(conversation_id: str) -> Optional[str]:
    """
    Check if summarization is needed and trigger it
    Returns summary text if created, None otherwise
    """
    from app.services.memory import should_summarize
    if await should_summarize(conversation_id):
        return await summarize_conversation(conversation_id)
    return None
