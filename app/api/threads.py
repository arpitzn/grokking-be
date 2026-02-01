"""Thread/conversation endpoints"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ConversationListItem, MessageItem
from app.services.conversation import list_threads, get_messages, delete_conversation
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_db_operation, log_error_with_context
)
from typing import List
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("/{user_id}", response_model=List[ConversationListItem])
async def get_threads(user_id: str):
    """List all conversations for a user"""
    start_time = time.time()
    log_request_start(logger, "GET", f"/threads/{user_id}", user_id=user_id)
    
    try:
        conversations = await list_threads(user_id)
        
        # Log DB result validation
        log_db_operation(
            logger, "find", "conversations",
            result_count=len(conversations),
            expected=False,  # Empty is valid for new users
            user_id=user_id
        )
        
        log_request_end(
            logger, "GET", f"/threads/{user_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"conversation_count": len(conversations)},
            user_id=user_id
        )
        
        return conversations
    except Exception as e:
        log_error_with_context(logger, e, "list_threads_error", context={"user_id": user_id})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages", response_model=List[MessageItem])
async def get_conversation_messages(conversation_id: str, limit: int = 10, offset: int = 0):
    """Get messages from a conversation"""
    start_time = time.time()
    log_request_start(
        logger, "GET", f"/threads/{conversation_id}/messages",
        query_params={"limit": limit, "offset": offset}
    )
    
    try:
        messages = await get_messages(conversation_id, limit=limit, offset=offset)
        
        # Warn if conversation doesn't exist
        log_db_operation(
            logger, "find", "messages",
            result_count=len(messages),
            expected=True,  # Should have messages if conversation_id is valid
            filters={"conversation_id": conversation_id}
        )
        
        log_request_end(
            logger, "GET", f"/threads/{conversation_id}/messages",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"message_count": len(messages), "limit": limit, "offset": offset}
        )
        
        return messages
    except Exception as e:
        log_error_with_context(
            logger, e, "get_messages_error",
            context={"conversation_id": conversation_id, "limit": limit, "offset": offset}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation and all related records"""
    start_time = time.time()
    log_request_start(logger, "DELETE", f"/threads/{conversation_id}")
    
    try:
        deleted = await delete_conversation(conversation_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        log_request_end(
            logger, "DELETE", f"/threads/{conversation_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000
        )
        
        return {"success": True, "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "delete_conversation_error",
            context={"conversation_id": conversation_id}
        )
        raise HTTPException(status_code=500, detail=str(e))
