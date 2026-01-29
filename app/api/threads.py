"""Thread/conversation endpoints"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import ConversationListItem, MessageItem
from app.services.conversation import list_threads, get_messages
from typing import List
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["threads"])


@router.get("/{user_id}", response_model=List[ConversationListItem])
async def get_threads(user_id: str):
    """List all conversations for a user"""
    try:
        conversations = await list_threads(user_id)
        return conversations
    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}/messages", response_model=List[MessageItem])
async def get_conversation_messages(conversation_id: str, limit: int = 10, offset: int = 0):
    """Get messages from a conversation"""
    try:
        messages = await get_messages(conversation_id, limit=limit, offset=offset)
        return messages
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))
