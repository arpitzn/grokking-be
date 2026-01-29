"""Conversation service for MongoDB CRUD operations"""
from app.infra.mongo import get_mongodb_client
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import logging
import json

logger = logging.getLogger(__name__)


async def create_conversation(user_id: str, title: Optional[str] = None) -> str:
    """Create a new conversation"""
    db = await get_mongodb_client()
    
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    conversation = {
        "_id": conversation_id,
        "user_id": user_id,
        "title": title or "New Conversation",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "message_count": 0
    }
    
    await db.conversations.insert_one(conversation)
    logger.info(f"Created conversation {conversation_id} for user {user_id}")
    return conversation_id


async def insert_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Insert a message into a conversation"""
    db = await get_mongodb_client()
    
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    message = {
        "_id": message_id,
        "conversation_id": conversation_id,
        "role": role,  # "user" | "assistant" | "system"
        "content": content,
        "created_at": datetime.utcnow(),
        "metadata": metadata or {}
    }
    
    await db.messages.insert_one(message)
    
    # Update conversation message count and updated_at
    await db.conversations.update_one(
        {"_id": conversation_id},
        {
            "$inc": {"message_count": 1},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    return message_id


async def get_messages(
    conversation_id: str,
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Get messages from a conversation"""
    db = await get_mongodb_client()
    
    # Check if conversation exists first
    conversation = await db.conversations.find_one({"_id": conversation_id})
    if not conversation:
        logger.warning(json.dumps({
            "event": "conversation_not_found",
            "conversation_id": conversation_id,
            "operation": "get_messages"
        }))
    
    cursor = db.messages.find(
        {"conversation_id": conversation_id}
    ).sort("created_at", 1).skip(offset).limit(limit)
    
    messages = await cursor.to_list(length=limit)
    
    logger.info(json.dumps({
        "event": "messages_retrieved",
        "conversation_id": conversation_id,
        "count": len(messages),
        "limit": limit,
        "offset": offset
    }))
    
    # Convert ObjectId to string and format
    result = []
    for msg in messages:
        result.append({
            "message_id": msg["_id"],
            "conversation_id": msg["conversation_id"],
            "role": msg["role"],
            "content": msg["content"],
            "created_at": msg["created_at"].isoformat(),
            "metadata": msg.get("metadata", {})
        })
    
    return result


async def list_threads(user_id: str) -> List[Dict[str, Any]]:
    """List all conversations for a user"""
    db = await get_mongodb_client()
    
    cursor = db.conversations.find(
        {"user_id": user_id}
    ).sort("updated_at", -1)
    
    conversations = await cursor.to_list(length=100)
    
    logger.info(json.dumps({
        "event": "conversations_listed",
        "user_id": user_id,
        "count": len(conversations)
    }))
    
    if len(conversations) == 0:
        logger.info(json.dumps({
            "event": "no_conversations_found",
            "user_id": user_id,
            "message": "User has no conversations yet"
        }))
    
    result = []
    for conv in conversations:
        result.append({
            "conversation_id": conv["_id"],
            "title": conv["title"],
            "created_at": conv["created_at"].isoformat(),
            "updated_at": conv["updated_at"].isoformat(),
            "message_count": conv.get("message_count", 0)
        })
    
    return result
