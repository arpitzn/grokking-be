"""Memory management API endpoints"""
from fastapi import APIRouter, HTTPException
from app.infra.mem0 import get_mem0_client
from app.infra.mongo import get_mongodb_client
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/{user_id}")
async def list_user_memories(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    List all memories for a user
    
    Returns memories from both Mem0 and MongoDB summaries
    """
    mem0_client = await get_mem0_client()
    db = await get_mongodb_client()
    
    memories = []
    
    # Get Mem0 memories (semantic + episodic)
    try:
        # Use get_all() to retrieve all memories for the user
        # This is the correct way to list all memories (not search with empty query)
        mem0_memories = await mem0_client.get_all(
            user_id=user_id
        )
        
        # Add Mem0 memories
        for mem in mem0_memories:
            memories.append({
                "memory_id": mem.get("id", str(mem.get("memory_id", ""))),
                "type": "mem0",
                "content": mem.get("memory", "") or mem.get("content", ""),
                "created_at": mem.get("created_at", ""),
                "metadata": mem.get("metadata", {})
            })
    except Exception as e:
        logger.error(f"Failed to fetch Mem0 memories: {e}", exc_info=True)
    
    # Get conversation summaries from MongoDB
    try:
        # Get summaries for conversations where user_id matches
        # Note: summaries collection doesn't have user_id directly, need to join via conversations
        conversations = await db.conversations.find(
            {"user_id": user_id}
        ).to_list(length=100)
        
        conversation_ids = [conv["_id"] for conv in conversations]
        
        if conversation_ids:
            summaries = await db.summaries.find(
                {"conversation_id": {"$in": conversation_ids}}
            ).sort("last_summarized_at", -1).limit(limit).to_list(length=limit)
            
            # Add conversation summaries
            for summary in summaries:
                memories.append({
                    "memory_id": summary["_id"],
                    "type": "summary",
                    "content": summary["summary"],
                    "created_at": summary["last_summarized_at"].isoformat(),
                    "metadata": {
                        "conversation_id": summary["conversation_id"],
                        "message_count": summary["message_count_at_summary"]
                    }
                })
    except Exception as e:
        logger.warning(f"Failed to fetch summaries: {e}")
    
    return memories


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, memory_type: str = "summary"):
    """
    Delete a specific memory
    
    Args:
        memory_id: Memory ID to delete
        memory_type: Type of memory (mem0 | summary)
    """
    if memory_type == "mem0":
        # Delete from Mem0 (would need Mem0 SDK delete method)
        # Note: Check Mem0 SDK documentation for delete API
        raise HTTPException(
            status_code=501,
            detail="Mem0 delete not yet implemented - check SDK docs"
        )
    elif memory_type == "summary":
        # Delete from MongoDB
        db = await get_mongodb_client()
        result = await db.summaries.delete_one({"_id": memory_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "memory_id": memory_id}
    else:
        raise HTTPException(status_code=400, detail="Invalid memory_type")


@router.put("/{memory_id}")
async def update_memory(
    memory_id: str,
    content: str,
    memory_type: str = "summary"
):
    """
    Update memory content
    
    Only supports updating summaries (MongoDB), not Mem0 memories
    """
    if memory_type != "summary":
        raise HTTPException(
            status_code=400,
            detail="Only summary memories can be updated"
        )
    
    db = await get_mongodb_client()
    result = await db.summaries.update_one(
        {"_id": memory_id},
        {"$set": {"summary": content}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"status": "updated", "memory_id": memory_id}
