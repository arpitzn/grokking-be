"""User service for MongoDB CRUD operations"""
from app.infra.mongo import get_mongodb_client
from app.utils.uuid_helpers import binary_to_uuid
from bson import Binary
from typing import Optional, Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


async def get_random_user_by_persona(
    persona: str, 
    sub_category: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a random user by persona and optional sub_category
    
    Args:
        persona: User persona ("area_manager", "customer_care_rep", "end_customer")
        sub_category: Optional sub_category for end_customer ("platinum", "standard", "high_risk")
    
    Returns:
        User document with user_id, persona, sub_category fields, or None if no users found
    """
    db = await get_mongodb_client()
    
    # Map frontend persona to MongoDB persona
    # Frontend uses "end_customer" but MongoDB stores "customer"
    mongo_persona = "customer" if persona == "end_customer" else persona
    
    # Build match filter
    match_filter = {
        "persona": mongo_persona,
        "status": "active"
    }
    
    # For end_customer, add sub_category filter (default to "standard" if not provided)
    if persona == "end_customer":
        match_filter["sub_category"] = sub_category or "standard"
    
    # Build aggregation pipeline with $sample for random selection
    pipeline = [
        {"$match": match_filter},
        {"$sample": {"size": 1}}
    ]
    
    # Execute aggregation
    cursor = db.users.aggregate(pipeline)
    result = await cursor.to_list(length=1)
    
    if not result or len(result) == 0:
        logger.info(json.dumps({
            "event": "no_users_found",
            "persona": persona,
            "sub_category": sub_category,
            "message": "No active users found matching criteria"
        }))
        return None
    
    user_doc = result[0]
    
    # Extract relevant fields
    # Map MongoDB "customer" back to frontend "end_customer"
    mongo_persona = user_doc.get("persona")
    frontend_persona = "end_customer" if mongo_persona == "customer" else mongo_persona
    
    # Get user_id - prefer user_id field, fallback to _id converted to string
    user_id = user_doc.get("user_id")
    if not user_id:
        # Convert _id (Binary UUID) to string if user_id doesn't exist
        _id = user_doc.get("_id")
        if isinstance(_id, Binary):
            user_id = binary_to_uuid(_id)
        else:
            user_id = str(_id)
    
    user_data = {
        "user_id": user_id,
        "persona": frontend_persona,
        "sub_category": user_doc.get("sub_category")
    }
    
    logger.info(json.dumps({
        "event": "user_resolved",
        "persona": persona,
        "sub_category": sub_category,
        "user_id": user_data["user_id"],
        "selected_user_persona": user_data["persona"],
        "selected_user_sub_category": user_data["sub_category"]
    }))
    
    return user_data
