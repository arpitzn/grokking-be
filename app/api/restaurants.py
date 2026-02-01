"""Restaurants API endpoint"""
from fastapi import APIRouter, HTTPException, Query
from app.infra.mongo import get_mongodb_client
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_db_operation, log_error_with_context
)
from typing import List, Dict, Any, Optional
import logging
import time
from bson import Binary, ObjectId
from uuid import UUID
from app.utils.uuid_helpers import binary_to_uuid, uuid_to_binary, is_uuid_string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/restaurants", tags=["restaurants"])


def sanitize_for_logging(data: Any) -> Any:
    """Convert ObjectId and Binary UUID objects to strings for JSON serialization"""
    if isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, Binary):
        try:
            return str(data.as_uuid())
        except (ValueError, AttributeError):
            return data.hex() if hasattr(data, 'hex') else str(data)
    elif isinstance(data, dict):
        return {key: sanitize_for_logging(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    else:
        return data


def serialize_restaurant(restaurant: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB restaurant document to API response format"""
    # Convert Binary UUID to string
    restaurant_id = restaurant.get("_id")
    if restaurant_id and isinstance(restaurant_id, Binary):
        try:
            restaurant_id = binary_to_uuid(restaurant_id)
        except (ValueError, AttributeError):
            # If Binary is not a UUID, convert to hex string
            restaurant_id = restaurant_id.hex() if hasattr(restaurant_id, 'hex') else str(restaurant_id)
    elif restaurant_id and isinstance(restaurant_id, ObjectId):
        restaurant_id = str(restaurant_id)
    elif restaurant_id:
        restaurant_id = str(restaurant_id)
    
    # Handle location.zone_id (can be Binary UUID or ObjectId)
    location = restaurant.get("location", {})
    if location and isinstance(location, dict):
        zone_id = location.get("zone_id")
        if zone_id and isinstance(zone_id, Binary):
            try:
                location["zone_id"] = binary_to_uuid(zone_id)
            except (ValueError, AttributeError):
                # If Binary is not a UUID, convert to hex string
                location["zone_id"] = zone_id.hex() if hasattr(zone_id, 'hex') else str(zone_id)
        elif zone_id and isinstance(zone_id, ObjectId):
            location["zone_id"] = str(zone_id)
        elif zone_id:
            location["zone_id"] = str(zone_id)
    
    # Serialize datetime fields
    created_at = restaurant.get("created_at")
    if created_at:
        if isinstance(created_at, str):
            created_at = created_at
        else:
            created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
    
    updated_at = restaurant.get("updated_at")
    if updated_at:
        if isinstance(updated_at, str):
            updated_at = updated_at
        else:
            updated_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
    
    # Handle current_metrics.updated_at
    current_metrics = restaurant.get("current_metrics", {})
    if current_metrics and isinstance(current_metrics, dict):
        metrics_updated_at = current_metrics.get("updated_at")
        if metrics_updated_at:
            if isinstance(metrics_updated_at, str):
                current_metrics["updated_at"] = metrics_updated_at
            else:
                current_metrics["updated_at"] = metrics_updated_at.isoformat() if hasattr(metrics_updated_at, 'isoformat') else str(metrics_updated_at)
    
    return {
        "restaurant_id": restaurant_id,
        "name": restaurant.get("name", ""),
        "type": restaurant.get("type", ""),
        "cuisines": restaurant.get("cuisines", []),
        "location": location,
        "is_open": restaurant.get("is_open", False),
        "is_paused": restaurant.get("is_paused", False),
        "status": restaurant.get("status", ""),
        "current_metrics": current_metrics,
        "created_at": created_at,
        "updated_at": updated_at,
    }


@router.get("")
async def get_restaurants(
    zone_id: Optional[str] = Query(None, description="Filter by zone_id"),
    status: Optional[str] = Query("active", description="Filter by status (default: active)")
):
    """Get restaurants with optional filtering by zone_id and status"""
    start_time = time.time()
    log_request_start(logger, "GET", "/api/restaurants", query_params={"zone_id": zone_id, "status": status})
    
    try:
        db = await get_mongodb_client()
        
        # Build query
        query: Dict[str, Any] = {}
        
        # Filter by zone_id if provided
        if zone_id:
            # Try to find the zone and get its Binary UUID _id
            # This handles cases where zone_id might be a UUID string, ObjectId string, or Binary UUID hex
            zone_doc = None
            
            # Approach 1: Try as UUID string and convert to Binary UUID
            if is_uuid_string(zone_id):
                try:
                    zone_id_binary = uuid_to_binary(zone_id)
                    zone_doc = await db.zones.find_one({"_id": zone_id_binary})
                except (ValueError, TypeError):
                    pass
            
            # Approach 2: If not found and looks like ObjectId (24 hex chars), try as ObjectId
            if not zone_doc and len(zone_id) == 24:
                try:
                    if all(c in '0123456789abcdef' for c in zone_id.lower()):
                        zone_id_objid = ObjectId(zone_id)
                        zone_doc = await db.zones.find_one({"_id": zone_id_objid})
                        # If zone found with ObjectId, also try querying restaurants with ObjectId
                        if zone_doc:
                            query["location.zone_id"] = zone_id_objid
                except (ValueError, TypeError):
                    pass
            
            # Approach 3: Use the zone's _id to query restaurants
            if zone_doc and zone_doc.get("_id") and "location.zone_id" not in query:
                # Use the zone's _id (Binary UUID or ObjectId) to query restaurants
                query["location.zone_id"] = zone_doc["_id"]
            elif "location.zone_id" not in query:
                # Fallback: query restaurants with the string zone_id
                # This handles cases where restaurants.location.zone_id might be stored as string
                query["location.zone_id"] = zone_id
        
        # Filter by status if provided
        if status:
            query["status"] = status
        
        # Fetch restaurants
        cursor = db.restaurants.find(query).sort([("name", 1)])
        restaurants_raw = await cursor.to_list(length=None)
        
        # Serialize restaurants
        restaurants = [serialize_restaurant(restaurant) for restaurant in restaurants_raw]
        
        # Sanitize query for logging (convert ObjectId/Binary to strings)
        query_for_logging = sanitize_for_logging(query)
        
        # Log DB result validation
        log_db_operation(
            logger, "find", "restaurants",
            result_count=len(restaurants),
            expected=False,  # Empty is valid if no restaurants match
            filters=query_for_logging
        )
        
        log_request_end(
            logger, "GET", "/api/restaurants",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"restaurant_count": len(restaurants), "filters": query_for_logging},
        )
        
        return {
            "restaurants": restaurants,
            "count": len(restaurants)
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "get_restaurants_error",
            context={"zone_id": zone_id, "status": status}
        )
        raise HTTPException(status_code=500, detail=str(e))
