"""Zones API endpoint"""
from fastapi import APIRouter, HTTPException
from app.infra.mongo import get_mongodb_client
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_db_operation, log_error_with_context
)
from typing import List, Dict, Any
import logging
import time
from bson import Binary, ObjectId
from uuid import UUID
from app.utils.uuid_helpers import binary_to_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/zones", tags=["zones"])


def serialize_zone(zone: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB zone document to API response format"""
    # Convert _id to string (handles both Binary UUID and ObjectId)
    zone_id = zone.get("_id")
    if zone_id and isinstance(zone_id, Binary):
        try:
            zone_id = binary_to_uuid(zone_id)
        except (ValueError, AttributeError):
            # If Binary is not a UUID, convert to hex string
            zone_id = zone_id.hex() if hasattr(zone_id, 'hex') else str(zone_id)
    elif zone_id and isinstance(zone_id, ObjectId):
        zone_id = str(zone_id)
    elif zone_id:
        zone_id = str(zone_id)
    
    # Serialize datetime fields
    created_at = zone.get("created_at")
    if created_at:
        if isinstance(created_at, str):
            created_at = created_at
        else:
            created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
    
    updated_at = zone.get("updated_at")
    if updated_at:
        if isinstance(updated_at, str):
            updated_at = updated_at
        else:
            updated_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
    
    # Handle live_metrics.updated_at
    live_metrics = zone.get("live_metrics", {})
    if live_metrics and isinstance(live_metrics, dict):
        live_metrics_updated_at = live_metrics.get("updated_at")
        if live_metrics_updated_at:
            if isinstance(live_metrics_updated_at, str):
                live_metrics["updated_at"] = live_metrics_updated_at
            else:
                live_metrics["updated_at"] = live_metrics_updated_at.isoformat() if hasattr(live_metrics_updated_at, 'isoformat') else str(live_metrics_updated_at)
    
    return {
        "zone_id": zone_id,
        "name": zone.get("name", ""),
        "city": zone.get("city", ""),
        "tier": zone.get("tier"),
        "status": zone.get("status", ""),
        "live_metrics": live_metrics,
        "boundary": zone.get("boundary"),
        "created_at": created_at,
        "updated_at": updated_at,
    }


@router.get("")
async def get_zones():
    """Get all zones"""
    start_time = time.time()
    log_request_start(logger, "GET", "/api/zones")
    
    try:
        db = await get_mongodb_client()
        
        # Fetch all zones
        cursor = db.zones.find({}).sort([("name", 1)])
        zones_raw = await cursor.to_list(length=None)
        
        # Serialize zones
        zones = [serialize_zone(zone) for zone in zones_raw]
        
        # Log DB result validation
        log_db_operation(
            logger, "find", "zones",
            result_count=len(zones),
            expected=False,  # Empty is valid if no zones
        )
        
        log_request_end(
            logger, "GET", "/api/zones",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"zone_count": len(zones)},
        )
        
        return {
            "zones": zones,
            "count": len(zones)
        }
    except Exception as e:
        log_error_with_context(
            logger, e, "get_zones_error",
            context={}
        )
        raise HTTPException(status_code=500, detail=str(e))
