"""Escalated tickets endpoint for Area Managers and Customer Care Representatives"""
from fastapi import APIRouter, HTTPException
from app.infra.mongo import get_mongodb_client
from app.models.schemas import EscalatedTicketsResponse, EscalatedTicketItem
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_db_operation, log_error_with_context
)
from typing import List, Dict, Any
import logging
import time
from bson import ObjectId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/escalated-tickets", tags=["escalated-tickets"])


def serialize_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB ticket document to API response format"""
    # Convert ObjectId to string
    ticket_id = str(ticket.get("_id", ""))
    if "ticket_id" in ticket:
        ticket_id = str(ticket["ticket_id"])
    
    # Convert Binary UUIDs to strings if present
    user_id = ticket.get("user_id")
    if user_id and isinstance(user_id, bytes):
        user_id = str(user_id)
    elif user_id:
        user_id = str(user_id)
    
    order_id = ticket.get("order_id")
    if order_id and isinstance(order_id, bytes):
        order_id = str(order_id)
    elif order_id:
        order_id = str(order_id)
    
    restaurant_id = ticket.get("restaurant_id")
    if restaurant_id and isinstance(restaurant_id, bytes):
        restaurant_id = str(restaurant_id)
    elif restaurant_id:
        restaurant_id = str(restaurant_id)
    
    # Handle affected_zones array
    affected_zones = ticket.get("affected_zones", [])
    if affected_zones:
        affected_zones = [str(zone) if isinstance(zone, bytes) else str(zone) for zone in affected_zones]
    
    # Serialize datetime fields
    created_at = ticket.get("created_at")
    if created_at:
        if isinstance(created_at, str):
            created_at = created_at
        else:
            created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
    
    updated_at = ticket.get("updated_at")
    if updated_at:
        if isinstance(updated_at, str):
            updated_at = updated_at
        else:
            updated_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
    
    timestamp = ticket.get("timestamp")
    if timestamp:
        if isinstance(timestamp, str):
            timestamp = timestamp
        else:
            timestamp = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
    
    # Serialize agent_notes - convert dict format to string format
    agent_notes_raw = ticket.get("agent_notes", [])
    agent_notes = []
    for note in agent_notes_raw:
        if isinstance(note, dict):
            # Convert dict format: {"note": "...", "created_at": ..., "created_by": "..."}
            # to string format: "[2026-02-01 12:00:00 by system] Note text"
            note_text = note.get("note", "")
            created_at_note = note.get("created_at")
            created_by = note.get("created_by", "unknown")
            
            if created_at_note:
                if isinstance(created_at_note, str):
                    timestamp_str = created_at_note
                else:
                    timestamp_str = created_at_note.isoformat() if hasattr(created_at_note, 'isoformat') else str(created_at_note)
                agent_notes.append(f"[{timestamp_str} by {created_by}] {note_text}")
            else:
                agent_notes.append(f"[by {created_by}] {note_text}")
        elif isinstance(note, str):
            # Already string format
            agent_notes.append(note)
        else:
            # Fallback: convert to string
            agent_notes.append(str(note))
    
    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "ticket_type": ticket.get("ticket_type", ""),
        "issue_type": ticket.get("issue_type", ""),
        "subtype": ticket.get("subtype"),
        "severity": ticket.get("severity"),
        "scope": ticket.get("scope", ""),
        "order_id": order_id,
        "restaurant_id": restaurant_id,
        "affected_zones": affected_zones,
        "affected_city": ticket.get("affected_city"),
        "title": ticket.get("title", ""),
        "description": ticket.get("description", ""),
        "status": ticket.get("status", ""),
        "created_at": created_at,
        "updated_at": updated_at,
        "timestamp": timestamp,
        "related_orders": [str(o) if isinstance(o, bytes) else str(o) for o in ticket.get("related_orders", [])],
        "related_tickets": [str(t) if isinstance(t, bytes) else str(t) for t in ticket.get("related_tickets", [])],
        "agent_notes": agent_notes,
        "resolution_history": ticket.get("resolution_history", []),
        "resolution": ticket.get("resolution"),
    }


@router.get("/{user_id}", response_model=EscalatedTicketsResponse)
async def get_escalated_tickets(user_id: str):
    """Get escalated tickets (complaint type, severity 1-2) for a user"""
    start_time = time.time()
    log_request_start(logger, "GET", f"/api/escalated-tickets/{user_id}", user_id=user_id)
    
    try:
        db = await get_mongodb_client()
        
        # Query for escalated tickets: ticket_type="complaint" AND severity IN [1, 2]
        query = {
            "ticket_type": "complaint",
            "severity": {"$in": [1, 2]}
        }
        
        # Fetch tickets sorted by severity ASC (Critical=1 first), then created_at DESC (latest first)
        cursor = db.support_tickets.find(query).sort([("severity", 1), ("created_at", -1)])
        tickets_raw = await cursor.to_list(length=None)  # No pagination - return all
        
        # Serialize tickets
        tickets = [serialize_ticket(ticket) for ticket in tickets_raw]
        
        # Log DB result validation
        log_db_operation(
            logger, "find", "support_tickets",
            result_count=len(tickets),
            expected=False,  # Empty is valid if no escalated tickets
            user_id=user_id,
            filters={"ticket_type": "complaint", "severity": [1, 2]}
        )
        
        log_request_end(
            logger, "GET", f"/api/escalated-tickets/{user_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"ticket_count": len(tickets)},
            user_id=user_id
        )
        
        return {
            "tickets": tickets,
            "count": len(tickets),
            "total": len(tickets)
        }
    except Exception as e:
        log_error_with_context(
            logger, e, "get_escalated_tickets_error",
            context={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail=str(e))
