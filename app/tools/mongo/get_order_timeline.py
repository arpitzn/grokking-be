"""
Tool: get_order_timeline
Fetches order timeline from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone
from typing import List, Literal, Type, Union

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import OrderEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import string_to_mongo_id, binary_to_uuid
from app.infra.mongo import get_mongodb_client

logger = logging.getLogger(__name__)


def safe_isoformat(value: Union[datetime, str, None]) -> Union[str, None]:
    """
    Safely convert datetime value to ISO format string.
    Handles both datetime objects and already-formatted strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value  # Already a string, return as-is
    return str(value)  # Fallback for other types

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_order_timeline",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_order_timeline(user_id: str, include: List[str]) -> OrderEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches order timeline from MongoDB for a user (customer_id)
    - Returns structured order events with ToolResult
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_order_timeline")
    print(f"  user_id: {user_id}")
    print(f"  include: {include}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_order_timeline] Starting - user_id={user_id}, include={include}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_order_timeline",
        "params": {"user_id": user_id, "include": include}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        query_user_id = string_to_mongo_id(user_id)
        logger.debug(f"[get_order_timeline] Query user_id (converted): {query_user_id}, type: {type(query_user_id).__name__}")
        
        # Query orders by user_id (customer_id in orders collection)
        # Get most recent order for this user
        query_filter = {"user_id": query_user_id}
        logger.debug(f"[get_order_timeline] MongoDB query filter: {query_filter}")
        
        order_doc = await db.orders.find_one(
            query_filter,
            sort=[("created_at", -1)]
        )
        
        if not order_doc:
            logger.warning(f"[get_order_timeline] No orders found for user_id={user_id}")
            # No orders found - return empty with gap
            return OrderEvidenceEnvelope(
                source="mongo",
                entity_refs=[user_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["order_timeline_unavailable"],
                provenance={"query": "get_order_timeline", "user_id": user_id},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="No orders found for user")
            )
        
        # Transform MongoDB document to tool output format
        # Convert Binary UUIDs and ObjectIds to strings for output
        from bson import ObjectId
        
        order_id_val = order_doc.get("order_id")
        if isinstance(order_id_val, Binary):
            order_id_val = binary_to_uuid(order_id_val)
        elif isinstance(order_id_val, ObjectId):
            order_id_val = str(order_id_val)
        
        user_id_val = order_doc.get("user_id")
        if isinstance(user_id_val, Binary):
            user_id_val = binary_to_uuid(user_id_val)
        elif isinstance(user_id_val, ObjectId):
            user_id_val = str(user_id_val)
        
        # Also handle restaurant_id and zone_id if present
        restaurant_id_val = order_doc.get("restaurant_id")
        if isinstance(restaurant_id_val, (Binary, ObjectId)):
            restaurant_id_val = str(restaurant_id_val)
        
        zone_id_val = order_doc.get("zone_id")
        if isinstance(zone_id_val, (Binary, ObjectId)):
            zone_id_val = str(zone_id_val)
        
        timeline_data = {
            "order_id": order_id_val,
            "user_id": user_id_val,
            "restaurant_id": restaurant_id_val,
            "zone_id": zone_id_val,
            "status": order_doc.get("status"),
            "created_at": safe_isoformat(order_doc.get("created_at")),
            "events": [
                {
                    "timestamp": safe_isoformat(event.get("timestamp")),
                    "event": event.get("event"),
                    "status": event.get("status")
                }
                for event in order_doc.get("events", [])
            ],
            "estimated_delivery": safe_isoformat(order_doc.get("estimated_delivery")),
            "actual_delivery": safe_isoformat(order_doc.get("actual_delivery")),
            "delivery_delay_minutes": order_doc.get("delivery_delay_minutes")
        }
        
        # Include refund and payment if present
        if order_doc.get("refund"):
            timeline_data["refund"] = order_doc.get("refund")
        if order_doc.get("payment"):
            timeline_data["payment"] = order_doc.get("payment")
        if order_doc.get("refund_status"):
            timeline_data["refund_status"] = order_doc.get("refund_status")
        
        events_count = len(timeline_data.get("events", []))
        logger.info(f"[get_order_timeline] Success - order_id={timeline_data.get('order_id')}, "
                   f"status={timeline_data.get('status')}, events_count={events_count}, "
                   f"has_refund={bool(timeline_data.get('refund'))}, has_payment={bool(timeline_data.get('payment'))}")
        logger.debug(f"[get_order_timeline] Output data keys: {list(timeline_data.keys())}")
        logger.debug(f"[get_order_timeline] Events: {timeline_data.get('events', [])}")
        
        result = OrderEvidenceEnvelope(
            source="mongo",
            entity_refs=[user_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.95 if order_doc.get("events") else 0.7,  # Lower confidence if no events
            data=timeline_data,
            gaps=[] if order_doc.get("events") else ["events_missing"],
            provenance={"query": "get_order_timeline", "user_id": user_id, "latency_ms": 50},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=timeline_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_order_timeline",
            "status": "success"
        })
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_order_timeline - SUCCESS")
        print(f"  order_id: {timeline_data.get('order_id')}")
        print(f"  status: {timeline_data.get('status')}")
        print(f"  events_count: {len(timeline_data.get('events', []))}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_order_timeline] Error - user_id={user_id}, error={str(e)}", exc_info=True)
        
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_order_timeline",
            "error": str(e)
        })
        
        return OrderEvidenceEnvelope(
            source="mongo",
            entity_refs=[user_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["order_timeline_unavailable"],
            provenance={"query": "get_order_timeline", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetOrderTimelineInput(BaseModel):
    """Input schema for get_order_timeline tool"""
    user_id: str = Field(
        description="User ID (customer_id) to fetch order timeline for"
    )
    include: List[Literal["events", "status", "timestamps", "refund", "payment"]] = Field(
        default=["events", "status", "timestamps"],
        description=(
            "Fields to include:\n"
            "- events: Order lifecycle events\n"
            "- status: Current status\n"
            "- timestamps: Delivery times\n"
            "- refund: Refund info (if any)\n"
            "- payment: Payment details (if any)"
        )
    )


class GetOrderTimelineTool(BaseTool):
    """LangChain tool wrapper for get_order_timeline"""
    name: str = "get_order_timeline"
    description: str = "Fetches order timeline from MongoDB for a user (customer_id). Returns order events, delivery times, and status information."
    args_schema: Type[BaseModel] = GetOrderTimelineInput
    
    async def _arun(self, user_id: str, include: List[str]) -> str:
        """Async execution - returns JSON string of OrderEvidenceEnvelope"""
        result = await get_order_timeline(user_id, include)
        return result.model_dump_json()
    
    def _run(self, user_id: str, include: List[str]) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
