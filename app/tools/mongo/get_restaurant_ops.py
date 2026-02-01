"""
Tool: get_restaurant_ops
Fetches restaurant operations data from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Type

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import RestaurantEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import string_to_mongo_id, binary_to_uuid
from app.infra.mongo import get_mongodb_client
from app.infra.demo_constants import DEMO_RESTAURANT_ID, DEMO_TIME_WINDOW

logger = logging.getLogger(__name__)

# Tool specification
TOOL_SPEC = ToolSpec(
    name="get_restaurant_ops",
    criticality=ToolCriticality.DECISION_CRITICAL
)


def parse_time_window(time_window: str) -> tuple[datetime, datetime]:
    """Parse time_window string to start/end datetime"""
    now = datetime.now(timezone.utc)
    
    if time_window.endswith("h"):
        hours = int(time_window[:-1])
        start_time = now - timedelta(hours=hours)
    elif time_window.endswith("d"):
        days = int(time_window[:-1])
        start_time = now - timedelta(days=days)
    else:
        # Default to 24h
        start_time = now - timedelta(hours=24)
    
    return start_time, now


async def get_restaurant_ops() -> RestaurantEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches restaurant operations data from MongoDB (uses hardcoded demo restaurant ID)
    - Returns prep time, quality metrics, and operational status
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    # Use hardcoded demo values
    restaurant_id = DEMO_RESTAURANT_ID
    time_window = DEMO_TIME_WINDOW
    
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_restaurant_ops")
    print(f"  restaurant_id: {restaurant_id}")
    print(f"  time_window: {time_window}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_restaurant_ops] Starting - restaurant_id={restaurant_id}, time_window={time_window}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_restaurant_ops",
        "params": {"restaurant_id": restaurant_id, "time_window": time_window}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        query_restaurant_id = string_to_mongo_id(restaurant_id)
        logger.debug(f"[get_restaurant_ops] Query restaurant_id (converted): {query_restaurant_id}, type: {type(query_restaurant_id).__name__}")
        
        # Query restaurant_metrics_history collection - get latest 10 documents
        query_filter = {"restaurant_id": query_restaurant_id}
        logger.debug(f"[get_restaurant_ops] MongoDB query filter: {query_filter}")
        
        metrics_docs = await db.restaurant_metrics_history.find(
            query_filter,
            sort=[("timestamp", -1)],
            limit=10
        ).to_list(length=10)
        
        logger.info(f"[get_restaurant_ops] Found {len(metrics_docs)} metrics documents")
        if metrics_docs:
            logger.debug(f"[get_restaurant_ops] Latest metrics doc timestamp: {metrics_docs[0].get('timestamp')}")
            logger.debug(f"[get_restaurant_ops] Latest metrics doc sample: {dict(list(metrics_docs[0].items())[:5])}")
        
        # Use the most recent document
        metrics_doc = metrics_docs[0] if metrics_docs else None
        
        # Also get current status from restaurants collection (query by _id)
        restaurant_doc = await db.restaurants.find_one({"_id": query_restaurant_id})
        logger.debug(f"[get_restaurant_ops] Restaurant doc found: {restaurant_doc is not None}")
        if restaurant_doc:
            logger.debug(f"[get_restaurant_ops] Restaurant is_open: {restaurant_doc.get('is_open')}")
        
        if not metrics_doc:
            logger.warning(f"[get_restaurant_ops] No metrics found for restaurant_id={restaurant_id}")
            # Metrics not found - return empty with gap
            return RestaurantEvidenceEnvelope(
                source="mongo",
                entity_refs=[restaurant_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["restaurant_metrics_unavailable"],
                provenance={"query": "get_restaurant_ops", "restaurant_id": restaurant_id, "time_window": time_window},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="Restaurant metrics not found")
            )
        
        # Transform MongoDB document to tool output format
        # Convert ObjectId fields to strings for JSON serialization
        restaurant_id_value = metrics_doc.get("restaurant_id")
        restaurant_id_str = str(restaurant_id_value) if isinstance(restaurant_id_value, (ObjectId, Binary)) else restaurant_id_value
        
        ops_data = {
            "restaurant_id": restaurant_id_str,
            "time_window": metrics_doc.get("time_window"),
            "avg_prep_time_minutes": metrics_doc.get("avg_prep_time_minutes"),
            "on_time_rate": metrics_doc.get("on_time_rate"),
            "quality_rating": metrics_doc.get("quality_rating"),
            "support_ticket_count": metrics_doc.get("support_ticket_count"),  # Changed from complaint_count
            "order_volume": metrics_doc.get("order_volume"),
            "is_open": restaurant_doc.get("is_open", False) if restaurant_doc else metrics_doc.get("is_open", False),
            "current_wait_time": metrics_doc.get("current_wait_time")
        }
        
        logger.info(f"[get_restaurant_ops] Success - avg_prep_time={ops_data.get('avg_prep_time_minutes')}, "
                   f"on_time_rate={ops_data.get('on_time_rate')}, quality_rating={ops_data.get('quality_rating')}, "
                   f"order_volume={ops_data.get('order_volume')}, is_open={ops_data.get('is_open')}")
        logger.debug(f"[get_restaurant_ops] Output data: {ops_data}")
        
        result = RestaurantEvidenceEnvelope(
            source="mongo",
            entity_refs=[restaurant_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.91,
            data=ops_data,
            gaps=[],
            provenance={"query": "get_restaurant_ops", "restaurant_id": restaurant_id, "time_window": time_window, "latency_ms": 40},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=ops_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_restaurant_ops",
            "status": "success"
        })
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_restaurant_ops - SUCCESS")
        print(f"  restaurant_id: {ops_data.get('restaurant_id')}")
        print(f"  avg_prep_time_minutes: {ops_data.get('avg_prep_time_minutes')}")
        print(f"  on_time_rate: {ops_data.get('on_time_rate')}")
        print(f"  is_open: {ops_data.get('is_open')}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_restaurant_ops] Error - restaurant_id={restaurant_id}, error={str(e)}", exc_info=True)
        
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_restaurant_ops",
            "error": str(e)
        })
        
        return RestaurantEvidenceEnvelope(
            source="mongo",
            entity_refs=[restaurant_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["restaurant_ops_unavailable"],
            provenance={"query": "get_restaurant_ops", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetRestaurantOpsInput(BaseModel):
    """No input needed - uses hardcoded demo values"""
    pass


class GetRestaurantOpsTool(BaseTool):
    """LangChain tool wrapper for get_restaurant_ops"""
    name: str = "get_restaurant_ops"
    description: str = "Fetches restaurant operations data from MongoDB (uses hardcoded DEMO_RESTAURANT_ID). Returns prep time metrics, quality ratings, support ticket counts, order volume, and operational status."
    args_schema: Type[BaseModel] = GetRestaurantOpsInput
    
    async def _arun(self) -> str:
        """Async execution - returns JSON string of RestaurantEvidenceEnvelope"""
        result = await get_restaurant_ops()
        return result.model_dump_json()
    
    def _run(self) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
