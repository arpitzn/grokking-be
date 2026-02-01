"""
Tool: get_customer_ops_profile
Fetches customer operations profile from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

import logging
from datetime import datetime, timezone
from typing import Type, Union

from bson import Binary, ObjectId
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import CustomerEvidenceEnvelope, ToolResult, ToolStatus
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
    name="get_customer_ops_profile",
    criticality=ToolCriticality.DECISION_CRITICAL
)


async def get_customer_ops_profile(customer_id: str) -> CustomerEvidenceEnvelope:
    """
    Tool Responsibility:
    - Fetches customer operations profile from MongoDB
    - Returns customer history, preferences, and operational metrics
    
    Criticality: decision-critical (declared in TOOL_SPEC)
    Failure handling: Triggers escalation
    
    Observability:
    - Emits tool_call_started, tool_call_completed, tool_call_failed events
    """
    print(f"\n{'='*80}")
    print(f"[TOOL INPUT] get_customer_ops_profile")
    print(f"  customer_id: {customer_id}")
    print(f"{'='*80}\n")
    
    logger.info(f"[get_customer_ops_profile] Starting - customer_id={customer_id}")
    
    emit_tool_event("tool_call_started", {
        "tool_name": "get_customer_ops_profile",
        "params": {"customer_id": customer_id}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert string ID to MongoDB ID (ObjectId or Binary UUID)
        query_user_id = string_to_mongo_id(customer_id)
        logger.debug(f"[get_customer_ops_profile] Query user_id (converted): {query_user_id}, type: {type(query_user_id).__name__}")
        
        # Query user from MongoDB by _id (users collection uses _id as primary key)
        query_filter = {"_id": query_user_id}
        logger.debug(f"[get_customer_ops_profile] MongoDB query filter: {query_filter}")
        
        user_doc = await db.users.find_one(query_filter)
        
        if not user_doc:
            logger.warning(f"[get_customer_ops_profile] User not found - customer_id={customer_id}")
            # User not found - return empty with gap
            return CustomerEvidenceEnvelope(
                source="mongo",
                entity_refs=[customer_id],
                freshness=datetime.now(timezone.utc),
                confidence=0.0,
                data={},
                gaps=["customer_profile_unavailable"],
                provenance={"query": "get_customer_ops_profile", "customer_id": customer_id},
                tool_result=ToolResult(status=ToolStatus.FAILED, error="User not found")
            )
        
        # Transform MongoDB document to tool output format
        # Convert _id to string (handles Binary UUID, ObjectId, or other types)
        _id = user_doc.get("_id")
        if isinstance(_id, Binary):
            user_id_str = binary_to_uuid(_id)
        elif isinstance(_id, ObjectId):
            user_id_str = str(_id)
        else:
            user_id_str = str(_id)
        
        profile_data = {
            "user_id": user_id_str,
            "persona": user_doc.get("persona"),
            "sub_category": user_doc.get("sub_category"),
            "total_orders": user_doc.get("total_orders"),
            "lifetime_value": user_doc.get("lifetime_value"),
            "avg_order_value": user_doc.get("avg_order_value"),
            "refund_count": user_doc.get("refund_count"),
            "refund_rate": user_doc.get("refund_rate"),
            "last_order_date": safe_isoformat(user_doc.get("last_order_date")),
            "preferred_cuisines": user_doc.get("preferred_cuisines", []),
            "vip_status": user_doc.get("vip_status", False)
        }
        
        logger.info(f"[get_customer_ops_profile] Success - persona={profile_data.get('persona')}, "
                   f"total_orders={profile_data.get('total_orders')}, lifetime_value={profile_data.get('lifetime_value')}, "
                   f"refund_rate={profile_data.get('refund_rate')}, vip_status={profile_data.get('vip_status')}")
        logger.debug(f"[get_customer_ops_profile] Output data: {profile_data}")
        
        result = CustomerEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.92,
            data=profile_data,
            gaps=[],
            provenance={"query": "get_customer_ops_profile", "customer_id": customer_id, "latency_ms": 45},
            tool_result=ToolResult(status=ToolStatus.SUCCESS, data=profile_data)
        )
        
        emit_tool_event("tool_call_completed", {
            "tool_name": "get_customer_ops_profile",
            "status": "success"
        })
        
        print(f"\n{'='*80}")
        print(f"[TOOL OUTPUT] get_customer_ops_profile - SUCCESS")
        print(f"  user_id: {profile_data.get('user_id')}")
        print(f"  persona: {profile_data.get('persona')}")
        print(f"  total_orders: {profile_data.get('total_orders')}")
        print(f"  vip_status: {profile_data.get('vip_status')}")
        print(f"{'='*80}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"[get_customer_ops_profile] Error - customer_id={customer_id}, error={str(e)}", exc_info=True)
        
        emit_tool_event("tool_call_failed", {
            "tool_name": "get_customer_ops_profile",
            "error": str(e)
        })
        
        return CustomerEvidenceEnvelope(
            source="mongo",
            entity_refs=[customer_id],
            freshness=datetime.now(timezone.utc),
            confidence=0.0,
            data={},
            gaps=["customer_profile_unavailable"],
            provenance={"query": "get_customer_ops_profile", "error": str(e)},
            tool_result=ToolResult(status=ToolStatus.FAILED, error=str(e))
        )


# LangChain BaseTool wrapper
class GetCustomerOpsProfileInput(BaseModel):
    """Input schema for get_customer_ops_profile tool"""
    customer_id: str = Field(description="Customer ID to fetch operations profile for")


class GetCustomerOpsProfileTool(BaseTool):
    """LangChain tool wrapper for get_customer_ops_profile"""
    name: str = "get_customer_ops_profile"
    description: str = "Fetches customer operations profile from MongoDB. Returns customer history, preferences, lifetime value, refund history, and operational metrics."
    args_schema: Type[BaseModel] = GetCustomerOpsProfileInput
    
    async def _arun(self, customer_id: str) -> str:
        """Async execution - returns JSON string of CustomerEvidenceEnvelope"""
        result = await get_customer_ops_profile(customer_id)
        return result.model_dump_json()
    
    def _run(self, customer_id: str) -> dict:
        """Sync execution - not supported for async tools"""
        raise NotImplementedError("This tool only supports async execution")
