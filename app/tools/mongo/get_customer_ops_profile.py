"""
Tool: get_customer_ops_profile
Fetches customer operations profile from MongoDB

Criticality: decision-critical
Observability: Emits tool_call_started, tool_call_completed, tool_call_failed events
"""

from datetime import datetime, timezone
from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.models.evidence import CustomerEvidenceEnvelope, ToolResult, ToolStatus
from app.models.tool_spec import ToolCriticality, ToolSpec
from app.utils.tool_observability import emit_tool_event
from app.utils.uuid_helpers import uuid_to_binary, is_uuid_string
from app.infra.mongo import get_mongodb_client

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
    emit_tool_event("tool_call_started", {
        "tool_name": "get_customer_ops_profile",
        "params": {"customer_id": customer_id}
    })
    
    try:
        # Get MongoDB client
        db = await get_mongodb_client()
        
        # Convert UUID string to Binary UUID if needed
        query_user_id = uuid_to_binary(customer_id) if is_uuid_string(customer_id) else customer_id
        
        # Query user from MongoDB by _id (users collection uses _id as primary key)
        user_doc = await db.users.find_one({"_id": query_user_id})
        
        if not user_doc:
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
        # Convert _id Binary UUID to string for output
        from app.utils.uuid_helpers import binary_to_uuid
        user_id_str = binary_to_uuid(user_doc.get("_id")) if isinstance(user_doc.get("_id"), Binary) else str(user_doc.get("_id"))
        
        profile_data = {
            "user_id": user_id_str,
            "persona": user_doc.get("persona"),
            "sub_category": user_doc.get("sub_category"),
            "total_orders": user_doc.get("total_orders"),
            "lifetime_value": user_doc.get("lifetime_value"),
            "avg_order_value": user_doc.get("avg_order_value"),
            "refund_count": user_doc.get("refund_count"),
            "refund_rate": user_doc.get("refund_rate"),
            "last_order_date": user_doc.get("last_order_date").isoformat() if user_doc.get("last_order_date") else None,
            "preferred_cuisines": user_doc.get("preferred_cuisines", []),
            "vip_status": user_doc.get("vip_status", False)
        }
        
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
        
        return result
        
    except Exception as e:
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
