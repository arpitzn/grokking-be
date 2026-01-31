"""Evidence envelope models - typed evidence containers for all tools"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolResult(BaseModel):
    """Structural tool result - represents success or failure"""
    status: ToolStatus
    error: Optional[str] = None  # Populated if status == FAILED
    data: Optional[Any] = None   # Populated if status == SUCCESS


class EvidenceEnvelope(BaseModel):
    """Base evidence envelope - all tools return this structure"""
    source: str  # "mongo", "elasticsearch", "mem0"
    entity_refs: List[str]  # order_id, user_id, etc. (customer_id kept for backward compatibility in tool params)
    freshness: datetime
    confidence: float  # 0.0 to 1.0
    data: Dict[str, Any]  # Typed payload
    gaps: List[str]  # Missing information
    provenance: Dict[str, Any]  # query, filters, latency
    tool_result: ToolResult  # Structural success/failure representation


class OrderEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for order-related evidence"""
    pass  # data contains OrderTimeline structure


class CustomerEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for customer operations profile"""
    pass  # data contains CustomerOpsProfile structure


class ZoneEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for zone operations metrics"""
    pass  # data contains ZoneOpsMetrics structure


class IncidentEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for incident signals"""
    pass  # data contains IncidentSignals structure


class RestaurantEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for restaurant operations"""
    pass  # data contains RestaurantOps structure


class CaseEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for aggregated case context"""
    pass  # data contains CaseContext structure


class PolicyEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for policy evidence"""
    pass  # data contains PolicySearchResults structure


class MemoryEvidenceEnvelope(EvidenceEnvelope):
    """Specific envelope for memory (episodic/semantic) evidence"""
    pass  # data contains MemoryResults structure
