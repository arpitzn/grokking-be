"""Filter schema and validation for document uploads"""
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    """Document category enum"""
    POLICY = "policy"
    SLA = "sla"
    SOP = "sop"
    FAQ = "faq"
    TEMPLATES = "templates"
    ESCALATION = "escalation"
    ZONE_RULES = "zone_rules"


class Persona(str, Enum):
    """Persona enum"""
    CUSTOMER_CARE_REP = "customer_care_rep"
    END_CUSTOMER = "end_customer"
    AREA_MANAGER = "area_manager"


class Priority(str, Enum):
    """Priority enum"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Issue types list (30+ types)
ISSUE_TYPES = [
    # Order Issues
    "refund",
    "cancellation",
    "order_issue",
    "delay",
    "late_delivery",
    "order_status",
    "missing_item",
    "wrong_order",
    "order_modification",
    # Quality & Safety
    "quality",
    "food_safety",
    "hygiene",
    "packaging",
    # Delivery
    "delivery",
    "delivery_partner",
    "rider_issue",
    # Payment
    "payment",
    "pricing",
    # Operations
    "incident",
    "zone_issue",
    "outage",
    "systemic_problem",
    "performance",
    # Support
    "escalation",
    "compensation",
    "service_recovery",
    "response_time",
    "resolution",
    # General
    "general_inquiry",
    "account",
    "feedback",
]


class DocumentFilters(BaseModel):
    """Filter schema for document uploads"""
    category: Category
    persona: List[Persona] = Field(..., min_length=1, description="At least one persona required")
    issue_type: List[str] = Field(..., min_length=1, description="At least one issue type required")
    priority: Priority
    doc_weight: float = Field(..., ge=1.0, le=3.0, description="Document weight between 1.0 and 3.0")
    
    @field_validator('issue_type')
    @classmethod
    def validate_issue_types(cls, v: List[str]) -> List[str]:
        """Validate issue types against allowed list"""
        invalid = [it for it in v if it not in ISSUE_TYPES]
        if invalid:
            raise ValueError(f"Invalid issue types: {invalid}")
        return v
    
    @field_validator('doc_weight')
    @classmethod
    def validate_doc_weight(cls, v: float) -> float:
        """Round doc_weight to nearest 0.5"""
        return round(v * 2) / 2
    
    @field_validator('persona', mode='before')
    @classmethod
    def validate_persona(cls, v):
        """Validate and convert persona strings to Persona enum"""
        if isinstance(v, list):
            try:
                return [Persona(p) if isinstance(p, str) else p for p in v]
            except ValueError as e:
                raise ValueError(f"Invalid persona values: {e}")
        return v
