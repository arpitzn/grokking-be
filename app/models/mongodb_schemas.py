"""MongoDB collection schema definitions and enums"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Support Tickets Enums
# ============================================================================

class TicketType(str, Enum):
    """Support ticket type enum"""
    GENERAL = "general"
    COMPLAINT = "complaint"


class IssueType(str, Enum):
    """Main issue type categories for support tickets"""
    ORDER_ISSUE = "order_issue"
    QUALITY_SAFETY = "quality_safety"
    DELIVERY = "delivery"
    PAYMENT = "payment"
    OPERATIONS = "operations"
    SUPPORT = "support"
    GENERAL = "general"


class OrderIssueSubtype(str, Enum):
    """Subtypes for order_issue category"""
    REFUND = "refund"
    CANCELLATION = "cancellation"
    ORDER_ISSUE = "order_issue"
    DELAY = "delay"
    LATE_DELIVERY = "late_delivery"
    ORDER_STATUS = "order_status"
    MISSING_ITEM = "missing_item"
    WRONG_ORDER = "wrong_order"
    ORDER_MODIFICATION = "order_modification"


class QualitySafetySubtype(str, Enum):
    """Subtypes for quality_safety category"""
    QUALITY = "quality"
    FOOD_SAFETY = "food_safety"
    HYGIENE = "hygiene"
    PACKAGING = "packaging"


class DeliverySubtype(str, Enum):
    """Subtypes for delivery category"""
    DELIVERY = "delivery"
    DELIVERY_PARTNER = "delivery_partner"
    RIDER_ISSUE = "rider_issue"


class PaymentSubtype(str, Enum):
    """Subtypes for payment category"""
    PAYMENT = "payment"
    PRICING = "pricing"


class OperationSubtype(str, Enum):
    """Subtypes for operations category"""
    INCIDENT = "incident"
    ZONE_ISSUE = "zone_issue"
    OUTAGE = "outage"
    SYSTEMIC_PROBLEM = "systemic_problem"
    PERFORMANCE = "performance"


class SupportSubtype(str, Enum):
    """Subtypes for support category"""
    ESCALATION = "escalation"
    COMPENSATION = "compensation"
    SERVICE_RECOVERY = "service_recovery"
    RESPONSE_TIME = "response_time"
    RESOLUTION = "resolution"


class GeneralSubtype(str, Enum):
    """Subtypes for general category"""
    GENERAL_INQUIRY = "general_inquiry"
    ACCOUNT = "account"
    FEEDBACK = "feedback"


class TicketSeverity(int, Enum):
    """Ticket severity levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TicketScope(str, Enum):
    """Ticket scope enum"""
    ORDER = "order"
    ZONE = "zone"
    RESTAURANT = "restaurant"


class TicketStatus(str, Enum):
    """Support ticket status enum"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    RESOLVED = "resolved"


# ============================================================================
# Users Collection Enums
# ============================================================================

class UserPersona(str, Enum):
    """User persona enum"""
    CUSTOMER = "customer"
    AREA_MANAGER = "area_manager"
    CUSTOMER_CARE_REP = "customer_care_rep"


class CustomerSubCategory(str, Enum):
    """Customer sub-category enum (only for persona=customer)"""
    PLATINUM = "platinum"
    STANDARD = "standard"
    HIGH_RISK = "high_risk"


class UserStatus(str, Enum):
    """User status enum"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"


# ============================================================================
# Orders Collection Enums
# ============================================================================

class OrderStatus(str, Enum):
    """Order status enum"""
    PLACED = "placed"
    CONFIRMED = "confirmed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class RefundStatus(str, Enum):
    """Refund status enum"""
    NONE = "none"
    PENDING = "pending"
    ISSUED = "issued"
    COMPLETED = "completed"


class PaymentMethod(str, Enum):
    """Payment method enum"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    WALLET = "wallet"
    CASH = "cash"
    UPI = "upi"
    NET_BANKING = "net_banking"


class PaymentStatus(str, Enum):
    """Payment status enum"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class OrderEventType(str, Enum):
    """Order event type enum"""
    ORDER_PLACED = "order_placed"
    RESTAURANT_CONFIRMED = "restaurant_confirmed"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderEventStatus(str, Enum):
    """Order event status enum"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# ============================================================================
# Zones Collection Enums
# ============================================================================

class ZoneStatus(str, Enum):
    """Zone status enum"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    PAUSED = "paused"


# ============================================================================
# Restaurants Collection Enums
# ============================================================================

class RestaurantType(str, Enum):
    """Restaurant type enum"""
    QUICK_SERVICE = "quick_service"
    CASUAL_DINING = "casual_dining"
    FINE_DINING = "fine_dining"
    CLOUD_KITCHEN = "cloud_kitchen"


class RestaurantStatus(str, Enum):
    """Restaurant status enum"""
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    DELISTED = "delisted"


# ============================================================================
# Pydantic Models for Validation
# ============================================================================

class SubtypeModel(BaseModel):
    """Subtype object model for support_tickets"""
    order_issues: Optional[List[OrderIssueSubtype]] = Field(default=None, description="Order issue subtypes")
    quality_safety: Optional[List[QualitySafetySubtype]] = Field(default=None, description="Quality & safety subtypes")
    delivery: Optional[List[DeliverySubtype]] = Field(default=None, description="Delivery subtypes")
    payment: Optional[List[PaymentSubtype]] = Field(default=None, description="Payment subtypes")
    operation: Optional[List[OperationSubtype]] = Field(default=None, description="Operations subtypes")
    support: Optional[List[SupportSubtype]] = Field(default=None, description="Support subtypes")
    general: Optional[List[GeneralSubtype]] = Field(default=None, description="General subtypes")
    
    @field_validator('order_issues', mode='before')
    @classmethod
    def validate_order_issues(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            # Convert strings to enums if needed
            return [OrderIssueSubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('quality_safety', mode='before')
    @classmethod
    def validate_quality_safety(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [QualitySafetySubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('delivery', mode='before')
    @classmethod
    def validate_delivery(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [DeliverySubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('payment', mode='before')
    @classmethod
    def validate_payment(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [PaymentSubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('operation', mode='before')
    @classmethod
    def validate_operation(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [OperationSubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('support', mode='before')
    @classmethod
    def validate_support(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [SupportSubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    @field_validator('general', mode='before')
    @classmethod
    def validate_general(cls, v):
        """Convert string lists to enum lists"""
        if v is None:
            return None
        if isinstance(v, list):
            return [GeneralSubtype(item) if isinstance(item, str) else item for item in v]
        return v
    
    def model_dump(self, **kwargs) -> Dict[str, Optional[List[str]]]:
        """Custom dump to convert enums to strings"""
        result = {}
        for key, value in self.__dict__.items():
            if value is None:
                result[key] = None
            elif isinstance(value, list):
                result[key] = [item.value if isinstance(item, Enum) else item for item in value]
            else:
                result[key] = value.value if isinstance(value, Enum) else value
        return result


class RefundModel(BaseModel):
    """Refund object model for orders"""
    amount: float = Field(..., ge=0, description="Refund amount")
    status: RefundStatus = Field(..., description="Refund status")
    issued_at: Optional[str] = Field(default=None, description="ISO datetime when refund was issued")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom dump to convert enum to string"""
        result = super().model_dump(**kwargs)
        if isinstance(result.get("status"), Enum):
            result["status"] = result["status"].value
        return result


class PaymentModel(BaseModel):
    """Payment object model for orders"""
    amount: float = Field(..., ge=0, description="Payment amount")
    method: PaymentMethod = Field(..., description="Payment method")
    status: PaymentStatus = Field(..., description="Payment status")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom dump to convert enums to strings"""
        result = super().model_dump(**kwargs)
        if isinstance(result.get("method"), Enum):
            result["method"] = result["method"].value
        if isinstance(result.get("status"), Enum):
            result["status"] = result["status"].value
        return result


class OrderEventModel(BaseModel):
    """Order event model"""
    timestamp: str = Field(..., description="ISO datetime of event")
    event: OrderEventType = Field(..., description="Event type")
    status: OrderEventStatus = Field(..., description="Event status")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom dump to convert enums to strings"""
        result = super().model_dump(**kwargs)
        if isinstance(result.get("event"), Enum):
            result["event"] = result["event"].value
        if isinstance(result.get("status"), Enum):
            result["status"] = result["status"].value
        return result


# ============================================================================
# Validation Helper Functions
# ============================================================================

def validate_subtype_dict(subtype_dict: Dict) -> Dict:
    """Validate subtype dictionary structure"""
    allowed_keys = {
        "order_issues", "quality_safety", "delivery",
        "payment", "operation", "support", "general"
    }
    
    # Check for invalid keys
    invalid_keys = set(subtype_dict.keys()) - allowed_keys
    if invalid_keys:
        raise ValueError(f"Invalid subtype keys: {invalid_keys}")
    
    # Validate each category's values
    subtype_validators = {
        "order_issues": [e.value for e in OrderIssueSubtype],
        "quality_safety": [e.value for e in QualitySafetySubtype],
        "delivery": [e.value for e in DeliverySubtype],
        "payment": [e.value for e in PaymentSubtype],
        "operation": [e.value for e in OperationSubtype],
        "support": [e.value for e in SupportSubtype],
        "general": [e.value for e in GeneralSubtype],
    }
    
    validated = {}
    for key, values in subtype_dict.items():
        if values is None:
            continue
        if not isinstance(values, list):
            raise ValueError(f"subtype.{key} must be a list")
        
        allowed_values = subtype_validators.get(key, [])
        invalid_values = [v for v in values if v not in allowed_values]
        if invalid_values:
            raise ValueError(f"Invalid values in subtype.{key}: {invalid_values}")
        
        validated[key] = values
    
    return validated
