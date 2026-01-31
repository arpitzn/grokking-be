"""Tool specification models - minimal criticality declaration"""

from enum import Enum

from pydantic import BaseModel


class ToolCriticality(str, Enum):
    """Tool criticality levels - determines failure handling"""
    NON_CRITICAL = "non_critical"  # Failure allows partial results, continue
    DECISION_CRITICAL = "decision_critical"  # Failure triggers escalation
    SAFETY_CRITICAL = "safety_critical"  # Failure blocks execution


class ToolSpec(BaseModel):
    """
    Minimal tool specification - every tool declares this.
    
    Agents may escalate criticality contextually but never downgrade.
    Example: get_order_timeline is decision-critical, but if order_id is missing,
    agent may escalate to safety-critical.
    """
    name: str  # Tool function name
    criticality: ToolCriticality  # Minimum criticality level
