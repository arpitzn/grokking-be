"""Persona-aware customer ID resolution"""
from typing import Optional, Dict, Any
from app.infra.demo_constants import DEMO_CUSTOMER_ID


def resolve_customer_id(
    case: Dict[str, Any],
    extracted_customer_id: Optional[str] = None
) -> str:
    """
    Resolve customer_id based on persona.
    
    Logic:
    - customer: user_id IS the customer_id
    - agent + extracted: use extracted customer_id
    - agent + no extraction: use DEMO_CUSTOMER_ID
    
    Args:
        case: Case dict with persona, user_id
        extracted_customer_id: Customer ID extracted from query (if any)
        
    Returns:
        Resolved customer_id to use for tools
    """
    persona = case.get("persona", "customer")
    user_id = case.get("user_id", "")
    
    if persona == "customer":
        return user_id
    else:
        # Agent personas: use extracted or fallback to demo
        return extracted_customer_id or DEMO_CUSTOMER_ID
