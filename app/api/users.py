"""User endpoints"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import UserByPersonaRequest, UserByPersonaResponse
from app.services.user import get_random_user_by_persona
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_error_with_context
)
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/by-persona", response_model=UserByPersonaResponse)
async def get_user_by_persona(request: UserByPersonaRequest):
    """
    Resolve persona to a random user_id from MongoDB
    
    - For area_manager and customer_care_rep: filters by persona only
    - For end_customer: filters by persona + sub_category (defaults to "standard" if not provided)
    - Returns random active user matching criteria
    - Returns null user_id if no users found
    """
    start_time = time.time()
    log_request_start(
        logger, 
        "POST", 
        "/users/by-persona",
        body={"persona": request.persona, "sub_category": request.sub_category}
    )
    
    # Validate persona
    valid_personas = ["area_manager", "customer_care_rep", "end_customer"]
    if request.persona not in valid_personas:
        log_error_with_context(
            logger, 
            ValueError(f"Invalid persona: {request.persona}"),
            "invalid_persona",
            context={"persona": request.persona}
        )
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid persona. Must be one of: {', '.join(valid_personas)}"
        )
    
    # Ignore sub_category if provided for non-end_customer personas
    sub_category = request.sub_category if request.persona == "end_customer" else None
    
    try:
        user_data = await get_random_user_by_persona(request.persona, sub_category)
        
        # Build response
        response = UserByPersonaResponse(
            user_id=user_data["user_id"] if user_data else None,
            persona=request.persona,
            sub_category=sub_category if request.persona == "end_customer" else None
        )
        
        log_request_end(
            logger,
            "POST",
            "/users/by-persona",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={
                "persona": request.persona,
                "sub_category": sub_category,
                "user_id": response.user_id,
                "found": user_data is not None
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, 
            e, 
            "get_user_by_persona_error",
            context={"persona": request.persona, "sub_category": request.sub_category}
        )
        raise HTTPException(status_code=500, detail=str(e))
