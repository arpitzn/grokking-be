"""Escalation endpoint for human handover"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import EscalationResponse, HandoverPacket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/escalations", tags=["escalations"])


@router.post("", response_model=EscalationResponse)
async def create_escalation(handover_packet: HandoverPacket):
    """
    Receives escalation requests from human_escalation_agent.
    
    Creates an escalation record and returns escalation_id.
    """
    try:
        # Generate escalation ID
        escalation_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        
        # Log escalation (in production, would store in MongoDB)
        logger.info(
            f"Escalation created: {escalation_id}",
            extra={
                "escalation_id": escalation_id,
                "case_id": handover_packet.case_id,
                "issue_type": handover_packet.issue_type,
                "severity": handover_packet.severity,
                "SLA_risk": handover_packet.SLA_risk
            }
        )
        
        # In production, would:
        # 1. Store handover_packet in MongoDB escalations collection
        # 2. Notify human agents via notification system
        # 3. Update case status
        
        return EscalationResponse(
            escalation_id=escalation_id,
            status="created",
            case_id=handover_packet.case_id,
            message=f"Escalation {escalation_id} created successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to create escalation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create escalation: {str(e)}")
