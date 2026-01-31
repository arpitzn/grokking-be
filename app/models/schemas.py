"""Pydantic models for API request/response schemas"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""

    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID (creates new if None)"
    )
    message: str = Field(..., min_length=1, max_length=4000, description="User message")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""

    conversation_id: str
    message_id: str
    status: str = "streaming"  # "streaming" | "completed" | "error"


class ConversationListItem(BaseModel):
    """Schema for conversation list item"""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageItem(BaseModel):
    """Schema for message item"""

    message_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: str
    metadata: Optional[dict] = None


class KnowledgeUploadRequest(BaseModel):
    """Request schema for knowledge upload"""

    user_id: str
    filename: str
    content: str  # Base64 encoded or raw text


class KnowledgeUploadResponse(BaseModel):
    """Response schema for knowledge upload"""

    document_id: Optional[str] = (
        None  # None when batch indexing doesn't return individual IDs
    )
    chunk_count: int
    status: str = "success"


class HealthCheckResponse(BaseModel):
    """Response schema for health check"""

    status: str  # "healthy" | "degraded" | "unhealthy"
    checks: dict
    timestamp: str


# Food Delivery Domain Schemas

class CaseRequest(BaseModel):
    """Request schema for food delivery case endpoint (replaces ChatRequest)"""

    user_id: str = Field(..., description="User identifier")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID (creates new if None)"
    )
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    persona: Optional[str] = Field("customer", description="Persona: customer, agent, admin")
    channel: Optional[str] = Field("web", description="Channel: web, mobile, phone, chat")


class HandoverPacket(BaseModel):
    """Schema for human escalation handover packet"""

    case_id: str
    customer_id: str
    order_id: Optional[str] = None
    issue_type: str
    severity: str
    SLA_risk: bool
    safety_flags: List[str] = []
    evidence_summary: Dict[str, Any]
    analysis: Dict[str, Any]
    guardrails: Dict[str, Any]
    raw_text: str
    cot_trace: List[Dict[str, Any]] = []  # turn can be int, phase/content are str


class EscalationResponse(BaseModel):
    """Response schema for escalation endpoint"""

    escalation_id: str
    status: str = "created"
    case_id: str
    message: str = "Escalation created successfully"


class EvidenceCard(BaseModel):
    """Schema for evidence card UI event"""

    source: str  # "mongo", "elasticsearch", "mem0"
    title: str
    content: Dict[str, Any]
    confidence: float
    citations: List[str] = []


class RefundRecommendation(BaseModel):
    """Schema for refund recommendation UI event"""

    recommended: bool
    amount: Optional[float] = None
    rationale: str
    policy_reference: Optional[str] = None


class IncidentBanner(BaseModel):
    """Schema for incident banner UI event"""

    incident_id: str
    severity: str
    message: str
    action_required: bool
