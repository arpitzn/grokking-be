"""Pydantic models for API request/response schemas"""

from typing import List, Optional

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
