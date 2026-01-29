"""Knowledge upload endpoint for RAG ingestion"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import KnowledgeUploadRequest, KnowledgeUploadResponse
from app.services.knowledge import ingest_document
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge(request: KnowledgeUploadRequest):
    """
    Upload a document for RAG ingestion
    
    Accepts text content, chunks it, embeds it, and stores in Elasticsearch
    """
    try:
        # Decode content if base64 (simplified - assume raw text for now)
        content = request.content
        
        # Ingest document
        result = await ingest_document(
            user_id=request.user_id,
            file_content=content,
            filename=request.filename
        )
        
        return KnowledgeUploadResponse(
            document_id=result.get("document_id", "unknown"),
            chunk_count=result.get("chunk_count", 0),
            status=result.get("status", "success")
        )
    except Exception as e:
        logger.error(f"Knowledge upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def list_knowledge(user_id: str):
    """
    List uploaded documents for a user
    
    Note: Simplified - returns basic info. In production, track documents in MongoDB
    """
    # TODO: Implement document tracking in MongoDB
    return {"user_id": user_id, "documents": []}
