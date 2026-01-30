"""Knowledge upload endpoint for RAG ingestion"""

import json
import logging
import time

from app.infra.elasticsearch import ElasticsearchDep
from app.models.schemas import KnowledgeUploadRequest, KnowledgeUploadResponse
from app.services.knowledge import ingest_document
from app.utils.logging_utils import (
    log_business_milestone,
    log_db_operation,
    log_error_with_context,
    log_request_end,
    log_request_start,
)
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge(
    request: KnowledgeUploadRequest, es_client: ElasticsearchDep
):
    """
    Upload a document for RAG ingestion

    Accepts text content, chunks it, embeds it, and stores in Elasticsearch
    """
    start_time = time.time()

    log_request_start(
        logger,
        "POST",
        "/knowledge/upload",
        user_id=request.user_id,
        body={"filename": request.filename, "content_length": len(request.content)},
    )

    try:
        log_business_milestone(
            logger,
            "document_ingestion_start",
            user_id=request.user_id,
            details={
                "filename": request.filename,
                "content_length": len(request.content),
            },
        )

        # Decode content if base64 (simplified - assume raw text for now)
        content = request.content

        # Ingest document
        result = await ingest_document(
            user_id=request.user_id,
            file_content=content,
            filename=request.filename,
            es_client=es_client,
        )

        log_business_milestone(
            logger,
            "document_ingestion_complete",
            user_id=request.user_id,
            details={
                "filename": request.filename,
                "chunk_count": result.get("chunk_count", 0),
                "status": result.get("status"),
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        log_request_end(
            logger,
            "POST",
            "/knowledge/upload",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"chunk_count": result.get("chunk_count")},
            user_id=request.user_id,
        )

        return KnowledgeUploadResponse(
            document_id=result.get("document_id")
            or "unknown",  # Handle None from batch indexing
            chunk_count=result.get("chunk_count", 0),
            status=result.get("status", "success"),
        )
    except Exception as e:
        log_error_with_context(
            logger,
            e,
            "knowledge_upload_error",
            context={
                "user_id": request.user_id,
                "filename": request.filename,
                "content_length": len(request.content),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}")
async def list_knowledge(user_id: str):
    """
    List uploaded documents for a user

    Note: Simplified - returns basic info. In production, track documents in MongoDB
    """
    start_time = time.time()
    log_request_start(logger, "GET", f"/knowledge/{user_id}", user_id=user_id)

    # TODO: Implement document tracking in MongoDB
    log_db_operation(
        logger, "find", "documents", result_count=0, expected=False, user_id=user_id
    )

    logger.warning(
        json.dumps(
            {
                "event": "feature_not_implemented",
                "endpoint": f"/knowledge/{user_id}",
                "message": "Document tracking not implemented",
            }
        )
    )

    log_request_end(
        logger,
        "GET",
        f"/knowledge/{user_id}",
        status_code=200,
        duration_ms=(time.time() - start_time) * 1000,
        user_id=user_id,
    )

    return {"user_id": user_id, "documents": []}
