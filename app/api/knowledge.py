"""Knowledge upload endpoint for RAG ingestion"""

import json
import logging
import time
import mimetypes

from app.infra.elasticsearch import ElasticsearchDep
from app.models.schemas import KnowledgeUploadResponse, FileUploadResult, DocumentListItem, DeleteFileResponse, DeleteAllResponse
from app.models.filters import DocumentFilters, Persona, Priority, Category
from app.services.knowledge import ingest_file
from app.utils.logging_utils import (
    log_business_milestone,
    log_db_operation,
    log_error_with_context,
    log_request_end,
    log_request_start,
)
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from pydantic import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/upload-multiple", response_model=List[KnowledgeUploadResponse])
async def upload_multiple_files(
    es_client: ElasticsearchDep,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    category: str = Form(...),
    persona: str = Form(...),  # JSON array string (shared for all files)
    issue_type: str = Form(...),  # JSON array string (shared for all files)
    priority: str = Form(...),
    doc_weight: float = Form(...),
):
    """
    Upload multiple files for RAG ingestion using multipart/form-data

    Processes each file independently with shared filters, continues on errors,
    returns per-file results
    """
    start_time = time.time()

    log_request_start(
        logger,
        "POST",
        "/knowledge/upload-multiple",
        user_id=user_id,
        body={"file_count": len(files)},
    )

    # Parse and validate filters once (shared for all files)
    try:
        persona_list = json.loads(persona)
        issue_type_list = json.loads(issue_type)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in filter arrays: {e}")
    
    try:
        filters = DocumentFilters(
            category=category,
            persona=persona_list,
            issue_type=issue_type_list,
            priority=priority,
            doc_weight=float(doc_weight)  # Ensure float type
        )
    except ValidationError as e:
        error_details = [{"field": err["loc"][0], "message": err["msg"]} for err in e.errors()]
        raise HTTPException(status_code=400, detail=f"Invalid filter values: {error_details}")

    results = []
    
    for file in files:
        file_start = time.time()
        try:
            # Read file content
            file_content = await file.read()
            
            # Get MIME type
            mime_type = file.content_type
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(file.filename or "")
                mime_type = mime_type or "application/octet-stream"
            
            # Ingest file with shared filters
            result = await ingest_file(
                user_id=user_id,
                file_content=file_content,
                filename=file.filename or "unknown",
                mime_type=mime_type,
                filters=filters,  # Shared filters for all files
                es_client=es_client,
            )
            
            results.append(KnowledgeUploadResponse(
                file_id=result.get("file_id", "unknown"),
                chunk_count=result.get("chunk_count", 0),
                status=result.get("status", "success"),
                error=result.get("error"),
            ))
            
            logger.info(json.dumps({
                "event": "file_processed_in_batch",
                "user_id": user_id,
                "filename": file.filename,
                "file_id": result.get("file_id"),
                "status": result.get("status"),
                "duration_ms": round((time.time() - file_start) * 1000, 2)
            }))
        
        except Exception as e:
            # Log error but continue processing other files
            log_error_with_context(
                logger,
                e,
                "file_upload_error_in_batch",
                context={
                    "user_id": user_id,
                    "filename": file.filename,
                },
            )
            
            results.append(KnowledgeUploadResponse(
                file_id=f"{user_id}_{file.filename}_{int(time.time())}" if file.filename else "unknown",
                chunk_count=0,
                status="failed",
                error=str(e),
            ))

    log_request_end(
        logger,
        "POST",
        "/knowledge/upload-multiple",
        status_code=200,
        duration_ms=(time.time() - start_time) * 1000,
        details={"total_files": len(files), "successful": sum(1 for r in results if r.status == "success")},
        user_id=user_id,
    )

    return results


@router.get("/{user_id}", response_model=List[DocumentListItem])
async def list_knowledge(user_id: str, es_client: ElasticsearchDep):
    """
    List all uploaded documents for a user with filters

    Aggregates chunks by file and returns one entry per file
    """
    start_time = time.time()
    log_request_start(logger, "GET", f"/knowledge/{user_id}", user_id=user_id)

    try:
        documents = await es_client.list_documents_by_user(user_id)

        log_request_end(
            logger,
            "GET",
            f"/knowledge/{user_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            user_id=user_id,
            details={"document_count": len(documents)}
        )

        return documents
    except Exception as e:
        log_error_with_context(
            logger, e, "list_documents_error", context={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/file/{file_id}", response_model=DeleteFileResponse)
async def delete_file(
    user_id: str,
    file_id: str,
    es_client: ElasticsearchDep
):
    """
    Delete a single file and all its chunks from Elasticsearch
    
    Reconstructs full file_id as {user_id}_{file_id} to match stored format
    """
    start_time = time.time()
    # Reconstruct full file_id: user_id_filename_timestamp
    full_file_id = f"{user_id}_{file_id}"
    
    log_request_start(logger, "DELETE", f"/knowledge/{user_id}/file/{file_id}", user_id=user_id)

    try:
        result = await es_client.delete_file_by_id(full_file_id)

        log_request_end(
            logger,
            "DELETE",
            f"/knowledge/{user_id}/file/{file_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            user_id=user_id,
            details={"file_id": full_file_id, "chunks_deleted": result["deleted"]}
        )

        return DeleteFileResponse(
            file_id=full_file_id,
            deleted=result["deleted"],
            status="success"
        )
    except Exception as e:
        log_error_with_context(
            logger, e, "delete_file_error", context={"file_id": full_file_id, "user_id": user_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all", response_model=DeleteAllResponse)
async def delete_all_files(
    es_client: ElasticsearchDep
):
    """
    Delete all files and chunks from Elasticsearch (global delete)
    """
    start_time = time.time()
    log_request_start(logger, "DELETE", "/knowledge/all")

    try:
        result = await es_client.delete_all_files()

        log_request_end(
            logger,
            "DELETE",
            "/knowledge/all",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"chunks_deleted": result["deleted"]}
        )

        return DeleteAllResponse(
            deleted=result["deleted"],
            status="success"
        )
    except Exception as e:
        log_error_with_context(
            logger, e, "delete_all_files_error"
        )
        raise HTTPException(status_code=500, detail=str(e))
