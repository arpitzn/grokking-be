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
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import List, Optional
from pydantic import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/upload-multiple", response_model=List[KnowledgeUploadResponse])
async def upload_multiple_files(
    es_client: ElasticsearchDep,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    category: str = Form(...),
    persona: str = Form(...),  # Uploader persona (area_manager, customer_care_rep, end_customer)
    subcategory: Optional[str] = Form(None),  # Subcategory for end_customer
    persona_filter: str = Form(...),  # Document persona filter (JSON array string)
    issue_type: str = Form(...),  # JSON array string (shared for all files)
    priority: str = Form(...),
    doc_weight: float = Form(...),
):
    """
    Upload multiple files for RAG ingestion using multipart/form-data

    Processes each file independently with shared filters, continues on errors,
    returns per-file results
    
    Access control:
    - End Customer cannot upload documents (403 error)
    """
    start_time = time.time()

    log_request_start(
        logger,
        "POST",
        "/knowledge/upload-multiple",
        user_id=user_id,
        body={"file_count": len(files)},
    )

    # Access control: End Customer cannot upload
    if persona == 'end_customer':
        raise HTTPException(status_code=403, detail="End Customer cannot upload documents")

    # Parse and validate filters once (shared for all files)
    try:
        persona_list = json.loads(persona_filter)
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
            # If missing or generic binary type, try to guess from filename
            if not mime_type or mime_type == "application/octet-stream":
                guessed_type, _ = mimetypes.guess_type(file.filename or "")
                if guessed_type:
                    mime_type = guessed_type
                elif not mime_type:  # Only use octet-stream if we had no content_type at all
                    mime_type = "application/octet-stream"
            
            # Ingest file with shared filters
            result = await ingest_file(
                user_id=user_id,
                file_content=file_content,
                filename=file.filename or "unknown",
                mime_type=mime_type,
                filters=filters,  # Shared filters for all files
                uploader_persona=persona,  # Uploader persona
                es_client=es_client,
                uploader_subcategory=subcategory,  # Uploader subcategory (if applicable)
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
async def list_knowledge(
    user_id: str,
    es_client: ElasticsearchDep,
    persona: str = Query(...),
    subcategory: Optional[str] = Query(None),
):
    """
    List all uploaded documents for a user with filters

    Aggregates chunks by file and returns one entry per file
    Access control based on persona:
    - area_manager: sees all documents (regardless of persona filter)
    - customer_care_rep: sees only documents where persona array contains "customer_care_rep"
    - end_customer: no access (403 error)
    """
    start_time = time.time()
    log_request_start(logger, "GET", f"/knowledge/{user_id}", user_id=user_id)

    # Validate persona
    if persona not in ['area_manager', 'customer_care_rep', 'end_customer']:
        raise HTTPException(status_code=403, detail=f"Invalid persona: {persona}")

    # End Customer: No access
    if persona == 'end_customer':
        raise HTTPException(status_code=403, detail="End Customer cannot access documents")

    try:
        # Fetch documents based on persona
        if persona == 'area_manager':
            # Fetch all documents (no persona filter)
            documents = await es_client.list_all_documents()
        elif persona == 'customer_care_rep':
            # Fetch only documents where persona array contains "customer_care_rep"
            documents = await es_client.list_documents_by_persona("customer_care_rep")

        log_request_end(
            logger,
            "GET",
            f"/knowledge/{user_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            user_id=user_id,
            details={"document_count": len(documents), "persona": persona}
        )

        return documents
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "list_documents_error", context={"user_id": user_id, "persona": persona}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}/file/{file_id}", response_model=DeleteFileResponse)
async def delete_file(
    user_id: str,
    file_id: str,
    es_client: ElasticsearchDep,
    persona: str = Query(...),
):
    """
    Delete a single file and all its chunks from Elasticsearch
    
    Reconstructs full file_id as {user_id}_{file_id} to match stored format
    Access control based on persona:
    - area_manager: can delete any document
    - customer_care_rep: can only delete documents where persona array contains "customer_care_rep"
    """
    start_time = time.time()
    # Reconstruct full file_id: user_id_filename_timestamp
    full_file_id = f"{user_id}_{file_id}"
    
    log_request_start(logger, "DELETE", f"/knowledge/{user_id}/file/{file_id}", user_id=user_id)

    # Validate persona
    if persona not in ['area_manager', 'customer_care_rep', 'end_customer']:
        raise HTTPException(status_code=403, detail=f"Invalid persona: {persona}")

    # End Customer: No access
    if persona == 'end_customer':
        raise HTTPException(status_code=403, detail="End Customer cannot delete documents")

    try:
        # For customer_care_rep, check if document's persona array contains "customer_care_rep"
        if persona == 'customer_care_rep':
            doc = await es_client.get_document_by_file_id(full_file_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
            
            doc_personas = doc.get("persona", [])
            # Normalize personas to lowercase for comparison
            doc_personas_lower = [p.lower() if isinstance(p, str) else p for p in doc_personas]
            
            if "customer_care_rep" not in doc_personas_lower:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: Document persona filter does not include 'customer_care_rep'"
                )
        
        # Proceed with deletion
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
    es_client: ElasticsearchDep,
    persona: str = Query(...),
    user_id: Optional[str] = Query(None),
):
    """
    Delete all files and chunks from Elasticsearch
    
    Access control based on persona:
    - area_manager: deletes all documents (global delete)
    - customer_care_rep: deletes only documents where persona array contains "customer_care_rep"
    """
    start_time = time.time()
    log_request_start(logger, "DELETE", "/knowledge/all")

    # Validate persona
    if persona not in ['area_manager', 'customer_care_rep', 'end_customer']:
        raise HTTPException(status_code=403, detail=f"Invalid persona: {persona}")

    # End Customer: No access
    if persona == 'end_customer':
        raise HTTPException(status_code=403, detail="End Customer cannot delete documents")

    try:
        # Delete based on persona
        if persona == 'area_manager':
            # Delete all documents (global delete)
            result = await es_client.delete_all_files()
        elif persona == 'customer_care_rep':
            # Delete only documents where persona array contains "customer_care_rep"
            result = await es_client.delete_files_by_persona("customer_care_rep")

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
