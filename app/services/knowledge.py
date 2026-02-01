"""Knowledge service for RAG ingestion and retrieval"""
from app.infra.elasticsearch import ElasticsearchDep, ElasticsearchClient
from app.infra.llm import get_llm_service
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
# from app.infra.llm import get_llm_client
from app.services.processors.factory import get_processor
from app.models.filters import DocumentFilters
from app.utils.logging_utils import log_business_milestone, log_error_with_context
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from typing import List, Dict, Any, Optional
# from datetime import datetime
import logging
import json
import time

logger = logging.getLogger(__name__)


async def ingest_document(
    user_id: str,
    file_content: str,
    filename: str,
    es_client: ElasticsearchClient
) -> Dict[str, Any]:
    """
    Ingest a document into Elasticsearch for RAG
    
    Steps:
    1. Parse text content
    2. Chunk with RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    3. Embed each chunk with OpenAI text-embedding-3-small
    4. Store in Elasticsearch with user_id metadata using batch operations
    """
    start_time = time.time()
    llm_service = get_llm_service()
    
    # Chunk the document
    chunk_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(file_content)
    chunk_duration = (time.time() - chunk_start) * 1000
    
    logger.info(json.dumps({
        "event": "document_chunked",
        "user_id": user_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "duration_ms": round(chunk_duration, 2)
    }))
    
    # Collect all documents for batch indexing
    embed_start = time.time()
    documents = []
    for idx, chunk in enumerate(chunks):
        # Create embedding
        embedding = await llm_service.embeddings(chunk, model="text-embedding-3-small")
        
        documents.append({
            "user_id": user_id,
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        })
    embed_duration = (time.time() - embed_start) * 1000
    
    logger.info(json.dumps({
        "event": "embeddings_generated",
        "user_id": user_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "duration_ms": round(embed_duration, 2),
        "avg_ms_per_chunk": round(embed_duration / len(chunks), 2) if chunks else 0
    }))
    
    # Batch index all documents
    index_start = time.time()
    results = await es_client.batch_index_documents(documents)
    index_duration = (time.time() - index_start) * 1000
    
    logger.info(json.dumps({
        "event": "document_indexed",
        "user_id": user_id,
        "filename": filename,
        "successful": results['successful'],
        "failed": results['failed'],
        "total": results['total'],
        "duration_ms": round(index_duration, 2),
        "total_duration_ms": round((time.time() - start_time) * 1000, 2)
    }))
    
    return {
        "document_id": None,  # Batch indexing doesn't return individual IDs
        "chunk_count": results["successful"],
        "status": "success" if results["failed"] == 0 else "partial",
        "failed_count": results["failed"]
    }


async def ingest_file(
    user_id: str,
    file_content: bytes,
    filename: str,
    mime_type: str,
    filters: DocumentFilters,
    uploader_persona: str,
    es_client: ElasticsearchClient,
    uploader_subcategory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a file into Elasticsearch for RAG using processor factory
    
    Steps:
    1. Get processor from factory based on MIME type
    2. Extract and chunk content via processor
    3. Generate batch embeddings (all chunks at once)
    4. Index to Elasticsearch with full metadata and filters
    
    Args:
        user_id: User identifier
        file_content: Raw file bytes
        filename: Original filename
        mime_type: MIME type string (e.g., "application/pdf")
        filters: DocumentFilters with category, persona, issue_type, priority, doc_weight
        es_client: Elasticsearch client
        
    Returns:
        Dict with file_id, chunk_count, status, etc.
    """
    start_time = time.time()
    
    # Generate file_id (filename-based: user_id_filename_timestamp)
    timestamp = int(time.time())
    file_id = f"{user_id}_{filename}_{timestamp}"
    
    log_business_milestone(
        logger,
        "file_ingestion_start",
        user_id=user_id,
        details={
            "filename": filename,
            "mime_type": mime_type,
            "file_id": file_id,
            "file_size": len(file_content),
        },
    )
    
    try:
        # Get processor from factory
        processor = get_processor(mime_type, filename)
        if not processor:
            raise ValueError(f"No processor found for MIME type: {mime_type}")
        
        # Process file (extract and chunk)
        process_start = time.time()
        processed_content = await processor.process(file_content, filename)
        process_duration = (time.time() - process_start) * 1000
        
        logger.info(json.dumps({
            "event": "file_processed",
            "user_id": user_id,
            "filename": filename,
            "file_id": file_id,
            "chunk_count": len(processed_content.chunks),
            "method": processed_content.structure.get("method", "unknown"),
            "duration_ms": round(process_duration, 2)
        }))
        
        if not processed_content.chunks:
            logger.warning(f"No chunks extracted from {filename}")
            return {
                "file_id": file_id,
                "chunk_count": 0,
                "status": "failed",
                "error": "No content extracted"
            }
        
        # Generate batch embeddings (all chunks at once)
        embed_start = time.time()
        llm_service = get_llm_service()
        
        # Collect all chunk texts
        chunk_texts = [chunk.content for chunk in processed_content.chunks]
        
        # Batch generate embeddings
        embeddings = await llm_service.embeddings_batch(
            texts=chunk_texts,
            model="text-embedding-3-small"
        )
        embed_duration = (time.time() - embed_start) * 1000
        
        logger.info(json.dumps({
            "event": "embeddings_generated_batch",
            "user_id": user_id,
            "filename": filename,
            "file_id": file_id,
            "chunk_count": len(chunk_texts),
            "duration_ms": round(embed_duration, 2),
            "avg_ms_per_chunk": round(embed_duration / len(chunk_texts), 2) if chunk_texts else 0
        }))
        
        # Prepare documents for indexing with full metadata and filters
        total_chunks = len(processed_content.chunks)
        created_at = datetime.now(timezone.utc).isoformat()
        
        documents = []
        for idx, (chunk, embedding) in enumerate(zip(processed_content.chunks, embeddings)):
            documents.append({
                "user_id": user_id,
                "content": chunk.content,
                "embedding": embedding,
                # Filters as top-level fields (for querying and filtering)
                "category": filters.category.value,
                "persona": [p.value for p in filters.persona],  # Array of keyword strings
                "issue_type": filters.issue_type,  # Array of strings
                "priority": filters.priority.value,
                "doc_weight": filters.doc_weight,
                # Metadata
                "metadata": {
                    "file_id": file_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                    "filename": filename,
                    "total_chunks": total_chunks,
                    "created_at": created_at
                }
            })
        
        # Batch index all documents
        index_start = time.time()
        results = await es_client.batch_index_documents(documents)
        index_duration = (time.time() - index_start) * 1000
        
        log_business_milestone(
            logger,
            "file_indexed",
            user_id=user_id,
            details={
                "filename": filename,
                "file_id": file_id,
                "successful": results['successful'],
                "failed": results['failed'],
                "total": results['total'],
                "duration_ms": round(index_duration, 2),
                "total_duration_ms": round((time.time() - start_time) * 1000, 2)
            },
        )
        
        return {
            "file_id": file_id,
            "chunk_count": results["successful"],
            "status": "success" if results["failed"] == 0 else "partial",
            "failed_count": results["failed"],
            "total_chunks": total_chunks
        }
    
    except Exception as e:
        log_error_with_context(
            logger,
            e,
            "file_ingestion_error",
            context={
                "user_id": user_id,
                "filename": filename,
                "mime_type": mime_type,
                "file_size": len(file_content),
            },
        )
        file_id = file_id if 'file_id' in locals() else f"{user_id}_{filename}_{int(time.time())}"
        return {
            "file_id": file_id,
            "chunk_count": 0,
            "status": "failed",
            "error": str(e)
        }


async def retrieve_chunks(
    user_id: str,
    query: str,
    top_k: int = 3,
    es_client: ElasticsearchClient = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from Elasticsearch using vector search
    
    Steps:
    1. Embed query with same embedding model
    2. Elasticsearch kNN search with user_id filter
    3. Return top_k chunks with metadata
    """
    # For backward compatibility, allow optional client parameter
    if es_client is None:
        from app.infra.elasticsearch import get_elasticsearch_client
        es_client = await get_elasticsearch_client()
    
    llm_service = get_llm_service()
    
    # Embed query
    query_embedding = await llm_service.embeddings(query, model="text-embedding-3-small")
    
    # Search Elasticsearch
    results = await es_client.search(
        user_id=user_id,
        query_embedding=query_embedding,
        top_k=top_k
    )
    
    logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
    return results
