"""Knowledge service for RAG ingestion and retrieval"""
from app.infra.elasticsearch import ElasticsearchDep, ElasticsearchClient
from app.infra.llm import get_llm_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from datetime import datetime
import logging

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
    llm_client = get_llm_client()
    
    # Chunk the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(file_content)
    logger.info(f"Split document into {len(chunks)} chunks")
    
    # Collect all documents for batch indexing
    documents = []
    for idx, chunk in enumerate(chunks):
        # Create embedding
        embedding = await llm_client.embeddings(chunk, model="text-embedding-3-small")
        
        documents.append({
            "user_id": user_id,
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "created_at": datetime.utcnow().isoformat()
            }
        })
    
    # Batch index all documents
    results = await es_client.batch_index_documents(documents)
    
    logger.info(f"Ingested {results['successful']}/{results['total']} chunks for user {user_id}, file {filename}")
    
    return {
        "document_id": None,  # Batch indexing doesn't return individual IDs
        "chunk_count": results["successful"],
        "status": "success" if results["failed"] == 0 else "partial",
        "failed_count": results["failed"]
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
    
    llm_client = get_llm_client()
    
    # Embed query
    query_embedding = await llm_client.embeddings(query, model="text-embedding-3-small")
    
    # Search Elasticsearch
    results = await es_client.search(
        user_id=user_id,
        query_embedding=query_embedding,
        top_k=top_k
    )
    
    logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
    return results
