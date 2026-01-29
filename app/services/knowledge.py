"""Knowledge service for RAG ingestion and retrieval"""
from app.infra.elasticsearch import get_elasticsearch_client
from app.infra.llm import get_llm_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


async def ingest_document(
    user_id: str,
    file_content: str,
    filename: str
) -> Dict[str, Any]:
    """
    Ingest a document into Elasticsearch for RAG
    
    Steps:
    1. Parse text content
    2. Chunk with RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
    3. Embed each chunk with OpenAI text-embedding-3-small
    4. Store in Elasticsearch with user_id metadata
    """
    es_client = await get_elasticsearch_client()
    llm_client = get_llm_client()
    
    # Chunk the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(file_content)
    logger.info(f"Split document into {len(chunks)} chunks")
    
    # Embed and index each chunk
    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        # Create embedding
        embedding = await llm_client.embeddings(chunk, model="text-embedding-3-small")
        
        # Index in Elasticsearch
        metadata = {
            "filename": filename,
            "chunk_index": idx,
            "created_at": datetime.utcnow().isoformat()
        }
        
        chunk_id = await es_client.index_document(
            user_id=user_id,
            content=chunk,
            embedding=embedding,
            metadata=metadata
        )
        chunk_ids.append(chunk_id)
    
    logger.info(f"Ingested {len(chunk_ids)} chunks for user {user_id}, file {filename}")
    
    return {
        "document_id": chunk_ids[0] if chunk_ids else None,
        "chunk_count": len(chunk_ids),
        "status": "success"
    }


async def retrieve_chunks(
    user_id: str,
    query: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from Elasticsearch using vector search
    
    Steps:
    1. Embed query with same embedding model
    2. Elasticsearch kNN search with user_id filter
    3. Return top_k chunks with metadata
    """
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


from datetime import datetime
