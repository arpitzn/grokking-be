"""
PromptFoo Provider for RAG Pipeline Testing

Tests retrieval quality with configurable top_k, chunk_size,
and embedding model parameters.

Usage in promptfooconfig.yaml:
  providers:
    - id: python:providers/rag_provider.py
      config:
        top_k: 3
        embedding_model: text-embedding-3-small
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def retrieve_chunks(
    user_id: str,
    query: str,
    top_k: int = 3,
    embedding_model: str = "text-embedding-3-small",
    persona: Optional[str] = None,
    issue_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from Elasticsearch.

    Args:
        user_id: User identifier for filtering
        query: Search query
        top_k: Number of chunks to retrieve
        embedding_model: Embedding model to use

    Returns:
        Dict with chunks and metadata
    """
    from app.infra.config import settings
    from app.infra.elasticsearch import get_elasticsearch_client
    from openai import AsyncOpenAI

    # Get embedding for query
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    embedding_response = await client.embeddings.create(
        model=embedding_model, input=query
    )
    query_embedding = embedding_response.data[0].embedding

    # Search Elasticsearch
    es_client = await get_elasticsearch_client()
    
    # Build search with persona/issue_type filtering if provided
    # Note: The actual filtering would need to be implemented in es_client.search
    # For now, we'll pass the parameters for potential future filtering
    results = await es_client.search(
        user_id=user_id, query_embedding=query_embedding, top_k=top_k
    )
    
    # Post-filter by persona/issue_type if metadata available
    if persona or issue_type:
        filtered_results = []
        for chunk in results:
            metadata = chunk.get("metadata", {})
            chunk_persona = metadata.get("persona")
            chunk_issue_type = metadata.get("issue_type")
            
            # Include if persona matches or no persona filter
            persona_match = not persona or chunk_persona == persona or not chunk_persona
            # Include if issue_type matches or no issue_type filter
            issue_match = not issue_type or chunk_issue_type == issue_type or not chunk_issue_type
            
            if persona_match and issue_match:
                filtered_results.append(chunk)
        
        # If filtering removed all results, return original results
        if filtered_results:
            results = filtered_results

    return {
        "query": query,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "chunks": results,
        "chunk_count": len(results),
    }


async def ingest_test_document(
    user_id: str,
    content: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """
    Ingest a test document for RAG testing.

    Args:
        user_id: User identifier
        content: Document content
        filename: Document filename
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Dict with ingestion results
    """
    from app.infra.elasticsearch import get_elasticsearch_client
    from app.services.knowledge import ingest_document

    es_client = await get_elasticsearch_client()
    result = await ingest_document(
        user_id=user_id, file_content=content, filename=filename, es_client=es_client
    )

    return result


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PromptFoo provider entry point.

    Args:
        prompt: The query for retrieval
        options: Provider options including top_k, embedding_model
        context: Test context including variables

    Returns:
        Dict with output containing retrieval results
    """
    # Extract configuration from options
    config = options.get("config", {})
    top_k = config.get("top_k", 3)
    embedding_model = config.get("embedding_model", "text-embedding-3-small")
    test_type = config.get("test_type", "retrieve")  # "retrieve" or "ingest"

    # Extract variables from context
    vars = context.get("vars", {})
    query = vars.get("query", prompt)
    user_id = vars.get("user_id", "test_user")
    persona = vars.get("persona")
    issue_type = vars.get("issue_type")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if test_type == "retrieve":
            result = loop.run_until_complete(
                retrieve_chunks(
                    user_id=user_id,
                    query=query,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    persona=persona,
                    issue_type=issue_type,
                )
            )

            # Format output for assertions
            output = {
                "query": result["query"],
                "chunk_count": result["chunk_count"],
                "chunks": [
                    {
                        "content": chunk.get("content", ""),
                        "score": chunk.get("score", 0),
                        "metadata": chunk.get("metadata", {}),
                    }
                    for chunk in result["chunks"]
                ],
                "persona_filter": persona,
                "issue_type_filter": issue_type,
                "top_k": result["top_k"],
                "embedding_model": result["embedding_model"],
            }
        elif test_type == "ingest":
            content = vars.get("content", "")
            filename = vars.get("filename", "test_doc.txt")
            chunk_size = config.get("chunk_size", 500)
            chunk_overlap = config.get("chunk_overlap", 50)

            result = loop.run_until_complete(
                ingest_test_document(
                    user_id=user_id,
                    content=content,
                    filename=filename,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

            output = {
                "status": result.get("status", "success"),
                "chunk_count": result.get("chunk_count", 0),
            }
        else:
            output = {"error": f"Unknown test_type: {test_type}"}

        loop.close()

        # Return result in PromptFoo format
        return {
            "output": json.dumps(output),
            "metadata": {"test_type": test_type, "top_k": top_k},
        }
    except Exception as e:
        return {
            "output": json.dumps({"error": str(e), "chunk_count": 0, "chunks": []}),
            "error": str(e),
        }


# For direct testing
if __name__ == "__main__":
    # Test retrieval
    test_result = call_api(
        prompt="What are the Q4 objectives?",
        options={"config": {"top_k": 3, "embedding_model": "text-embedding-3-small"}},
        context={
            "vars": {"query": "What are the Q4 objectives?", "user_id": "test_user"}
        },
    )
    print("RAG Test:", json.dumps(json.loads(test_result["output"]), indent=2))
