"""Elasticsearch client for vector search"""
from elasticsearch import AsyncElasticsearch
from app.infra.config import settings
from typing import Optional, List, Dict, Any, Annotated
from fastapi import Depends
import logging

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for RAG vector search"""
    
    def __init__(self):
        connection_params = {
            "hosts": [settings.elasticsearch_node],
            "request_timeout": 3.0,
            "max_retries": 2,
            "retry_on_timeout": True,
        }
        
        # Add HTTP Basic Auth if credentials provided
        if settings.elasticsearch_username and settings.elasticsearch_password:
            connection_params["basic_auth"] = (
                settings.elasticsearch_username,
                settings.elasticsearch_password
            )
        
        self.client = AsyncElasticsearch(**connection_params)
        self.index_name = settings.elasticsearch_index_name
    
    async def create_index_if_not_exists(self):
        """Create the knowledge_base index with vector mapping if it doesn't exist"""
        if not await self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,  # OpenAI text-embedding-3-small dimensions
                            "index": True,
                            "similarity": "cosine"
                        },
                        # Filter fields (top-level for querying and filtering)
                        "category": {"type": "keyword"},
                        "persona": {"type": "keyword"},  # Array of keywords
                        "issue_type": {"type": "keyword"},  # Array of keywords
                        "priority": {"type": "keyword"},
                        "doc_weight": {"type": "float"},
                        # Existing metadata
                        "metadata": {
                            "properties": {
                                "file_id": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "chunk_type": {"type": "keyword"},  # "text" or "image"
                                "filename": {"type": "keyword"},
                                "total_chunks": {"type": "integer"},
                                "created_date": {"type": "date"},
                                "created_at": {"type": "date"}  # Keep for backward compat
                            }
                        }
                    }
                }
            }
            await self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {self.index_name}")
    
    async def index_document(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Index a document chunk with embedding"""
        doc = {
            "user_id": user_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }
        result = await self.client.index(index=self.index_name, document=doc)
        return result["_id"]
    
    async def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using kNN with user_id filter"""
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 10
            },
            "filter": {
                "term": {"user_id": user_id}
            },
            "_source": ["content", "metadata"]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"]
                })
            
            return results
        except Exception as e:
            logger.error(f"Elasticsearch search error: {e}")
            return []
    
    async def list_documents_by_user(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        List all documents for a user, aggregated by file
        
        Groups chunks by user_id + filename, returns one entry per file
        with filters from first chunk and total chunk count
        """
        search_body = {
            "query": {
                "term": {"user_id": user_id}
            },
            "size": 10000,  # Large size to get all documents
            "_source": [
                "category", "persona", "issue_type", "priority", "doc_weight",
                "metadata.file_id", "metadata.filename", "metadata.created_date"
            ],
            "sort": [
                {"metadata.created_date": {"order": "desc"}}
            ]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Group by user_id + filename (user_filename)
            files_dict = {}
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                filename = metadata.get("filename", "unknown")
                file_key = f"{user_id}_{filename}"
                
                if file_key not in files_dict:
                    files_dict[file_key] = {
                        "filename": filename,
                        "file_id": metadata.get("file_id", ""),
                        "category": source.get("category"),
                        "persona": source.get("persona", []),
                        "issue_type": source.get("issue_type", []),
                        "priority": source.get("priority"),
                        "doc_weight": source.get("doc_weight"),
                        "chunk_count": 0,
                        "created_at": metadata.get("created_date", "")
                    }
                
                files_dict[file_key]["chunk_count"] += 1
            
            # Return as list, sorted by newest first (already sorted by ES)
            return list(files_dict.values())
        except Exception as e:
            logger.error(f"Elasticsearch list documents error: {e}")
            return []
    
    async def batch_index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Batch index multiple document chunks
        
        Args:
            documents: List of dicts with keys: user_id, content, embedding, metadata, and filter fields (category, persona, issue_type, priority, doc_weight)
        
        Returns:
            Dict with total, successful, failed counts
        """
        actions = []
        for doc in documents:
            actions.append({"index": {"_index": self.index_name}})
            # Include ALL fields from the document (including filter fields)
            actions.append(doc)
        
        try:
            response = await self.client.bulk(
                operations=actions,
                refresh=True  # Refresh index immediately so documents are searchable
            )
            
            results = {
                "total": len(documents),
                "successful": 0,
                "failed": 0,
                "errors": []
            }
            
            for item in response["items"]:
                if "index" in item:
                    if item["index"]["status"] in [200, 201]:
                        results["successful"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(item["index"].get("error", "Unknown error"))
            
            logger.info(f"Batch indexed {results['successful']}/{results['total']} documents")
            return results
        except Exception as e:
            logger.error(f"Batch indexing error: {e}")
            return {
                "total": len(documents),
                "successful": 0,
                "failed": len(documents),
                "errors": [str(e)]
            }
    
    async def delete_file_by_id(
        self,
        file_id: str
    ) -> Dict[str, Any]:
        """
        Delete all chunks for a file by file_id
        
        Uses delete_by_query to remove all documents matching metadata.file_id
        """
        # Elasticsearch 8.x API: query parameter instead of body
        response = await self.client.delete_by_query(
            index=self.index_name,
            query={
                "term": {"metadata.file_id": file_id}
            },
            wait_for_completion=True,  # Wait for operation to complete before returning
            refresh=True  # Refresh index immediately so deletions are visible
        )
        
        return {
            "deleted": response.get("deleted", 0),
            "file_id": file_id
        }
    
    async def delete_all_files(
        self
    ) -> Dict[str, Any]:
        """
        Delete all documents from the index (global delete)
        
        Uses delete_by_query with match_all query
        """
        # Elasticsearch 8.x API: query parameter instead of body
        response = await self.client.delete_by_query(
            index=self.index_name,
            query={
                "match_all": {}
            },
            wait_for_completion=True,  # Wait for operation to complete before returning
            refresh=True  # Refresh index immediately so deletions are visible
        )
        
        return {
            "deleted": response.get("deleted", 0)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Elasticsearch cluster health with detailed info"""
        try:
            # Get cluster info
            info = await self.client.info()
            health = await self.client.cluster.health()
            
            return {
                "status": "healthy" if health["status"] in ["green", "yellow"] else "unhealthy",
                "cluster_name": info.get("cluster_name"),
                "version": info.get("version", {}).get("number"),
                "cluster_health": health.get("status"),
                "number_of_nodes": health.get("number_of_nodes"),
                "authentication_configured": bool(settings.elasticsearch_username)
            }
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "authentication_configured": bool(settings.elasticsearch_username)
            }
    
    async def close(self):
        """Close Elasticsearch connection"""
        await self.client.close()


# Global Elasticsearch client instance (for DI)
_elasticsearch_client: Optional[ElasticsearchClient] = None


async def get_elasticsearch_client() -> ElasticsearchClient:
    """Get or create Elasticsearch client instance (for DI)"""
    global _elasticsearch_client
    if _elasticsearch_client is None:
        _elasticsearch_client = ElasticsearchClient()
        await _elasticsearch_client.create_index_if_not_exists()
    return _elasticsearch_client


# FastAPI dependency
ElasticsearchDep = Annotated[ElasticsearchClient, Depends(get_elasticsearch_client)]
