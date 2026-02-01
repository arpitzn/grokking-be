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
            "request_timeout": 30.0,  # Increased from 3.0 for Elasticsearch Cloud
            "max_retries": 5,  # Increased from 2 for better resilience
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
        """Create the unified index with vector mapping if it doesn't exist"""
        if not await self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "content": {
                            "type": "text",
                            "analyzer": "english"
                        },
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
                        # Metadata
                        "metadata": {
                            "properties": {
                                "file_id": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "chunk_type": {"type": "keyword"},  # "text" or "image"
                                "filename": {"type": "keyword"},
                                "total_chunks": {"type": "integer"},
                                "created_at": {"type": "date"}
                            }
                        }
                    }
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "content_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
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
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
            "filter": {"term": {"user_id": user_id}}
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                knn=knn_query,
                source=["content", "metadata"],
                size=top_k
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
                "metadata.file_id", "metadata.filename", "metadata.created_at"
            ],
            "sort": [
                {"metadata.created_at": {"order": "desc"}}
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
                        "created_at": metadata.get("created_at", "")
                    }
                
                files_dict[file_key]["chunk_count"] += 1
            
            # Return as list, sorted by newest first (already sorted by ES)
            return list(files_dict.values())
        except Exception as e:
            logger.error(f"Elasticsearch list documents error: {e}")
            return []
    
    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the index (Area Manager access)
        
        Groups chunks by file, returns one entry per file
        """
        search_body = {
            "query": {
                "match_all": {}
            },
            "size": 10000,
            "_source": [
                "user_id", "category", "persona", "issue_type", "priority", "doc_weight",
                "metadata.file_id", "metadata.filename", "metadata.created_at"
            ],
            "sort": [
                {"metadata.created_at": {"order": "desc"}}
            ]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Group by file_id (includes user_id)
            files_dict = {}
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                file_id = metadata.get("file_id", "")
                
                if file_id and file_id not in files_dict:
                    files_dict[file_id] = {
                        "filename": metadata.get("filename", "unknown"),
                        "file_id": file_id,
                        "user_id": source.get("user_id", ""),
                        "category": source.get("category"),
                        "persona": source.get("persona", []),
                        "issue_type": source.get("issue_type", []),
                        "priority": source.get("priority"),
                        "doc_weight": source.get("doc_weight"),
                        "chunk_count": 0,
                        "created_at": metadata.get("created_at", "")
                    }
                
                if file_id in files_dict:
                    files_dict[file_id]["chunk_count"] += 1
            
            return list(files_dict.values())
        except Exception as e:
            logger.error(f"Elasticsearch list all documents error: {e}")
            return []
    
    async def list_documents_by_persona(
        self,
        persona: str
    ) -> List[Dict[str, Any]]:
        """
        List all documents where persona array contains the specified persona value
        
        Groups chunks by file, returns one entry per file
        Filters documents where persona field (array) contains the specified persona
        """
        # Normalize persona to lowercase for case-insensitive matching
        persona_lower = persona.lower()
        
        search_body = {
            "query": {
                "term": {"persona": persona_lower}
            },
            "size": 10000,
            "_source": [
                "user_id", "category", "persona", "issue_type", "priority", "doc_weight",
                "metadata.file_id", "metadata.filename", "metadata.created_at"
            ],
            "sort": [
                {"metadata.created_at": {"order": "desc"}}
            ]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Group by file_id (includes user_id)
            files_dict = {}
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                file_id = metadata.get("file_id", "")
                
                if file_id and file_id not in files_dict:
                    files_dict[file_id] = {
                        "filename": metadata.get("filename", "unknown"),
                        "file_id": file_id,
                        "user_id": source.get("user_id", ""),
                        "category": source.get("category"),
                        "persona": source.get("persona", []),
                        "issue_type": source.get("issue_type", []),
                        "priority": source.get("priority"),
                        "doc_weight": source.get("doc_weight"),
                        "chunk_count": 0,
                        "created_at": metadata.get("created_at", "")
                    }
                
                if file_id in files_dict:
                    files_dict[file_id]["chunk_count"] += 1
            
            return list(files_dict.values())
        except Exception as e:
            logger.error(f"Elasticsearch list documents by persona error: {e}")
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
    
    async def delete_files_by_persona(
        self,
        persona: str
    ) -> Dict[str, Any]:
        """
        Delete all documents where persona array contains the specified persona
        
        Uses delete_by_query with term query on persona field
        Returns count of deleted chunks
        """
        # Normalize persona to lowercase for case-insensitive matching
        persona_lower = persona.lower()
        
        # Elasticsearch 8.x API: query parameter instead of body
        response = await self.client.delete_by_query(
            index=self.index_name,
            query={
                "term": {"persona": persona_lower}
            },
            wait_for_completion=True,  # Wait for operation to complete before returning
            refresh=True  # Refresh index immediately so deletions are visible
        )
        
        return {
            "deleted": response.get("deleted", 0),
            "persona": persona_lower
        }
    
    async def get_document_by_file_id(
        self,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single document by file_id to check its persona field
        
        Returns the first chunk's metadata for the file, or None if not found
        """
        search_body = {
            "query": {
                "term": {"metadata.file_id": file_id}
            },
            "size": 1,
            "_source": [
                "persona", "metadata.file_id", "metadata.filename"
            ]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            if response["hits"]["hits"]:
                source = response["hits"]["hits"][0]["_source"]
                return {
                    "persona": source.get("persona", []),
                    "file_id": source.get("metadata", {}).get("file_id", ""),
                    "filename": source.get("metadata", {}).get("filename", "")
                }
            return None
        except Exception as e:
            logger.error(f"Elasticsearch get document by file_id error: {e}")
            return None
    
    async def search_policies(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search policy documents using vector search
        
        Args:
            query: Search query string (will be embedded)
            filters: Optional dict with keys: priority, category, issue_type (not used)
            top_k: Number of results to return
        
        Returns:
            List of policy documents with scores
        """
        # Embed query
        from app.infra.llm import get_llm_service
        llm_service = get_llm_service()
        query_embedding = await llm_service.embeddings(query, model="text-embedding-3-small")
        
        # Build kNN query without filters
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                knn=knn_query,
                source=["content", "category", "priority", "issue_type", "metadata"],
                size=top_k
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                results.append({
                    "file_id": metadata.get("file_id"),
                    "filename": metadata.get("filename", ""),
                    "content": source.get("content", ""),
                    "category": source.get("category"),
                    "priority": source.get("priority"),
                    "issue_type": source.get("issue_type", []),
                    "score": hit["_score"]
                })
            
            return results
        except Exception as e:
            logger.error(f"Elasticsearch policy search error: {e}")
            return []
    
    async def lookup_policy_by_file_id(
        self,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Lookup all chunks for a file_id and concatenate content
        
        Args:
            file_id: File identifier
        
        Returns:
            Policy document with concatenated content or None if not found
        """
        search_body = {
            "query": {
                "term": {"metadata.file_id": file_id}
            },
            "size": 10000,  # Get all chunks
            "sort": [{"metadata.chunk_index": {"order": "asc"}}],
            "_source": ["content", "category", "priority", "issue_type", "metadata"]
        }
        
        try:
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            if not response["hits"]["hits"]:
                return None
            
            chunks = []
            first_chunk = None
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                chunks.append(source.get("content", ""))
                
                if first_chunk is None:
                    first_chunk = {
                        "file_id": metadata.get("file_id"),
                        "filename": metadata.get("filename", ""),
                        "category": source.get("category"),
                        "priority": source.get("priority"),
                        "issue_type": source.get("issue_type", [])
                    }
            
            # Concatenate all chunks
            first_chunk["content"] = "\n\n".join(chunks)
            return first_chunk
        except Exception as e:
            logger.error(f"Elasticsearch policy lookup error: {e}")
            return None
    
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
