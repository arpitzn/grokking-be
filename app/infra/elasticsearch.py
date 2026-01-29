"""Elasticsearch cloud client for vector search"""
from elasticsearch import AsyncElasticsearch
from app.infra.config import settings
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for RAG vector search"""
    
    def __init__(self, cloud_id: Optional[str] = None, api_key: Optional[str] = None):
        cloud_id = cloud_id or settings.elasticsearch_cloud_id
        api_key = api_key or settings.elasticsearch_api_key
        
        self.client = AsyncElasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            request_timeout=3.0,
            max_retries=2,
            retry_on_timeout=True
        )
        self.index_name = "knowledge_base"
    
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
                        "metadata": {
                            "properties": {
                                "filename": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "created_at": {"type": "date"}
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
    
    async def health_check(self) -> bool:
        """Check Elasticsearch cluster health"""
        try:
            health = await self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return False
    
    async def close(self):
        """Close Elasticsearch connection"""
        await self.client.close()


# Global Elasticsearch client instance
elasticsearch_client: Optional[ElasticsearchClient] = None


async def get_elasticsearch_client() -> ElasticsearchClient:
    """Get or create Elasticsearch client instance"""
    global elasticsearch_client
    if elasticsearch_client is None:
        elasticsearch_client = ElasticsearchClient()
        await elasticsearch_client.create_index_if_not_exists()
    return elasticsearch_client
