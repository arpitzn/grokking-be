"""Mem0 client for semantic memory"""
from app.infra.config import settings
from typing import Optional, List, Dict, Any
import httpx
import logging

logger = logging.getLogger(__name__)


class Mem0Client:
    """Mem0 client for semantic memory operations"""
    
    def __init__(self, api_key: Optional[str] = None, org_id: Optional[str] = None):
        self.api_key = api_key or settings.mem0_api_key
        self.org_id = org_id or settings.mem0_org_id
        self.base_url = "https://api.mem0.ai/v1"
        self.client = httpx.AsyncClient(
            timeout=5.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def add_memory(
        self,
        user_id: str,
        memory: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add a memory for a user (async, fire-and-forget)"""
        try:
            response = await self.client.post(
                f"{self.base_url}/memories/",
                json={
                    "user_id": user_id,
                    "memory": memory,
                    "metadata": metadata or {}
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Mem0 add_memory error: {e}")
            return {}
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search user memories"""
        try:
            response = await self.client.post(
                f"{self.base_url}/memories/search",
                json={
                    "user_id": user_id,
                    "query": query,
                    "limit": limit
                }
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"Mem0 search_memories error: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Mem0 API health"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Global Mem0 client instance
mem0_client: Optional[Mem0Client] = None


async def get_mem0_client() -> Mem0Client:
    """Get or create Mem0 client instance"""
    global mem0_client
    if mem0_client is None:
        mem0_client = Mem0Client()
    return mem0_client
