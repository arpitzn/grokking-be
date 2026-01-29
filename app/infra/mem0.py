"""Mem0 client for semantic memory using official SDK"""
from app.infra.config import settings
from mem0 import MemoryClient
from typing import Optional, List, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class Mem0Service:
    """Mem0 service wrapper using official MemoryClient SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mem0 client with API key"""
        self.api_key = api_key or settings.mem0_api_key
        self.client = MemoryClient(api_key=self.api_key)
    
    async def ping(self) -> bool:
        """
        Check if API key is valid using ping() method
        
        Validates API key and populates org/project IDs
        """
        try:
            # Run ping in thread pool since it might be synchronous
            result = await asyncio.to_thread(self.client.ping)
            return bool(result)
        except Exception as e:
            logger.error(f"Mem0 ping failed: {e}")
            return False
    
    async def add_interaction(
        self,
        user_id: str,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Add user-assistant interaction to Mem0
        
        Args:
            user_id: User identifier
            messages: List of message dicts with 'role' and 'content' keys
                     Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            Result dict from Mem0 API
        """
        try:
            # Run add in thread pool since SDK might be synchronous
            result = await asyncio.to_thread(self.client.add, messages, user_id=user_id)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Mem0 add_interaction error: {e}")
            return {}
    
    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user memories using semantic search
        
        Args:
            query: Search query string
            user_id: User identifier
            limit: Maximum number of results (default: 5)
        
        Returns:
            List of memory dicts with 'memory' field
        """
        try:
            # Run search in thread pool since SDK might be synchronous
            result = await asyncio.to_thread(self.client.search, query, user_id=user_id, limit=limit)
            # Handle response format: result['results'] array with 'memory' field
            if isinstance(result, dict) and 'results' in result:
                return result['results']
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"Mem0 search error: {e}")
            return []
    
    async def close(self):
        """Close Mem0 client (if needed)"""
        # SDK handles cleanup automatically, but keeping for compatibility
        pass


# Global Mem0 service instance
mem0_service: Optional[Mem0Service] = None


async def get_mem0_client() -> Mem0Service:
    """Get or create Mem0 service instance"""
    global mem0_service
    if mem0_service is None:
        mem0_service = Mem0Service()
    return mem0_service
