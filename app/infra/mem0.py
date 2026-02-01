"""Mem0 client for semantic memory using official SDK"""
from app.infra.config import settings
from mem0 import AsyncMemoryClient
from typing import Optional, List, Dict, Any, Literal
import logging

logger = logging.getLogger(__name__)


class Mem0Service:
    """Mem0 service wrapper using AsyncMemoryClient SDK"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mem0 async client with API key"""
        self.api_key = api_key or settings.mem0_api_key
        self.app_id = settings.mem0_app_id
        self.client = AsyncMemoryClient(api_key=self.api_key)
    
    async def ping(self) -> bool:
        """
        Check if API key is valid using a lightweight search operation
        
        Uses the documented search API with minimal query as a health check.
        This is more reliable than using internal _validate_api_key method.
        """
        try:
            # Use search with minimal query as health check
            # This validates API key and connectivity using documented API
            result = await self.client.search(
                query="health_check",
                filters={"user_id": "health_check_user"},
                limit=1
            )
            # If we get a response (even empty results), API is accessible
            return True
        except Exception as e:
            logger.error(f"Mem0 health check failed: {e}", exc_info=True)
            # Check if it's an authentication error
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["401", "403", "unauthorized", "authentication", "invalid", "api key"]):
                logger.error("Mem0 API key appears to be invalid or expired")
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
            result = await self.client.add(messages, user_id=user_id)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Mem0 add_interaction error: {e}")
            return {}
    
    async def add_memory(
        self,
        content: str,
        memory_type: Literal["episodic", "semantic", "procedural"],
        user_id: Optional[str] = None,
        additional_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add memory with proper classification.
        
        Args:
            content: Declarative memory statement (not "remember this")
            memory_type: episodic, semantic, or procedural
            user_id: Required for episodic (user-scoped), None for semantic/procedural (app-scoped)
            additional_metadata: Optional extra metadata
        
        Returns:
            Result dict from Mem0 API
        """
        try:
            # All memory types use "role": "user"
            messages = [{"role": "user", "content": content}]
            
            # Build metadata
            metadata = {"memory_type": memory_type}
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # User-scoped (episodic)
            if user_id:
                result = await self.client.add(
                    messages,
                    user_id=user_id,
                    app_id=self.app_id,
                    metadata=metadata
                )
            else:
                # App-scoped (semantic/procedural)
                result = await self.client.add(
                    messages,
                    app_id=self.app_id,
                    metadata=metadata
                )
            
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Mem0 add_memory error: {e}")
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
            result = await self.client.search(query, user_id=user_id, limit=limit)
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
    
    async def search_memory(
        self,
        query: str,
        memory_type: Optional[Literal["episodic", "semantic", "procedural"]] = None,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search memories with type and scope filtering.
        
        Args:
            query: Search query string
            memory_type: Optional memory type filter (episodic, semantic, procedural)
            user_id: Optional user ID for user-scoped search (episodic)
            limit: Maximum number of results (default: 5)
        
        Returns:
            List of memory dicts
        """
        try:
            # Build filters
            filter_conditions = []
            
            if user_id:
                # User-scoped search
                filter_conditions.append({"user_id": user_id})
            
            if memory_type:
                # Add memory type filter
                filter_conditions.append({"metadata": {"memory_type": memory_type}})
            
            # Construct filter
            filters = None
            if len(filter_conditions) == 1:
                filters = filter_conditions[0]
            elif len(filter_conditions) > 1:
                filters = {"AND": filter_conditions}
            
            result = await self.client.search(query, filters=filters, limit=limit)
            
            # Handle response format
            if isinstance(result, dict) and 'results' in result:
                return result['results']
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"Mem0 search_memory error: {e}")
            return []
    
    async def get_all(
        self,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all memories for a user (or app-scoped if no user_id).
        
        Uses Mem0's get_all() method which is the correct way to list all memories
        without requiring a search query.
        
        Args:
            user_id: Optional user identifier for user-scoped memories
            filters: Optional additional filters dict
            limit: Optional maximum number of results
        
        Returns:
            List of memory dicts
        """
        try:
            # Build filters dict
            mem0_filters = filters or {}
            if user_id:
                mem0_filters["user_id"] = user_id
            # Always include app_id in filters
            if self.app_id:
                mem0_filters["app_id"] = self.app_id
            
            # Call get_all with filters
            if limit:
                result = await self.client.get_all(filters=mem0_filters, limit=limit)
            else:
                result = await self.client.get_all(filters=mem0_filters)
            
            # Handle response format
            if isinstance(result, dict) and 'results' in result:
                return result['results']
            elif isinstance(result, list):
                return result
            else:
                return []
        except Exception as e:
            logger.error(f"Mem0 get_all error: {e}", exc_info=True)
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
