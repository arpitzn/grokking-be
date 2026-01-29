"""External search stub tool - returns hardcoded responses for demo"""
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ExternalSearchStub:
    """
    Stub implementation of external search tool.
    Returns hardcoded responses for demo queries.
    
    PRODUCTION: Replace with real API (Tavily, Google News, Bing Search)
    """
    
    def __init__(self):
        # Hardcoded responses for common demo queries
        self.response_templates = {
            "president of india 2020": {
                "title": "President of India (2020)",
                "content": "Ram Nath Kovind served as the 14th President of India from 2017 to 2022.",
                "url": "https://example.com/president-india",
                "score": 1.0
            },
            "capital of france": {
                "title": "Capital of France",
                "content": "Paris is the capital and largest city of France.",
                "url": "https://example.com/paris",
                "score": 1.0
            },
            "who is": {
                "title": "General Information",
                "content": "This is a general knowledge query. In production, this would query real-time search APIs.",
                "url": "https://example.com/search",
                "score": 0.8
            },
        }
    
    async def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Stub search - returns hardcoded responses
        
        Args:
            query: Search query string
            max_results: Number of results to return
        
        Returns:
            {
                "results": [...],
                "query": "original query",
                "is_stub": true  # Flag indicating stub response
            }
        """
        # Normalize query for matching
        query_lower = query.lower()
        
        # Find matching template
        result = None
        for key, template in self.response_templates.items():
            if key in query_lower:
                result = template
                break
        
        # Default response if no match
        if not result:
            result = {
                "title": "Search Result",
                "content": f"Information about: {query}",
                "url": "https://example.com/search",
                "score": 0.8
            }
        
        logger.info(f"External search stub returning result for query: {query}")
        
        return {
            "results": [result],
            "query": query,
            "is_stub": True  # Important: flag for monitoring/testing
        }


# Global stub instance
external_search_stub = ExternalSearchStub()
