"""Simple cache manager for LLM instances"""
import threading
from typing import Dict, Any, Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SimpleCache:
    """Thread-safe simple dictionary-based cache for LLM instances"""
    
    def __init__(self):
        self._data: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        with self._lock:
            if key in self._data:
                self._hits += 1
                return self._data[key]
            else:
                self._misses += 1
                return None
    
    def put(self, key: str, value: T) -> None:
        """Put value in cache"""
        with self._lock:
            self._data[key] = value
    
    def remove(self, key: str) -> bool:
        """Remove key from cache"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "entry_count": len(self._data)
            }


# Global cache instance
_cache: Optional[SimpleCache] = None
_cache_lock = threading.Lock()


def get_llm_cache() -> SimpleCache:
    """Get global LLM cache instance (thread-safe singleton)"""
    global _cache
    
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SimpleCache()
                logger.info("Created global LLM cache instance")
    
    return _cache
