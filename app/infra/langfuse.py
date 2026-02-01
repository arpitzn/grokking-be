"""Langfuse client for manual trace creation (if needed)"""
from langfuse import Langfuse
from app.infra.config import settings
from typing import Optional

# Global Langfuse client instance (for manual trace creation if needed)
_langfuse_client: Optional[Langfuse] = None


def get_langfuse_client() -> Langfuse:
    """Get or create Langfuse client instance for manual trace creation"""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host
        )
    return _langfuse_client
