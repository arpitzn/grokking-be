"""Langfuse integration for observability"""
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from app.infra.config import settings
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Langfuse client for tracing and observability"""
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None
    ):
        public_key = public_key or settings.langfuse_public_key
        secret_key = secret_key or settings.langfuse_secret_key
        host = host or settings.langfuse_host
        
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
    
    def create_trace(
        self,
        name: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a new trace"""
        trace_metadata = metadata or {}
        if conversation_id:
            trace_metadata["conversation_id"] = conversation_id
        if user_id:
            trace_metadata["user_id"] = user_id
        
        trace = self.client.trace(
            name=name,
            metadata=trace_metadata
        )
        
        langfuse_context.update_current_trace_id(trace.id)
        return trace
    
    def create_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a span within a trace"""
        span = self.client.span(
            name=name,
            trace_id=trace_id,
            metadata=metadata
        )
        return span
    
    def span(self, name: str, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for span"""
        return SpanContext(self.client, name, trace_id, metadata)


class SpanContext:
    """Context manager for Langfuse spans"""
    def __init__(self, client, name: str, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.client = client
        self.name = name
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.span = None
    
    def __enter__(self):
        self.span = self.client.span(
            name=self.name,
            trace_id=self.trace_id,
            metadata=self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end()
        return False
    
    def update(self, metadata: Optional[Dict[str, Any]] = None):
        """Update span metadata"""
        if self.span and metadata:
            self.span.update(metadata=metadata)
    
    def span(self, name: str, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for span"""
        return SpanContext(self.client, name, trace_id, metadata)
    
    def flush(self):
        """Flush pending events"""
        self.client.flush()


class SpanContext:
    """Context manager for Langfuse spans"""
    def __init__(self, client, name: str, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.client = client
        self.name = name
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.span = None
    
    def __enter__(self):
        self.span = self.client.span(
            name=self.name,
            trace_id=self.trace_id,
            metadata=self.metadata
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            self.span.end()
        return False
    
    def update(self, metadata: Optional[Dict[str, Any]] = None):
        """Update span metadata"""
        if self.span and metadata:
            self.span.update(metadata=metadata)


# Global Langfuse manager instance
langfuse_manager: Optional[LangfuseManager] = None


def get_langfuse_manager() -> LangfuseManager:
    """Get or create Langfuse manager instance"""
    global langfuse_manager
    if langfuse_manager is None:
        langfuse_manager = LangfuseManager()
    return langfuse_manager
