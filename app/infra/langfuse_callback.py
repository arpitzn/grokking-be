"""Langfuse CallbackHandler for automatic tracing"""
import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from app.infra.config import settings

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host
)

# Set tracing environment from settings (defaults to development if not set)
environment = getattr(settings, 'environment', 'development')
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = environment

# Create global CallbackHandler instance (singleton pattern)
langfuse_handler = CallbackHandler()
