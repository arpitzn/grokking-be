"""Configuration management using pydantic-settings"""
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB
    mongodb_uri: str
    mongodb_db_name: str = "hackathon_agent"
    
    # OpenAI
    openai_api_key: str
    
    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"
    
    # Environment for Langfuse tracing
    environment: str = "development"  # development, staging, production
    
    # Mem0
    mem0_api_key: str
    mem0_org_id: str
    
    # Elasticsearch
    elasticsearch_node: str
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    elasticsearch_index_name: str = "knowledge_base"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
