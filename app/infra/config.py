"""Configuration management using pydantic-settings"""
from pydantic_settings import BaseSettings
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
    
    # Elasticsearch Cloud
    elasticsearch_cloud_id: str
    elasticsearch_api_key: str
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
