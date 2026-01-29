"""LLM client factory for OpenAI"""
from openai import AsyncOpenAI
from app.infra.config import settings
from typing import Optional, AsyncIterator, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI client wrapper"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.openai_api_key
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """Create a chat completion"""
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if stream:
            return await self.client.chat.completions.create(**params, stream=True)
        else:
            return await self.client.chat.completions.create(**params)
    
    async def embeddings(self, text: str, model: str = "text-embedding-3-small"):
        """Create embeddings"""
        response = await self.client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding


# Global LLM client instance
llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client instance"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client


def get_cheap_model() -> str:
    """Get cheap model name (GPT-3.5) for planning and summarization"""
    return "gpt-3.5-turbo"


def get_expensive_model() -> str:
    """Get expensive model name (GPT-4) for execution"""
    return "gpt-4"
