"""LLM client factory for OpenAI with LangChain integration"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.infra.config import settings
from typing import Optional, AsyncIterator, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """LangChain ChatOpenAI wrapper for automatic callback integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.openai_api_key
        # Create LangChain ChatOpenAI client (callbacks will be automatically picked up)
        self.client = ChatOpenAI(
            api_key=api_key,
            temperature=0.7,
            model="gpt-3.5-turbo"  # Default, can be overridden per call
        )
    
    def _convert_messages(self, messages: List[Dict[str, str]]):
        """Convert dict messages to LangChain message objects"""
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:  # user
                lc_messages.append(HumanMessage(content=content))
        
        return lc_messages
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ):
        """Create a chat completion using LangChain (callbacks automatically integrated)"""
        # Convert messages to LangChain format
        lc_messages = self._convert_messages(messages)
        
        # Create a temporary client with specific model and temperature
        client = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=stream
        )
        
        # Callbacks will be automatically picked up from LangGraph config
        if stream:
            # Return async generator for streaming
            return client.astream(lc_messages)
        else:
            # Return response (will be AIMessage from LangChain)
            response = await client.ainvoke(lc_messages)
            
            # Convert back to OpenAI-like response format for compatibility
            class MockResponse:
                def __init__(self, content, usage=None):
                    self.content = content
                    self.usage = usage
                    
                class MockChoice:
                    def __init__(self, content):
                        self.message = type('obj', (object,), {'content': content})()
                
                @property
                def choices(self):
                    return [self.MockChoice(self.content)]
            
            # Extract usage if available (LangChain may not always provide this)
            usage = None
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                token_usage = response.response_metadata['token_usage']
                usage = type('obj', (object,), {
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0)
                })()
            
            return MockResponse(response.content, usage)
    
    async def embeddings(self, text: str, model: str = "text-embedding-3-small"):
        """Create embeddings (keeping OpenAI client for embeddings)"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    async def embeddings_batch(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """Create embeddings for multiple texts in batch"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model=model,
            input=texts
        )
        # Return list of embeddings in same order as input texts
        return [item.embedding for item in response.data]


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
