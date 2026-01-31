"""LLM service with caching, tool binding, and structured output support"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from app.infra.config import settings
from app.infra.cache_manager import get_llm_cache
from typing import Optional, Dict, Any, List, Type, Union
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM instances with caching"""
    
    def __init__(self):
        self._cache = get_llm_cache()
    
    def _get_openai_instance(self, model_name: str, **kwargs) -> ChatOpenAI:
        """
        Get cached ChatOpenAI instance or create new one.
        
        Args:
            model_name: OpenAI model name (e.g., gpt-4o, gpt-3.5-turbo)
            **kwargs: Additional configuration including:
                - disable_streaming: bool
                - max_completion_tokens: int
                - request_timeout: float (in seconds)
                - temperature: float
                
        Returns:
            ChatOpenAI instance (cached or newly created)
        """
        disable_streaming = kwargs.get("disable_streaming", False)
        request_timeout = kwargs.get("request_timeout", None)
        max_completion_tokens = kwargs.get("max_completion_tokens", None)
        temperature = kwargs.get("temperature", None)
        
        # Create cache key from parameters including streaming control
        cache_key = f"openai_{model_name}_streaming_{not disable_streaming}_{hash(str(sorted(kwargs.items())))}"
        
        # Try to get from cache
        cached_instance = self._cache.get(cache_key)
        if cached_instance is not None:
            logger.info(f"Cache HIT - Returning cached OpenAI instance | model={model_name} | cache_key={cache_key[:50]}")
            return cached_instance
        
        # Cache miss - create new instance
        logger.info(f"Cache MISS - Creating new OpenAI instance | model={model_name} | streaming={not disable_streaming} | timeout={request_timeout}s | max_tokens={max_completion_tokens} | temp={temperature}")
        
        try:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is not configured")
            
            # Extract max_completion_tokens and request_timeout from kwargs if present
            max_completion_tokens = kwargs.pop("max_completion_tokens", None)
            request_timeout = kwargs.pop("request_timeout", None)
            disable_streaming = kwargs.pop("disable_streaming", False)
            
            llm_instance = ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                max_completion_tokens=max_completion_tokens,
                request_timeout=request_timeout,
                streaming=not disable_streaming,
                **kwargs
            )
            
            # Cache the instance
            self._cache.put(cache_key, llm_instance)
            
            logger.info(f"✓ OpenAI instance created & cached | model={model_name} | cache_key={cache_key[:50]}")
            
            return llm_instance
            
        except Exception as e:
            logger.error(f"✗ Failed to create OpenAI instance | model={model_name} | error={type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Failed to initialize ChatOpenAI with model {model_name}: {str(e)}") from e
    
    def get_llm_instance(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get non-streaming LLM instance based on model configuration.
        
        Args:
            model_name: Name of the model to use (e.g., gpt-4, gpt-3.5-turbo)
            **kwargs: Additional parameters for the model.
                     Use disable_streaming=True to explicitly disable streaming.
            
        Returns:
            BaseChatModel: Configured LLM instance
        """
        kwargs["disable_streaming"] = True
        return self._get_openai_instance(model_name, **kwargs)
    
    def get_streaming_llm_instance(self, model_name: str, **kwargs) -> BaseChatModel:
        """
        Get streaming-enabled LLM instance based on model configuration.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional parameters for the model
            
        Returns:
            BaseChatModel: Configured streaming LLM instance
        """
        kwargs["disable_streaming"] = False
        return self._get_openai_instance(model_name, **kwargs)
    
    def get_llm_instance_with_tools(
        self,
        model_name: str,
        tools: List[Union[BaseTool, Dict[str, Any]]],
        **kwargs
    ) -> BaseChatModel:
        """
        Get non-streaming LLM instance with tool binding for function calling.
        Uses caching to avoid rebinding the same tools repeatedly.
        
        Args:
            model_name: Model name (e.g., gpt-4o, gpt-3.5-turbo)
            tools: List of tools (BaseTool instances or dict definitions)
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseChatModel instance with tools bound for function calling
        """
        try:
            # Generate cache key that includes tools
            disable_streaming = True  # Always disable streaming for this method
            tools_hash = self._get_tools_hash(tools)
            cache_key = f"openai_{model_name}_tools_{tools_hash}_streaming_{not disable_streaming}_{hash(str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_instance = self._cache.get(cache_key)
            if cached_instance is not None:
                logger.info(f"Retrieved LLM instance with tools from cache: model={model_name}, tools_count={len(tools)}")
                return cached_instance
            
            # Not in cache, create new instance
            non_streaming_kwargs = kwargs.copy()
            non_streaming_kwargs["disable_streaming"] = True
            
            # Get base LLM instance
            llm = self.get_llm_instance(model_name, **non_streaming_kwargs)
            
            # Convert tools to proper format if needed
            formatted_tools = self._format_tools_for_binding(tools)
            
            # Bind tools to the LLM for function calling
            llm_with_tools = llm.bind_tools(formatted_tools)
            
            # Cache the instance
            self._cache.put(cache_key, llm_with_tools)
            
            logger.info(f"Successfully created and cached non-streaming LLM instance with {len(formatted_tools)} tools bound: model={model_name}")
            return llm_with_tools
            
        except Exception as e:
            logger.error(f"Error creating LLM instance with tools: {str(e)}")
            raise ValueError(f"Failed to create LLM instance with tools: {str(e)}")
    
    def get_streaming_llm_instance_with_tools(
        self,
        model_name: str,
        tools: List[Union[BaseTool, Dict[str, Any]]],
        **kwargs
    ) -> BaseChatModel:
        """
        Get streaming LLM instance with tool binding for function calling.
        Uses caching to avoid rebinding the same tools repeatedly.
        
        Args:
            model_name: Model name (e.g., gpt-4o, gpt-3.5-turbo)
            tools: List of tools (BaseTool instances or dict definitions)
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseChatModel instance with tools bound for streaming function calling
        """
        try:
            # Generate cache key that includes tools
            disable_streaming = kwargs.get("disable_streaming", False)  # Default to False for streaming
            tools_hash = self._get_tools_hash(tools)
            cache_key = f"openai_{model_name}_tools_{tools_hash}_streaming_{not disable_streaming}_{hash(str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_instance = self._cache.get(cache_key)
            if cached_instance is not None:
                logger.info(f"Retrieved streaming LLM instance with tools from cache: model={model_name}, tools_count={len(tools)}")
                return cached_instance
            
            # Not in cache, create new instance
            # Get base streaming LLM instance
            llm = self.get_streaming_llm_instance(model_name, **kwargs)
            
            # Convert tools to proper format if needed
            formatted_tools = self._format_tools_for_binding(tools)
            
            # Bind tools to the LLM for function calling
            llm_with_tools = llm.bind_tools(formatted_tools)
            
            # Cache the instance
            self._cache.put(cache_key, llm_with_tools)
            
            logger.info(f"Successfully created and cached streaming LLM instance with {len(formatted_tools)} tools bound: model={model_name}")
            return llm_with_tools
            
        except Exception as e:
            logger.error(f"Error creating streaming LLM instance with tools: {str(e)}")
            raise ValueError(f"Failed to create streaming LLM instance with tools: {str(e)}")
    
    def get_structured_output_llm_instance(
        self,
        model_name: str,
        schema: Union[Type, Dict[str, Any]],
        **kwargs
    ) -> BaseChatModel:
        """
        Get non-streaming LLM instance with structured output binding.
        Uses caching to avoid rebinding the same schema repeatedly.
        
        Args:
            model_name: Model name (e.g., gpt-4o, gpt-3.5-turbo)
            schema: Pydantic model class or schema dict for structured output
            **kwargs: Additional configuration parameters (e.g., temperature)
            
        Returns:
            BaseChatModel instance with structured output bound
        """
        try:
            # Generate cache key that includes structured output schema
            disable_streaming = True  # Always disable streaming for this method
            schema_hash = self._get_structured_output_hash(schema)
            schema_name = getattr(schema, '__name__', str(schema)[:50])
            cache_key = f"openai_{model_name}_structured_{schema_hash}_streaming_{not disable_streaming}_{hash(str(sorted(kwargs.items())))}"
            
            request_timeout = kwargs.get("request_timeout", None)
            temperature = kwargs.get("temperature", None)
            
            # Try to get from cache
            cached_instance = self._cache.get(cache_key)
            if cached_instance is not None:
                logger.info(f"Cache HIT - Structured output | model={model_name} | schema={schema_name}")
                return cached_instance
            
            # Cache miss - create new instance
            logger.info(f"Cache MISS - Creating structured output LLM | model={model_name} | schema={schema_name} | timeout={request_timeout}s | temp={temperature}")
            
            # Not in cache, create new instance
            non_streaming_kwargs = kwargs.copy()
            non_streaming_kwargs["disable_streaming"] = True
            
            # Get base LLM instance
            llm = self.get_llm_instance(model_name, **non_streaming_kwargs)
            
            # Bind structured output to the LLM
            llm_with_structured_output = llm.with_structured_output(schema)
            
            # Cache the instance
            self._cache.put(cache_key, llm_with_structured_output)
            
            logger.info(f"✓ Structured output LLM created & cached | model={model_name} | schema={schema_name}")
            return llm_with_structured_output
            
        except Exception as e:
            logger.error(f"✗ Failed to create structured output LLM | model={model_name} | schema={getattr(schema, '__name__', 'unknown')} | error={type(e).__name__}: {str(e)}")
            raise ValueError(f"Failed to create LLM instance with structured output: {str(e)}")
    
    def get_streaming_structured_output_llm_instance(
        self,
        model_name: str,
        schema: Union[Type, Dict[str, Any]],
        **kwargs
    ) -> BaseChatModel:
        """
        Get streaming LLM instance with structured output binding.
        Uses caching to avoid rebinding the same schema repeatedly.
        Note: Structured output typically doesn't stream, but this method is provided for consistency.
        
        Args:
            model_name: Model name (e.g., gpt-4o, gpt-3.5-turbo)
            schema: Pydantic model class or schema dict for structured output
            **kwargs: Additional configuration parameters (e.g., temperature)
            
        Returns:
            BaseChatModel instance with structured output bound for streaming
        """
        try:
            # Generate cache key that includes structured output schema
            disable_streaming = kwargs.get("disable_streaming", False)  # Default to False for streaming
            schema_hash = self._get_structured_output_hash(schema)
            cache_key = f"openai_{model_name}_structured_{schema_hash}_streaming_{not disable_streaming}_{hash(str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_instance = self._cache.get(cache_key)
            if cached_instance is not None:
                logger.info(f"Retrieved streaming LLM instance with structured output from cache: model={model_name}, schema={getattr(schema, '__name__', str(schema)[:50])}")
                return cached_instance
            
            # Not in cache, create new instance
            # Get base streaming LLM instance
            llm = self.get_streaming_llm_instance(model_name, **kwargs)
            
            # Bind structured output to the LLM
            llm_with_structured_output = llm.with_structured_output(schema)
            
            # Cache the instance
            self._cache.put(cache_key, llm_with_structured_output)
            
            logger.info(f"Successfully created and cached streaming LLM instance with structured output: model={model_name}, schema={getattr(schema, '__name__', str(schema)[:50])}")
            return llm_with_structured_output
            
        except Exception as e:
            logger.error(f"Error creating streaming LLM instance with structured output: {str(e)}")
            raise ValueError(f"Failed to create streaming LLM instance with structured output: {str(e)}")
    
    def _get_structured_output_hash(self, schema: Union[Type, Dict[str, Any]]) -> str:
        """
        Generate a consistent hash from structured output schema for caching.
        
        Args:
            schema: Pydantic model class or schema dict
            
        Returns:
            String hash representing the schema configuration
        """
        try:
            if hasattr(schema, 'model_json_schema'):
                # Pydantic model - use its JSON schema
                schema_dict = schema.model_json_schema()
                schema_str = str(sorted(schema_dict.items()))
            elif hasattr(schema, '__name__'):
                # Class with name - use class name and module
                schema_str = f"{schema.__module__}.{schema.__name__}"
            elif isinstance(schema, dict):
                # Already a dict - use sorted items
                schema_str = str(sorted(schema.items()))
            else:
                # Fallback - use string representation
                schema_str = str(schema)
            
            return str(hash(schema_str))
        except Exception as e:
            logger.warning(f"Error generating structured output hash, using fallback: {str(e)}")
            return str(hash(str(schema)))
    
    def _get_tools_hash(self, tools: List[Union[BaseTool, Dict[str, Any]]]) -> str:
        """
        Generate a consistent hash from tools for caching.
        
        Args:
            tools: List of tools to hash
            
        Returns:
            String hash representing the tools configuration
        """
        formatted_tools = self._format_tools_for_binding(tools)
        # Sort the tools by name to ensure consistent ordering
        sorted_tools = sorted(formatted_tools, key=lambda x: x.get('function', {}).get('name', ''))
        # Create a deterministic string representation and hash it
        tools_str = str(sorted_tools)
        return str(hash(tools_str))
    
    def _format_tools_for_binding(self, tools: List[Union[BaseTool, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert tools to the format expected by LangChain's bind_tools method.
        
        Args:
            tools: List of tools (BaseTool instances or dict definitions)
            
        Returns:
            List of tool definitions in OpenAI function format
        """
        formatted_tools = []
        
        for tool in tools:
            try:
                if isinstance(tool, BaseTool):
                    # Convert BaseTool to dict format
                    # Get schema - use Pydantic V2 model_json_schema() if available, fallback to schema() for compatibility
                    parameters = {}
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        if hasattr(tool.args_schema, 'model_json_schema'):
                            # Pydantic V2
                            parameters = tool.args_schema.model_json_schema()
                        elif hasattr(tool.args_schema, 'schema'):
                            # Pydantic V1 fallback
                            parameters = tool.args_schema.schema()
                    
                    tool_dict = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": parameters
                        }
                    }
                    formatted_tools.append(tool_dict)
                    
                elif isinstance(tool, dict):
                    # Already in dict format, validate and use
                    if "type" in tool and "function" in tool:
                        formatted_tools.append(tool)
                    else:
                        logger.warning(f"Skipping invalid tool dict format: {tool}")
                        
                else:
                    logger.warning(f"Skipping unsupported tool type: {type(tool)}")
                    
            except Exception as e:
                logger.error(f"Error formatting tool {tool}: {str(e)}")
                continue
        
        if not formatted_tools:
            logger.warning("No valid tools found for binding")
            
        return formatted_tools
    
    def convert_messages(self, messages: List[Dict[str, str]]):
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
    
    async def embeddings(self, text: str, model: str = "text-embedding-3-small"):
        """Create embeddings (keeping OpenAI client for embeddings)"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    def clear_cache(self):
        """Clear the LLM instance cache."""
        self._cache.clear()
        logger.info("LLM cache cleared")
        
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


# Global LLM service instance
llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service


def get_cheap_model() -> str:
    """Get cheap model name (GPT-4.1-Mini) for planning and summarization"""
    return "gpt-4.1-mini"


def get_expensive_model() -> str:
    """Get expensive model name (GPT-4.1) for execution"""
    return "gpt-4.1"
