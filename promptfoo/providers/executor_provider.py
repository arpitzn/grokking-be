"""
PromptFoo Provider for Executor Node Testing

Tests response generation quality with configurable model, temperature,
and max_tokens parameters.

Usage in promptfooconfig.yaml:
  providers:
    - id: python:providers/executor_provider.py
      config:
        model: gpt-4
        temperature: 0.7
        max_tokens: 2000
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_client(model: str, temperature: float, max_tokens: int):
    """Create LLM client with specified configuration"""
    from app.infra.config import settings
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def generate_response(
    query: str,
    working_memory: List[Dict[str, str]],
    tool_results: List[Dict[str, Any]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a response based on query, context, and tool results.

    Args:
        query: User query
        working_memory: Conversation context
        tool_results: Retrieved chunks or search results
        model: LLM model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        prompt_template: Optional custom prompt template

    Returns:
        Dict with response and metadata
    """
    llm = get_llm_client(model, temperature, max_tokens)

    # Build working memory text
    working_memory_text = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in working_memory])
        if working_memory
        else "No previous conversation."
    )

    # Build tool context
    tool_context = ""
    if tool_results:
        for tool_result in tool_results:
            source = tool_result.get("source", "unknown")
            if source == "internal":
                chunks = tool_result.get("chunks", [])
                chunk_texts = "\n".join([c.get("content", "") for c in chunks])
                tool_context += f"\n\nRetrieved from internal documents:\n{chunk_texts}"
            elif source == "external":
                results = tool_result.get("results", [])
                result_texts = "\n".join([r.get("content", "") for r in results])
                tool_context += f"\n\nRetrieved from external search:\n{result_texts}"

    # Use custom prompt or default
    if prompt_template:
        executor_prompt = (
            prompt_template.replace("{{query}}", query)
            .replace("{{context}}", working_memory_text)
            .replace("{{tool_context}}", tool_context)
        )
    else:
        executor_prompt = f"""You are a helpful AI assistant. Answer the user's query based on the conversation context and any retrieved information.

Conversation context:
{working_memory_text}
{tool_context}

User query: {query}

Provide a clear, helpful response based on the available information. If you used retrieved information, synthesize it naturally without mentioning the sources explicitly."""

    from langchain_core.messages import HumanMessage

    response = await llm.ainvoke([HumanMessage(content=executor_prompt)])

    return {
        "response": response.content,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "context_used": bool(tool_context),
    }


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PromptFoo provider entry point.

    Args:
        prompt: The prompt/query to respond to
        options: Provider options including model, temperature, max_tokens
        context: Test context including variables

    Returns:
        Dict with output containing generated response
    """
    # Extract configuration from options
    config = options.get("config", {})
    model = config.get("model", "gpt-4")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 2000)

    # Extract variables from context
    vars = context.get("vars", {})
    query = vars.get("query", prompt)
    working_memory = vars.get("working_memory", [])
    tool_results = vars.get("tool_results", [])
    prompt_template = vars.get("prompt_template")

    # Parse JSON strings if needed
    if isinstance(working_memory, str):
        try:
            working_memory = json.loads(working_memory)
        except json.JSONDecodeError:
            working_memory = []

    if isinstance(tool_results, str):
        try:
            tool_results = json.loads(tool_results)
        except json.JSONDecodeError:
            tool_results = []

    try:
        # Run async generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            generate_response(
                query=query,
                working_memory=working_memory,
                tool_results=tool_results,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_template=prompt_template,
            )
        )
        loop.close()

        # Return result in PromptFoo format
        return {
            "output": result["response"],
            "metadata": {
                "model": result["model"],
                "temperature": result["temperature"],
                "context_used": result["context_used"],
            },
        }
    except Exception as e:
        return {"output": f"Error generating response: {str(e)}", "error": str(e)}


# For direct testing
if __name__ == "__main__":
    test_result = call_api(
        prompt="What is the capital of France?",
        options={"config": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 500}},
        context={"vars": {"query": "What is the capital of France?"}},
    )
    print(json.dumps(test_result, indent=2))
