"""
PromptFoo Provider for Planner Node Testing

Tests query classification accuracy (INTERNAL vs EXTERNAL vs NONE)
with configurable model and temperature parameters.

Usage in promptfooconfig.yaml:
  providers:
    - id: python:providers/planner_provider.py
      config:
        model: gpt-3.5-turbo
        temperature: 0.3
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_llm_client(model: str, temperature: float):
    """Create LLM client with specified configuration"""
    from app.infra.config import settings
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=model,
        temperature=temperature,
        max_tokens=500,
    )


async def classify_query(
    query: str,
    working_memory: list,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.3,
    prompt_template: Optional[str] = None,
    persona: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify a user query into knowledge source categories.

    Args:
        query: User query to classify
        working_memory: Conversation context
        model: LLM model to use
        temperature: Temperature for generation
        prompt_template: Optional custom prompt template

    Returns:
        Dict with knowledge_source, reasoning, and query_for_retrieval
    """
    llm = get_llm_client(model, temperature)

    # Build working memory text
    working_memory_text = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in working_memory])
        if working_memory
        else "No previous conversation."
    )

    # Use custom prompt or default
    if prompt_template:
        planner_prompt = prompt_template.replace("{{query}}", query).replace(
            "{{context}}", working_memory_text
        )
        if persona:
            planner_prompt = planner_prompt.replace("{{persona}}", persona)
    else:
        # Add persona context to prompt if provided
        persona_context = ""
        if persona:
            persona_context = f"\n\nPersona: {persona}\n"
            if persona == "end_customer":
                persona_context += "Note: This is a customer query. Focus on customer-facing policies and order information.\n"
            elif persona == "customer_care_rep":
                persona_context += "Note: This is a customer care representative query. Focus on SOPs, policies, and operational data.\n"
            elif persona == "area_manager":
                persona_context += "Note: This is an area manager query. Focus on operational metrics, zone data, and analytics.\n"
        
        planner_prompt = f"""You are a planning agent that classifies user queries and creates execution plans for a food delivery support system.

Given the user query and conversation context, determine the appropriate knowledge source:

1. **INTERNAL** - Use internal RAG (company documents, policies, SOPs) if:
   - Query references company-specific information (policies, SOPs, SLAs, refund rules)
   - Query asks about food delivery operations (orders, refunds, delivery delays, quality issues)
   - Query mentions internal processes or procedures
   - Examples: "What is the refund policy?", "How do I handle a delayed order?", "What are the SLA guidelines?"

2. **EXTERNAL** - Use external search (web search) if:
   - Query asks about public facts, current events, general knowledge unrelated to food delivery
   - Query references well-known people, places, or historical events
   - Query needs real-time or recent information outside company knowledge
   - Examples: "Who is the President of India in 2020?", "What is the capital of France?"

3. **NONE** - No retrieval needed if:
   - Query is conversational (greetings, clarifications, acknowledgments)
   - Query can be answered from conversation history alone
   - Examples: "Thanks!", "Can you explain that differently?", "Hello!"

{persona_context}Conversation context:
{working_memory_text}

User query: {query}

Return JSON only:
{{
  "knowledge_source": "internal" | "external" | "none",
  "reasoning": "Brief explanation of classification",
  "query_for_retrieval": "Optimized query string (if retrieval needed, otherwise empty string)"
}}"""

    from langchain_core.messages import HumanMessage

    response = await llm.ainvoke([HumanMessage(content=planner_prompt)])
    response_text = response.content.strip()

    # Parse JSON from response
    try:
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        plan = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback to default
        plan = {
            "knowledge_source": "none",
            "reasoning": "Failed to parse planner response",
            "query_for_retrieval": "",
            "raw_response": response_text,
        }

    return plan


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PromptFoo provider entry point.

    Args:
        prompt: The prompt/query to classify
        options: Provider options including model and temperature
        context: Test context including variables

    Returns:
        Dict with output containing classification result
    """
    # Extract configuration from options
    config = options.get("config", {})
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.3)

    # Extract variables from context
    vars = context.get("vars", {})
    query = vars.get("query", prompt)
    working_memory = vars.get("working_memory", [])
    prompt_template = vars.get("prompt_template")
    persona = vars.get("persona")

    # If working_memory is a string (JSON), parse it
    if isinstance(working_memory, str):
        try:
            working_memory = json.loads(working_memory)
        except json.JSONDecodeError:
            working_memory = []

    try:
        # Run async classification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            classify_query(
                query=query,
                working_memory=working_memory,
                model=model,
                temperature=temperature,
                prompt_template=prompt_template,
                persona=persona,
            )
        )
        loop.close()

        # Return result in PromptFoo format
        return {
            "output": json.dumps(result),
            "tokenUsage": {
                "total": 0,  # Would need to track this from LangChain
                "prompt": 0,
                "completion": 0,
            },
        }
    except Exception as e:
        return {
            "output": json.dumps(
                {
                    "error": str(e),
                    "knowledge_source": "error",
                    "reasoning": f"Error during classification: {str(e)}",
                }
            ),
            "error": str(e),
        }


# For direct testing
if __name__ == "__main__":
    test_result = call_api(
        prompt="What did our director say in yesterday's call?",
        options={"config": {"model": "gpt-3.5-turbo", "temperature": 0.3}},
        context={"vars": {"query": "What did our director say in yesterday's call?"}},
    )
    print(json.dumps(test_result, indent=2))
