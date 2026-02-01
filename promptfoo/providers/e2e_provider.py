"""
PromptFoo Provider for End-to-End Chat Testing

Tests the complete chat flow via /chat/stream API endpoint,
including SSE streaming handling.

Usage in promptfooconfig.yaml:
  providers:
    - id: python:providers/e2e_provider.py
      config:
        base_url: http://localhost:8000
        timeout: 60
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

import httpx

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def stream_chat(
    base_url: str,
    user_id: str,
    message: str,
    conversation_id: Optional[str] = None,
    persona: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Make a streaming chat request and collect the full response.

    Args:
        base_url: API base URL
        user_id: User identifier
        message: Message to send
        conversation_id: Optional existing conversation ID
        timeout: Request timeout in seconds

    Returns:
        Dict with response content and metadata
    """
    url = f"{base_url}/chat/stream"
    payload = {"user_id": user_id, "message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if persona:
        payload["persona"] = persona

    full_response = ""
    thinking_phases = []
    guardrail_triggered = None
    status = "unknown"
    evidence_cards = []
    refund_recommendations = []
    incident_banners = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                return {
                    "error": f"HTTP {response.status_code}: {error_text.decode()}",
                    "status_code": response.status_code,
                    "response": "",
                }

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        break

                    try:
                        parsed = json.loads(data)

                        # Handle thinking events
                        if parsed.get("event") == "thinking":
                            thinking_phases.append(
                                {
                                    "phase": parsed.get("phase"),
                                    "content": parsed.get("content", ""),
                                }
                            )
                        
                        # Handle evidence card events
                        if parsed.get("event") == "evidence_card":
                            evidence_cards.append(parsed.get("data", {}))
                        
                        # Handle refund recommendation events
                        if parsed.get("event") == "refund_recommendation":
                            refund_recommendations.append(parsed.get("data", {}))
                        
                        # Handle incident banner events
                        if parsed.get("event") == "incident_banner":
                            incident_banners.append(parsed.get("data", {}))

                        # Handle content chunks
                        if "content" in parsed and not parsed.get("event"):
                            full_response += parsed["content"]

                        # Handle completion
                        if parsed.get("status") == "completed":
                            status = "completed"
                            guardrail_triggered = parsed.get("guardrail_triggered")

                        # Handle error
                        if parsed.get("status") == "error" or parsed.get("error"):
                            status = "error"
                            if not full_response:
                                full_response = parsed.get("error", "Unknown error")
                    except json.JSONDecodeError:
                        continue

    return {
        "response": full_response,
        "thinking_phases": thinking_phases,
        "guardrail_triggered": guardrail_triggered,
        "status": status,
        "conversation_id": conversation_id,
        "evidence_cards": evidence_cards,
        "refund_recommendations": refund_recommendations,
        "incident_banners": incident_banners,
    }


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    PromptFoo provider entry point.

    Args:
        prompt: The message to send
        options: Provider options including base_url, timeout
        context: Test context including variables

    Returns:
        Dict with output containing chat response
    """
    # Extract configuration from options
    config = options.get("config", {})
    base_url = config.get(
        "base_url", os.getenv("API_BASE_URL", "http://localhost:8000")
    )
    timeout = config.get("timeout", 60)

    # Extract variables from context
    vars = context.get("vars", {})
    message = vars.get("message", prompt)
    user_id = vars.get("user_id", "promptfoo_test_user")
    conversation_id = vars.get("conversation_id")
    persona = vars.get("persona")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            stream_chat(
                base_url=base_url,
                user_id=user_id,
                message=message,
                conversation_id=conversation_id,
                persona=persona,
                timeout=timeout,
            )
        )

        loop.close()

        # Return result in PromptFoo format
        return {
            "output": result["response"],
            "metadata": {
                "thinking_phases": result.get("thinking_phases", []),
                "guardrail_triggered": result.get("guardrail_triggered"),
                "status": result.get("status", "unknown"),
                "evidence_cards": result.get("evidence_cards", []),
                "refund_recommendations": result.get("refund_recommendations", []),
                "incident_banners": result.get("incident_banners", []),
                "conversation_id": result.get("conversation_id"),
            },
        }
    except httpx.ConnectError as e:
        return {
            "output": f"Connection error: Could not connect to {base_url}. Is the server running?",
            "error": str(e),
        }
    except httpx.TimeoutException as e:
        return {
            "output": f"Timeout: Request exceeded {timeout} seconds",
            "error": str(e),
        }
    except Exception as e:
        return {"output": f"Error: {str(e)}", "error": str(e)}


# For direct testing
if __name__ == "__main__":
    import sys

    # Test basic chat
    print("Testing E2E Chat Provider...")

    test_result = call_api(
        prompt="Hello, how are you?",
        options={"config": {"base_url": "http://localhost:8000", "timeout": 30}},
        context={"vars": {"message": "Hello, how are you?", "user_id": "test_user"}},
    )

    print(f"Response: {test_result['output'][:200]}...")
    if test_result.get("metadata"):
        print(f"Status: {test_result['metadata'].get('status')}")
        print(
            f"Thinking phases: {len(test_result['metadata'].get('thinking_phases', []))}"
        )
