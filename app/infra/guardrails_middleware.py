"""
Guardrails middleware wrapper for LLM calls
Intercepts model calls, applies safety checks, logs violations
"""

import logging
from typing import Dict, Any, List, Optional

from app.infra.guardrails import validate_message, GuardrailResult

logger = logging.getLogger(__name__)


async def wrap_llm_call(
    llm_func,
    messages: List[Dict[str, str]],
    **kwargs
) -> Any:
    """
    Wrap LLM call with guardrails safety checks.
    
    Args:
        llm_func: The LLM function to call
        messages: List of messages for the LLM
        **kwargs: Additional arguments for the LLM call
    
    Returns:
        LLM response (or safe fallback response)
    """
    # Extract user message for validation
    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    if user_message:
        # Validate message through guardrails
        guardrail_result: GuardrailResult = validate_message(user_message)
        
        if not guardrail_result.passed:
            # Guardrail triggered - log and return safe response
            logger.warning(
                f"Guardrail triggered: {guardrail_result.detection_type}",
                extra={"details": guardrail_result.details}
            )
            
            # Return safe fallback response
            class SafeResponse:
                def __init__(self, content):
                    self.content = content
                    self.usage = None
                
                class MockChoice:
                    def __init__(self, content):
                        self.message = type('obj', (object,), {'content': content})()
                
                @property
                def choices(self):
                    return [self.MockChoice(self.content)]
            
            return SafeResponse(guardrail_result.message)
    
    # No guardrail triggered - proceed with normal LLM call
    return await llm_func(messages, **kwargs)
