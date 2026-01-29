"""NeMo Guardrails integration"""
from nemoguardrails import RailsConfig, LLMRails
from app.infra.config import settings
from typing import Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class GuardrailsManager:
    """NeMo Guardrails manager for input/output validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "guardrails"
            )
        
        try:
            config = RailsConfig.from_path(config_path)
            self.rails = LLMRails(config)
            self.enabled = True
            logger.info("NeMo Guardrails initialized successfully")
        except Exception as e:
            logger.warning(f"NeMo Guardrails initialization failed: {e}. Continuing without guardrails.")
            self.rails = None
            self.enabled = False
    
    async def validate_input(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Validates user input through NeMo Guardrails
        
        Returns:
            {
                "allowed": bool,
                "message": str,
                "reason": Optional[str]
            }
        """
        if not self.enabled:
            return {"allowed": True, "message": message, "reason": None}
        
        try:
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": message}]
            )
            
            # Check if guardrails blocked the message
            if result and isinstance(result, dict):
                if result.get("stop", False):
                    return {
                        "allowed": False,
                        "message": "",
                        "reason": result.get("messages", [{}])[0].get("content", "Input blocked by guardrails")
                    }
            
            return {"allowed": True, "message": message, "reason": None}
        except Exception as e:
            logger.error(f"Guardrails validation error: {e}")
            # Fail open - allow message if guardrails fail
            return {"allowed": True, "message": message, "reason": None}
    
    async def validate_output(self, response: str, context: Dict[str, Any]) -> str:
        """
        Validates and potentially modifies LLM output
        
        Returns:
            sanitized response
        """
        if not self.enabled:
            return response
        
        try:
            result = await self.rails.generate_async(
                messages=[
                    {"role": "assistant", "content": response}
                ],
                context=context
            )
            
            if result and isinstance(result, dict):
                messages = result.get("messages", [])
                if messages:
                    return messages[0].get("content", response)
            
            return response
        except Exception as e:
            logger.error(f"Guardrails output validation error: {e}")
            # Fail open - return original response
            return response


# Global guardrails manager instance
guardrails_manager: Optional[GuardrailsManager] = None


def get_guardrails_manager() -> GuardrailsManager:
    """Get or create guardrails manager instance"""
    global guardrails_manager
    if guardrails_manager is None:
        guardrails_manager = GuardrailsManager()
    return guardrails_manager
