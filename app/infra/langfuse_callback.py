"""Langfuse CallbackHandler for automatic tracing with domain-specific metadata"""
import os
from typing import Any, Dict
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from app.infra.config import settings

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host
)

# Set tracing environment from settings (defaults to development if not set)
environment = getattr(settings, 'environment', 'development')
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = environment

# Create global CallbackHandler instance (singleton pattern)
langfuse_handler = CallbackHandler()


class DomainMetadataCallback(BaseCallbackHandler):
    """
    Custom callback to enrich Langfuse traces with food delivery domain metadata.
    Extracts metadata from agent state and adds to trace metadata.
    """
    
    def __init__(self):
        super().__init__()
        self.current_state: Dict[str, Any] = {}
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Capture agent state on chain start"""
        if isinstance(inputs, dict):
            # Extract domain metadata from agent state
            case = inputs.get("case", {})
            intent = inputs.get("intent", {})
            analysis = inputs.get("analysis", {})
            guardrails = inputs.get("guardrails", {})
            
            # Store for later use
            self.current_state = {
                "persona": case.get("persona"),
                "issue_type": intent.get("issue_type"),
                "severity": intent.get("severity"),
                "zone_id": case.get("zone_id"),
                "order_id": case.get("order_id"),
                "user_id": case.get("user_id"),  # Changed from customer_id
                "SLA_risk": intent.get("SLA_risk"),
                "confidence": analysis.get("confidence"),
                "routing_decision": guardrails.get("routing_decision"),
            }
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Update trace metadata with domain information"""
        # Update metadata with domain-specific fields
        metadata = kwargs.get("metadata", {})
        metadata.update({
            "domain": "food_delivery",
            **{k: v for k, v in self.current_state.items() if v is not None}
        })
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs: Any
    ) -> None:
        """Add domain metadata to LLM calls"""
        metadata = kwargs.get("metadata", {})
        metadata.update({
            "domain": "food_delivery",
            **{k: v for k, v in self.current_state.items() if v is not None}
        })


# Create domain metadata callback instance
domain_metadata_callback = DomainMetadataCallback()
