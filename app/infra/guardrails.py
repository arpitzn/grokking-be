"""
NeMo Guardrails Integration

This module provides a comprehensive guardrails implementation with:
- Single master switch to enable/disable ALL guardrails (GUARDRAILS_ENABLED)
- PII detection using Presidio
- Jailbreak detection using LLM self-check
- Hallucination detection for RAG responses
- Content safety checks
- User-friendly messages instead of blocking

IMPORTANT: This implementation returns friendly messages to users instead of
throwing HTTP exceptions. All detections are logged for audit purposes.

Configuration:
- Set GUARDRAILS_ENABLED=true to enable all guardrails (PII, jailbreak, hallucination, etc.)
- Set GUARDRAILS_ENABLED=false to disable all guardrails processing
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.infra.config import settings
from app.infra.guardrails_messages import (
    GuardrailDetectionType,
    get_content_safety_message,
    get_friendly_message,
    get_hallucination_warning,
    get_pii_message,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """
    Result from guardrail validation.

    Attributes:
        passed: True if no guardrail was triggered, False if something was detected
        message: The original message if passed, or a friendly message if not passed
        detection_type: Type of detection (pii, jailbreak, etc.) or None if passed
        details: Additional details for logging/audit (e.g., what PII was found)
    """

    passed: bool
    message: str
    detection_type: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HallucinationResult:
    """
    Result from hallucination check.

    Attributes:
        detected: True if hallucination was detected
        confidence: Confidence score (0.0 to 1.0)
        warning_message: Warning message to append to response if detected
    """

    detected: bool
    confidence: float = 0.0
    warning_message: str = ""


class GuardrailsManager:
    """
    NeMo Guardrails manager with single master switch configuration.

    Configuration via environment variable:
    - GUARDRAILS_ENABLED=true: All guardrails active (PII, jailbreak, hallucination, content safety)
    - GUARDRAILS_ENABLED=false: No guardrails processing, all messages pass through
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the GuardrailsManager.

        Args:
            config_path: Optional path to guardrails config directory.
                        Defaults to config/guardrails in the project root.
        """
        # Single master switch controls everything
        self.enabled = settings.guardrails_enabled

        # NeMo Guardrails instance
        self.rails = None
        self.initialized = False

        # If disabled, don't initialize anything
        if not self.enabled:
            logger.info("NeMo Guardrails DISABLED via GUARDRAILS_ENABLED=false")
            logger.info("  - All guardrails functionality is OFF")
            logger.info("  - Messages will pass through without validation")
            return

        # Set default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "guardrails",
            )

        try:
            from nemoguardrails import LLMRails, RailsConfig

            config = RailsConfig.from_path(config_path)
            self.rails = LLMRails(config)
            self.initialized = True
            logger.info("NeMo Guardrails ENABLED and initialized successfully")
            logger.info("  - PII Detection: ACTIVE")
            logger.info("  - Jailbreak Detection: ACTIVE")
            logger.info("  - Hallucination Check: ACTIVE")
            logger.info("  - Content Safety: ACTIVE")
            logger.info("  - Audit Logging: ACTIVE")
        except ImportError as e:
            logger.warning(
                f"NeMo Guardrails import failed: {e}. Running without guardrails."
            )
            self.initialized = False
        except Exception as e:
            logger.warning(
                f"NeMo Guardrails initialization failed: {e}. Running without guardrails."
            )
            self.initialized = False

    async def validate_input(self, message: str, user_id: str) -> GuardrailResult:
        """
        Validate user input through NeMo Guardrails.

        When GUARDRAILS_ENABLED=true, checks for:
        - PII: SSN, credit cards, emails, phone numbers, etc.
        - Jailbreak attempts: Prompt injection, role-play attacks, etc.
        - Custom rules: Domain-specific content restrictions

        When GUARDRAILS_ENABLED=false, all messages pass through unchanged.

        IMPORTANT: This method NEVER throws exceptions. It always returns a
        GuardrailResult with a friendly message if something was detected.

        Args:
            message: The user's input message
            user_id: The user's ID for logging purposes

        Returns:
            GuardrailResult with passed=True if allowed, passed=False with friendly message if blocked
        """
        # If guardrails are disabled, allow everything
        if not self.enabled or not self.initialized:
            return GuardrailResult(passed=True, message=message)

        try:
            # Run NeMo Guardrails input validation
            result = await self.rails.generate_async(
                messages=[{"role": "user", "content": message}]
            )

            # Check if any rail was triggered
            if result and isinstance(result, dict):
                response_content = result.get("content", "")

                # NeMo Guardrails returns blocked responses as {'role': 'assistant', 'content': '...'}
                # We need to detect these refusal patterns in the content
                refusal_patterns = [
                    "i'm not able to help",
                    "i cannot help",
                    "i can't help",
                    "refuse to respond",
                    "unable to assist",
                    "cannot assist",
                    "i'm designed to be helpful, harmless",
                    "not able to help with that request",
                ]
                is_refusal = any(
                    pattern in response_content.lower() for pattern in refusal_patterns
                )

                # Check for blocked/stopped state OR refusal patterns in response
                if (
                    result.get("stop", False)
                    or result.get("blocked", False)
                    or is_refusal
                ):
                    detection_type = self._determine_input_detection_type(
                        result, message
                    )
                    friendly_message = get_friendly_message(detection_type)

                    # Log detection for audit
                    self._log_detection(
                        detection_type=detection_type,
                        user_id=user_id,
                        message_preview=message[:100],
                        result=result,
                    )

                    return GuardrailResult(
                        passed=False,
                        message=friendly_message,
                        detection_type=detection_type,
                        details={
                            "user_id": user_id,
                            "original_message_length": len(message),
                            "rail_response": result,
                        },
                    )

                # Check for PII masking (message was modified but not blocked)
                output_messages = result.get("messages", [])
                if output_messages:
                    modified_content = output_messages[-1].get("content", message)

                    if "[REDACTED]" in modified_content or modified_content != message:
                        # PII was found and masked
                        self._log_detection(
                            detection_type=GuardrailDetectionType.PII_DETECTED,
                            user_id=user_id,
                            message_preview=message[:100],
                            result={"pii_masked": True},
                        )

                        return GuardrailResult(
                            passed=False,
                            message=get_pii_message(),
                            detection_type=GuardrailDetectionType.PII_DETECTED,
                            details={
                                "user_id": user_id,
                                "pii_detected": True,
                                "masked_message": modified_content,
                            },
                        )

            # Input passed all checks
            return GuardrailResult(passed=True, message=message)

        except Exception as e:
            logger.error(f"Guardrails input validation error: {e}")
            # Fail open - allow message if guardrails fail
            return GuardrailResult(
                passed=True,
                message=message,
                details={"error": str(e), "failed_open": True},
            )

    async def validate_output(
        self, response: str, context: Dict[str, Any]
    ) -> GuardrailResult:
        """
        Validate bot output through NeMo Guardrails.

        When GUARDRAILS_ENABLED=true, checks for:
        - Content safety: Harmful, explicit, or abusive content
        - PII in output: Prevents leaking sensitive information

        When GUARDRAILS_ENABLED=false, all responses pass through unchanged.

        IMPORTANT: This method NEVER throws exceptions. It returns a friendly
        message if the output is blocked.

        Args:
            response: The bot's response to validate
            context: Additional context (conversation_id, user_id, etc.)

        Returns:
            GuardrailResult with the original or modified response
        """
        # If guardrails are disabled, allow everything
        if not self.enabled or not self.initialized:
            return GuardrailResult(passed=True, message=response)

        try:
            # Run NeMo Guardrails output validation
            result = await self.rails.generate_async(
                messages=[{"role": "assistant", "content": response}]
            )

            if result and isinstance(result, dict):
                # Check if output was blocked
                if result.get("stop", False) or result.get("blocked", False):
                    friendly_message = get_content_safety_message()

                    self._log_detection(
                        detection_type=GuardrailDetectionType.CONTENT_SAFETY,
                        user_id=context.get("user_id", "unknown"),
                        message_preview=response[:100],
                        result=result,
                    )

                    return GuardrailResult(
                        passed=False,
                        message=friendly_message,
                        detection_type=GuardrailDetectionType.CONTENT_SAFETY,
                        details={"original_response_length": len(response)},
                    )

                # Check for modified output (e.g., PII masking)
                output_messages = result.get("messages", [])
                if output_messages:
                    modified_content = output_messages[-1].get("content", response)
                    return GuardrailResult(passed=True, message=modified_content)

            return GuardrailResult(passed=True, message=response)

        except Exception as e:
            logger.error(f"Guardrails output validation error: {e}")
            # Fail open - return original response
            return GuardrailResult(
                passed=True,
                message=response,
                details={"error": str(e), "failed_open": True},
            )

    async def check_hallucination(
        self, response: str, rag_context: str, user_id: str = "unknown"
    ) -> HallucinationResult:
        """
        Check if the response contains hallucinations compared to RAG context.

        This is a separate check specifically for RAG responses to detect
        when the LLM makes claims not supported by the retrieved context.

        Only runs when GUARDRAILS_ENABLED=true.

        Args:
            response: The bot's response to check
            rag_context: The retrieved context from knowledge base
            user_id: User ID for logging

        Returns:
            HallucinationResult indicating if hallucination was detected
        """
        # If guardrails are disabled, return no detection
        if not self.enabled or not self.initialized:
            return HallucinationResult(detected=False, confidence=0.0)

        try:
            # Use the hallucination check flow
            # NeMo Guardrails will use the self_check_hallucination prompt
            result = await self.rails.generate_async(
                messages=[
                    {"role": "context", "content": rag_context},
                    {"role": "assistant", "content": response},
                ],
                options={"check_hallucination": True},
            )

            if result and isinstance(result, dict):
                # Check if hallucination was detected
                hallucination_detected = result.get("hallucination_detected", False)
                confidence = result.get("hallucination_score", 0.0)

                if hallucination_detected:
                    self._log_detection(
                        detection_type=GuardrailDetectionType.HALLUCINATION_WARNING,
                        user_id=user_id,
                        message_preview=response[:100],
                        result={"confidence": confidence},
                    )

                    return HallucinationResult(
                        detected=True,
                        confidence=confidence,
                        warning_message=get_hallucination_warning(),
                    )

            return HallucinationResult(detected=False, confidence=0.0)

        except Exception as e:
            logger.error(f"Hallucination check error: {e}")
            # Don't add warning if check fails
            return HallucinationResult(detected=False, confidence=0.0)

    def _determine_input_detection_type(
        self, result: Dict[str, Any], message: str
    ) -> str:
        """
        Determine the type of input detection based on the guardrails result.

        Args:
            result: The result from NeMo Guardrails
            message: The original user message

        Returns:
            The detection type string
        """
        # Check for specific detection types in the result
        if "pii" in str(result).lower() or "[REDACTED]" in str(result):
            return GuardrailDetectionType.PII_DETECTED

        # Check the original message for PII patterns
        # This is important because NeMo may block PII but not include "pii" in the response
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),  # Social Security Number: 123-45-6789
            (r"\b\d{3}\s\d{2}\s\d{4}\b", "SSN"),  # SSN with spaces: 123 45 6789
            (
                r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                "Credit Card",
            ),  # Credit card: 1234-5678-9012-3456
            (r"\b\d{13,19}\b", "Credit Card"),  # Credit card without separators
            (
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "Email",
            ),  # Email address
            (
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "Phone",
            ),  # Phone: 123-456-7890 or 123.456.7890
            (r"\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b", "Phone"),  # Phone: (123) 456-7890
            (r"\b\d{10}\b", "Phone"),  # Phone without separators: 1234567890
            (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                "Date",
            ),  # Date that might be DOB: MM/DD/YYYY
            (r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", "Date"),  # Date: YYYY-MM-DD
        ]

        for pattern, pii_type in pii_patterns:
            if re.search(pattern, message):
                return GuardrailDetectionType.PII_DETECTED

        if "jailbreak" in str(result).lower():
            return GuardrailDetectionType.JAILBREAK_DETECTED

        # Check message patterns for jailbreak attempts
        jailbreak_patterns = [
            "ignore previous",
            "forget your instructions",
            "pretend you are",
            "act as if",
            "you are now",
            "dan mode",
            "developer mode",
            "bypass",
            "jailbreak",
        ]
        message_lower = message.lower()
        if any(pattern in message_lower for pattern in jailbreak_patterns):
            return GuardrailDetectionType.JAILBREAK_DETECTED

        # Check for harmful content indicators
        if "harmful" in str(result).lower() or "blocked" in str(result).lower():
            return GuardrailDetectionType.HARMFUL_CONTENT

        # Default to generic harmful content
        return GuardrailDetectionType.HARMFUL_CONTENT

    def _log_detection(
        self,
        detection_type: str,
        user_id: str,
        message_preview: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Log guardrail detection for audit purposes.

        Args:
            detection_type: Type of detection
            user_id: User ID
            message_preview: First 100 chars of message
            result: Raw result from guardrails
        """
        log_data = {
            "event": "guardrail_detection",
            "detection_type": detection_type,
            "user_id": user_id,
            "message_preview": message_preview[:100] if message_preview else "",
            "details": str(result)[:500],  # Limit size
        }

        logger.warning(f"Guardrail detection: {json.dumps(log_data)}")


# Global guardrails manager instance
guardrails_manager: Optional[GuardrailsManager] = None


def get_guardrails_manager() -> GuardrailsManager:
    """
    Get or create the global guardrails manager instance.

    This is a singleton pattern to ensure only one instance exists.

    Returns:
        The global GuardrailsManager instance
    """
    global guardrails_manager
    if guardrails_manager is None:
        guardrails_manager = GuardrailsManager()
    return guardrails_manager


def reset_guardrails_manager() -> None:
    """
    Reset the global guardrails manager instance.

    Useful for testing or when configuration changes.
    """
    global guardrails_manager
    guardrails_manager = None
