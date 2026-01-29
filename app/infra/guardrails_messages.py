"""
User-friendly message templates for NeMo Guardrails detections.

These messages are returned to users when guardrails detect issues,
providing helpful guidance instead of cryptic error messages.
"""

from enum import Enum
from typing import Dict


class GuardrailDetectionType(str, Enum):
    """Types of guardrail detections"""

    PII_DETECTED = "pii_detected"
    JAILBREAK_DETECTED = "jailbreak_detected"
    HARMFUL_CONTENT = "harmful_content"
    HALLUCINATION_WARNING = "hallucination_warning"
    OFF_TOPIC = "off_topic"
    CONTENT_SAFETY = "content_safety"
    UNKNOWN = "unknown"


# User-friendly messages for each detection type
GUARDRAIL_MESSAGES: Dict[str, str] = {
    GuardrailDetectionType.PII_DETECTED: (
        "I noticed your message contains sensitive personal information "
        "(such as email addresses, phone numbers, social security numbers, or credit card details). "
        "For your privacy and security, I'm unable to process messages containing such data. "
        "Please rephrase your question without including personal details, and I'll be happy to help!"
    ),
    GuardrailDetectionType.JAILBREAK_DETECTED: (
        "I'm designed to be helpful, harmless, and honest. "
        "Your request appears to ask me to bypass my safety guidelines or act outside my intended purpose. "
        "I'd be happy to help you with a different question that aligns with how I'm designed to assist. "
        "Feel free to ask me anything else!"
    ),
    GuardrailDetectionType.HARMFUL_CONTENT: (
        "I'm not able to assist with this type of request as it may involve content that could be harmful. "
        "I'm here to provide helpful, accurate, and safe information. "
        "Please feel free to ask me something else, and I'll do my best to help!"
    ),
    GuardrailDetectionType.HALLUCINATION_WARNING: (
        "\n\n---\n"
        "**Note**: I want to be transparent with you - I'm not entirely confident about the accuracy "
        "of some parts of my response above. The information may not be fully verified against "
        "authoritative sources. Please verify any critical information before relying on it."
    ),
    GuardrailDetectionType.OFF_TOPIC: (
        "I'm specifically designed to help with questions related to our company and its products. "
        "This topic seems to be outside my area of expertise. "
        "Is there something about our products or services I can help you with instead?"
    ),
    GuardrailDetectionType.CONTENT_SAFETY: (
        "I'm not able to provide that response as it may contain content that doesn't meet our safety guidelines. "
        "Let me try to help you in a different way. Could you rephrase your question or ask about something else?"
    ),
    GuardrailDetectionType.UNKNOWN: (
        "I encountered an issue processing your request. "
        "Please try rephrasing your question, and I'll do my best to assist you."
    ),
}


def get_friendly_message(detection_type: str) -> str:
    """
    Get a user-friendly message for a specific detection type.

    Args:
        detection_type: The type of guardrail detection (e.g., 'pii_detected', 'jailbreak_detected')

    Returns:
        A user-friendly message explaining the issue and how to proceed
    """
    # Try to get the message for the specific detection type
    if detection_type in GUARDRAIL_MESSAGES:
        return GUARDRAIL_MESSAGES[detection_type]

    # Try enum value
    try:
        detection_enum = GuardrailDetectionType(detection_type)
        return GUARDRAIL_MESSAGES.get(
            detection_enum, GUARDRAIL_MESSAGES[GuardrailDetectionType.UNKNOWN]
        )
    except ValueError:
        return GUARDRAIL_MESSAGES[GuardrailDetectionType.UNKNOWN]


def get_pii_message() -> str:
    """Get the PII detection message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.PII_DETECTED]


def get_jailbreak_message() -> str:
    """Get the jailbreak detection message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.JAILBREAK_DETECTED]


def get_harmful_content_message() -> str:
    """Get the harmful content message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.HARMFUL_CONTENT]


def get_hallucination_warning() -> str:
    """Get the hallucination warning message (to be appended to responses)"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.HALLUCINATION_WARNING]


def get_off_topic_message() -> str:
    """Get the off-topic message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.OFF_TOPIC]


def get_content_safety_message() -> str:
    """Get the content safety message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.CONTENT_SAFETY]
