"""
User-friendly message templates for NeMo Guardrails detections.

These messages are returned to users when guardrails detect issues,
providing helpful guidance instead of cryptic error messages.

Includes variation system to prevent robotic repetitive responses.
"""

from enum import Enum
from typing import Dict, List, Optional


class GuardrailDetectionType(str, Enum):
    """Types of guardrail detections"""

    PII_DETECTED = "pii_detected"
    JAILBREAK_DETECTED = "jailbreak_detected"
    HARMFUL_CONTENT = "harmful_content"
    HALLUCINATION_WARNING = "hallucination_warning"
    OFF_TOPIC = "off_topic"
    CONTENT_SAFETY = "content_safety"
    VIOLENCE_DETECTED = "violence_detected"
    SELF_HARM_DETECTED = "self_harm_detected"
    SEXUAL_CONTENT_DETECTED = "sexual_content_detected"
    HATE_SPEECH_DETECTED = "hate_speech_detected"
    UNKNOWN = "unknown"


# Base user-friendly messages for each detection type (fallback)
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
    GuardrailDetectionType.VIOLENCE_DETECTED: (
        "I understand you're frustrated with your experience, and I genuinely want to help resolve this. "
        "However, I'm not able to engage with messages that suggest harm to anyone. "
        "Let's focus on solving your order issue - what specifically went wrong? I'm here to make this right."
    ),
    GuardrailDetectionType.SELF_HARM_DETECTED: (
        "I'm really sorry you're feeling this way. Your wellbeing matters more than any order issue. "
        "If you're going through a difficult time, please reach out to someone who can help: "
        "iCall: 9152987821, Vandrevala Foundation: 1860-2662-345 (24/7), NIMHANS: 080-46110007. "
        "Once you're feeling better, I'm here to help with your order. Take care of yourself first. 💙"
    ),
    GuardrailDetectionType.SEXUAL_CONTENT_DETECTED: (
        "I'm your food delivery support assistant, and I'm here to help with order-related questions only. "
        "Is there something about your order I can help you with today?"
    ),
    GuardrailDetectionType.HATE_SPEECH_DETECTED: (
        "I want to help you with your order concern, but I noticed some language that goes against our community values. "
        "Everyone deserves to be treated with respect - our customers, restaurant partners, and delivery partners. "
        "Could you please share what's bothering you about your order? I'll do my best to help."
    ),
    GuardrailDetectionType.UNKNOWN: (
        "I encountered an issue processing your request. "
        "Please try rephrasing your question, and I'll do my best to assist you."
    ),
}


# Variation templates for each detection type (3-4 variations each)
GUARDRAIL_MESSAGE_VARIATIONS: Dict[str, List[str]] = {
    GuardrailDetectionType.PII_DETECTED: [
        (
            "I noticed your message contains sensitive personal information "
            "(such as email addresses, phone numbers, social security numbers, or credit card details). "
            "For your privacy and security, I'm unable to process messages containing such data. "
            "Please rephrase your question without including personal details, and I'll be happy to help!"
        ),
        (
            "For your privacy and security, I can't process messages with personal details "
            "(like email addresses, phone numbers, or credit card information). "
            "Could you please rephrase your question without including any sensitive information? "
            "I'm here to help!"
        ),
        (
            "I see some sensitive information in your message. To protect your privacy, "
            "I'm not able to handle messages containing personal details such as phone numbers, "
            "email addresses, or payment information. Please share your question without those details, "
            "and I'll be glad to assist you."
        ),
        (
            "Your message appears to include personal information that I can't process for security reasons. "
            "This includes things like contact details, identification numbers, or financial information. "
            "Feel free to ask your question again without including any personal data, and I'll help right away!"
        ),
    ],
    GuardrailDetectionType.JAILBREAK_DETECTED: [
        (
            "I'm designed to be helpful, harmless, and honest. "
            "Your request appears to ask me to bypass my safety guidelines or act outside my intended purpose. "
            "I'd be happy to help you with a different question that aligns with how I'm designed to assist. "
            "Feel free to ask me anything else!"
        ),
        (
            "I understand you're trying something creative, but I need to stay within my designed purpose. "
            "I'm here to help with food delivery support questions. "
            "Is there something about your order or our service I can help you with instead?"
        ),
        (
            "I appreciate your creativity, but I'm designed specifically to help with food delivery support. "
            "I can't change my behavior or bypass my guidelines. "
            "What can I help you with regarding your order or our service today?"
        ),
        (
            "I'm here to assist with food delivery support, and I need to stay true to that purpose. "
            "Your request seems to ask me to act differently than intended. "
            "Let's focus on how I can help with your order or delivery questions - what do you need?"
        ),
    ],
    GuardrailDetectionType.VIOLENCE_DETECTED: [
        (
            "I understand you're frustrated with your experience, and I genuinely want to help resolve this. "
            "However, I'm not able to engage with messages that suggest harm to anyone. "
            "Let's focus on solving your order issue - what specifically went wrong? I'm here to make this right."
        ),
        (
            "I hear your frustration, and I want to help make things right with your order. "
            "I can't engage with language that suggests harm, but I'm absolutely here to address your concerns. "
            "Can you tell me what happened with your order so I can help resolve it?"
        ),
        (
            "I understand you're upset about your experience, and I'm here to help fix it. "
            "However, I need to keep our conversation focused on resolving your order issue constructively. "
            "What specifically went wrong? Let me help make this right."
        ),
    ],
    GuardrailDetectionType.SELF_HARM_DETECTED: [
        (
            "I'm really sorry you're feeling this way. Your wellbeing matters more than any order issue. "
            "If you're going through a difficult time, please reach out to someone who can help: "
            "iCall: 9152987821, Vandrevala Foundation: 1860-2662-345 (24/7), NIMHANS: 080-46110007. "
            "Once you're feeling better, I'm here to help with your order. Take care of yourself first. 💙"
        ),
        (
            "Your wellbeing is the most important thing right now. "
            "If you're struggling, please contact someone who can help: "
            "iCall: 9152987821, Vandrevala Foundation: 1860-2662-345 (24/7), NIMHANS: 080-46110007. "
            "When you're ready, I'm here to help with your order. Please take care. 💙"
        ),
        (
            "I'm concerned about you. Your mental health matters more than anything else. "
            "Please reach out for support: iCall: 9152987821, Vandrevala Foundation: 1860-2662-345 (24/7), "
            "NIMHANS: 080-46110007. I'll be here when you're ready to discuss your order. Take care. 💙"
        ),
    ],
    GuardrailDetectionType.SEXUAL_CONTENT_DETECTED: [
        (
            "I'm your food delivery support assistant, and I'm here to help with order-related questions only. "
            "Is there something about your order I can help you with today?"
        ),
        (
            "I'm here specifically to help with food delivery and order support. "
            "Is there an issue with your order or delivery that I can assist with?"
        ),
        (
            "I focus on helping with food delivery questions and order support. "
            "What can I help you with regarding your order today?"
        ),
    ],
    GuardrailDetectionType.HATE_SPEECH_DETECTED: [
        (
            "I want to help you with your order concern, but I noticed some language that goes against our community values. "
            "Everyone deserves to be treated with respect - our customers, restaurant partners, and delivery partners. "
            "Could you please share what's bothering you about your order? I'll do my best to help."
        ),
        (
            "I'm here to help with your order, but I need to keep our conversation respectful. "
            "We value everyone in our community. Can you tell me what's wrong with your order "
            "so I can help resolve it?"
        ),
        (
            "I understand you're frustrated, but let's keep our conversation respectful. "
            "Everyone deserves dignity - our customers, partners, and team members. "
            "What's the issue with your order? I'm here to help fix it."
        ),
    ],
    GuardrailDetectionType.HARMFUL_CONTENT: [
        (
            "I'm not able to assist with this type of request as it may involve content that could be harmful. "
            "I'm here to provide helpful, accurate, and safe information. "
            "Please feel free to ask me something else, and I'll do my best to help!"
        ),
        (
            "I can't help with that type of request, but I'm here to assist with food delivery support. "
            "Is there something about your order or our service I can help you with?"
        ),
        (
            "That's outside what I can help with, but I'm here for food delivery support. "
            "What can I help you with regarding your order today?"
        ),
    ],
    GuardrailDetectionType.CONTENT_SAFETY: [
        (
            "I'm not able to provide that response as it may contain content that doesn't meet our safety guidelines. "
            "Let me try to help you in a different way. Could you rephrase your question or ask about something else?"
        ),
        (
            "I can't provide that type of response, but I'm here to help with food delivery questions. "
            "Could you rephrase your question or ask about your order instead?"
        ),
        (
            "That's not something I can help with, but I'm here for order support. "
            "What can I help you with regarding your food delivery?"
        ),
    ],
    GuardrailDetectionType.OFF_TOPIC: [
        (
            "I'm specifically designed to help with questions related to our company and its products. "
            "This topic seems to be outside my area of expertise. "
            "Is there something about our products or services I can help you with instead?"
        ),
        (
            "I focus on food delivery support, so I'm not the best person to help with that topic. "
            "Is there something about your order or our service I can assist with?"
        ),
        (
            "That's outside my area - I specialize in food delivery support. "
            "What can I help you with regarding your order or delivery?"
        ),
    ],
}


def get_friendly_message(detection_type: str, conversation_id: Optional[str] = None) -> str:
    """
    Get a user-friendly message for a specific detection type with variation support.

    Args:
        detection_type: The type of guardrail detection (e.g., 'pii_detected', 'jailbreak_detected')
        conversation_id: Optional conversation ID for variation selection

    Returns:
        A user-friendly message explaining the issue and how to proceed
    """
    # Get variations if available, otherwise use base message
    variations = GUARDRAIL_MESSAGE_VARIATIONS.get(detection_type, [])
    
    if variations and conversation_id:
        # Use hash-based selection for consistent variation per conversation
        # This ensures same conversation gets same variation, but different conversations get different ones
        hash_value = hash(f"{conversation_id}_{detection_type}")
        idx = abs(hash_value) % len(variations)
        return variations[idx]
    
    # Fallback to base message or first variation
    if variations:
        return variations[0]
    
    # Try to get the base message for the specific detection type
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


def get_pii_message(conversation_id: Optional[str] = None) -> str:
    """Get the PII detection message with variation support"""
    return get_friendly_message(GuardrailDetectionType.PII_DETECTED, conversation_id=conversation_id)


def get_jailbreak_message(conversation_id: Optional[str] = None) -> str:
    """Get the jailbreak detection message with variation support"""
    return get_friendly_message(GuardrailDetectionType.JAILBREAK_DETECTED, conversation_id=conversation_id)


def get_harmful_content_message() -> str:
    """Get the harmful content message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.HARMFUL_CONTENT]


def get_hallucination_warning() -> str:
    """Get the hallucination warning message (to be appended to responses)"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.HALLUCINATION_WARNING]


def get_off_topic_message() -> str:
    """Get the off-topic message"""
    return GUARDRAIL_MESSAGES[GuardrailDetectionType.OFF_TOPIC]


def get_content_safety_message(conversation_id: Optional[str] = None) -> str:
    """Get the content safety message with variation support"""
    return get_friendly_message(GuardrailDetectionType.CONTENT_SAFETY, conversation_id=conversation_id)


def get_i_dont_know_message() -> str:
    """Get the 'I don't know' message for low confidence responses"""
    return (
        "I want to be honest with you - I'm not certain about this specific situation.\n\n"
        "Rather than give you incorrect information, let me connect you with a "
        "specialist who can give you a definitive answer."
    )
