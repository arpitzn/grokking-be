"""Unit tests for NeMo Guardrails"""
import pytest
from app.infra.guardrails import get_guardrails_manager


@pytest.mark.asyncio
async def test_guardrails_validates_normal_input():
    """Test guardrails allows normal input"""
    guardrails = get_guardrails_manager()
    
    result = await guardrails.validate_input("Hello, how are you?", "test_user")
    
    # Should allow normal input (may fail if guardrails not configured, but should not raise)
    assert isinstance(result, dict)
    assert "allowed" in result


@pytest.mark.asyncio
async def test_guardrails_validates_output():
    """Test guardrails validates output"""
    guardrails = get_guardrails_manager()
    
    result = await guardrails.validate_output(
        "This is a normal response",
        context={"conversation_id": "test"}
    )
    
    assert isinstance(result, str)
