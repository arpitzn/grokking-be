"""Unit tests for external search stub tool"""
import pytest
from app.infra.external_search import external_search_stub


@pytest.mark.asyncio
async def test_external_search_returns_stub_result():
    """Test external search stub returns hardcoded result"""
    result = await external_search_stub.search("president of india 2020")
    
    assert "results" in result
    assert "is_stub" in result
    assert result["is_stub"] is True
    assert len(result["results"]) > 0


@pytest.mark.asyncio
async def test_external_search_handles_unknown_query():
    """Test external search stub handles unknown queries"""
    result = await external_search_stub.search("unknown query about something")
    
    assert "results" in result
    assert len(result["results"]) > 0  # Should return default result
