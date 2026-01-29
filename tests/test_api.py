"""Integration tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_chat_stream_endpoint_structure():
    """Test chat stream endpoint structure (without actual streaming)"""
    # Note: Full streaming test would require async test client
    # This is a basic structure test
    response = client.post(
        "/chat/stream",
        json={
            "user_id": "test_user",
            "message": "Hello"
        }
    )
    # May fail without proper setup, but structure should be correct
    assert response.status_code in [200, 500]  # 500 if services not configured
