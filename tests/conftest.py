"""Pytest configuration and fixtures"""
import pytest
import os
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(".env.test", override=False)


@pytest.fixture(scope="session")
def test_settings():
    """Test settings fixture"""
    return {
        "mongodb_uri": os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017/test"),
        "openai_api_key": os.getenv("TEST_OPENAI_API_KEY", "test-key"),
    }
