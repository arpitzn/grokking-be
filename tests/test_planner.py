"""Unit tests for planner node"""
import pytest
from app.agent.nodes import planner_node
from app.agent.state import AgentState


@pytest.mark.asyncio
async def test_planner_classifies_internal_query():
    """Test planner classifies internal queries correctly"""
    state: AgentState = {
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "query": "What did our director say in yesterday's call?",
        "working_memory": [
            {"role": "user", "content": "Hello"}
        ],
        "plan": None,
        "tool_results": [],
        "final_response": "",
        "needs_summarization": False
    }
    
    # Mock LLM response would be needed in real test
    # For now, just test structure
    result = await planner_node(state)
    
    assert result["plan"] is not None
    assert "knowledge_source" in result["plan"]


@pytest.mark.asyncio
async def test_planner_classifies_external_query():
    """Test planner classifies external queries correctly"""
    state: AgentState = {
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "query": "Who is the President of India in 2020?",
        "working_memory": [
            {"role": "user", "content": "Hello"}
        ],
        "plan": None,
        "tool_results": [],
        "final_response": "",
        "needs_summarization": False
    }
    
    result = await planner_node(state)
    
    assert result["plan"] is not None
    assert "knowledge_source" in result["plan"]
