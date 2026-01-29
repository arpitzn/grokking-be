"""Unit tests for router node"""
import pytest
from app.agent.nodes import router_node
from app.agent.state import AgentState


@pytest.mark.asyncio
async def test_router_routes_to_internal_rag():
    """Test router routes to internal RAG for internal queries"""
    state: AgentState = {
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "query": "test",
        "working_memory": [],
        "plan": {"knowledge_source": "internal"},
        "tool_results": [],
        "final_response": "",
        "needs_summarization": False
    }
    
    result = await router_node(state)
    assert result == "internal_rag_tool"


@pytest.mark.asyncio
async def test_router_routes_to_external_search():
    """Test router routes to external search for external queries"""
    state: AgentState = {
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "query": "test",
        "working_memory": [],
        "plan": {"knowledge_source": "external"},
        "tool_results": [],
        "final_response": "",
        "needs_summarization": False
    }
    
    result = await router_node(state)
    assert result == "external_search_tool"


@pytest.mark.asyncio
async def test_router_routes_to_executor():
    """Test router routes to executor for no-retrieval queries"""
    state: AgentState = {
        "user_id": "test_user",
        "conversation_id": "test_conv",
        "query": "test",
        "working_memory": [],
        "plan": {"knowledge_source": "none"},
        "tool_results": [],
        "final_response": "",
        "needs_summarization": False
    }
    
    result = await router_node(state)
    assert result == "executor"
