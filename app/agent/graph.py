"""LangGraph definition with parallel execution"""
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from app.agent.state import AgentState
from app.agent.nodes import (
    memory_node,
    planner_node,
    router_node,
    internal_rag_tool_node,
    external_search_tool_node,
    executor_node,
    summarizer_node,
    semantic_memory_node
)
import logging

logger = logging.getLogger(__name__)


def create_graph():
    """Create and configure LangGraph"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("internal_rag_tool", internal_rag_tool_node)
    graph.add_node("external_search_tool", external_search_tool_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("semantic_memory", semantic_memory_node)
    
    # Set entry point
    graph.set_entry_point("memory")
    
    # Add edges
    graph.add_edge("memory", "planner")
    graph.add_edge("planner", "router")
    
    # Conditional routing based on knowledge_source
    graph.add_conditional_edges(
        "router",
        router_node,
        {
            "internal_rag_tool": "internal_rag_tool",
            "external_search_tool": "external_search_tool",
            "executor": "executor"
        }
    )
    
    # Both tool nodes route to executor
    graph.add_edge("internal_rag_tool", "executor")
    graph.add_edge("external_search_tool", "executor")
    
    # Parallel execution after executor: Send semantic memory write and optionally summarizer
    def after_executor(state: AgentState):
        """Route after executor - parallel execution using Send"""
        results = []
        # Always send semantic memory write (non-blocking)
        results.append(Send("semantic_memory", state))
        # Conditionally send summarizer if needed
        if state.get("needs_summarization", False):
            results.append(Send("summarizer", state))
        return results
    
    graph.add_conditional_edges("executor", after_executor)
    
    # Both parallel nodes end
    graph.add_edge("summarizer", END)
    graph.add_edge("semantic_memory", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("LangGraph compiled successfully")
    return compiled_graph


# Global graph instance
_graph = None


def get_graph():
    """Get or create graph instance"""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
