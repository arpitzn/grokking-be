"""Agent state definition for LangGraph"""
from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict):
    """State passed between LangGraph nodes"""
    user_id: str
    conversation_id: str
    query: str
    working_memory: List[Dict[str, str]]  # Assembled context
    plan: Optional[Dict[str, Any]]  # {knowledge_source, steps, reasoning, query_for_retrieval}
    tool_results: List[Dict[str, Any]]  # Results from RAG or external search
    final_response: str  # Final synthesized response
    needs_summarization: bool  # Whether to trigger summarization
