"""Memory retrieval subgraph - simplified with ToolNode"""

import logging
import json
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.infra.llm import get_cheap_model, get_llm_service
from app.infra.retrieval_prompts import get_prompts
from app.tools.registry import MEMORY_TOOLS

logger = logging.getLogger(__name__)


def create_memory_retrieval_subgraph():
    """Create memory retrieval subgraph with ToolNode"""
    
    # Get LLM with tools bound
    llm_service = get_llm_service()
    llm_with_tools = llm_service.get_llm_instance_with_tools(
        model_name=get_cheap_model(),
        tools=MEMORY_TOOLS,
        temperature=0.3
    )
    
    def agent_node(state: AgentState) -> AgentState:
        """Agent decides which memory tools to call"""
        case = state.get("case", {})
        intent = state.get("intent", {})
        
        # Initialize state slices
        if "messages" not in state:
            state["messages"] = []
        if "evidence" not in state:
            state["evidence"] = {}
        if "memory" not in state["evidence"]:
            state["evidence"]["memory"] = []
        if "cot_trace" not in state:
            state["cot_trace"] = []
        
        # Get prompts with variables substituted
        system_prompt, user_prompt = get_prompts(
            "memory_retrieval_agent",
            {
                "customer_id": case.get("customer_id") or case.get("user_id", "N/A"),
                "issue_type": intent.get("issue_type", "unknown"),
                "severity": intent.get("severity", "low"),
                "raw_text": case.get("raw_text", "")
            }
        )
        
        # First call: initialize messages
        if not state["messages"]:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        else:
            # Subsequent calls: use existing message history
            messages = state["messages"]
        
        # Call LLM
        try:
            response = llm_with_tools.invoke(messages)
            
            # Log to CoT trace
            turn = state.get("turn_number", 1)
            state["cot_trace"].append({
                "phase": "memory_retrieval",
                "turn": turn,
                "content": f"[Turn {turn}] {response.content or 'Calling tools...'}"
            })
            
            # Update messages
            if not state["messages"]:
                state["messages"] = messages + [response]
            else:
                state["messages"].append(response)
                
        except Exception as e:
            logger.error(f"Memory agent error: {e}")
            # Let ToolNode handle errors
        
        return state
    
    def extract_evidence(state: AgentState) -> AgentState:
        """Extract evidence from tool messages"""
        messages = state.get("messages", [])
        
        # Find ToolMessages and extract evidence
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                try:
                    if isinstance(msg.content, str):
                        evidence = json.loads(msg.content)
                    else:
                        evidence = msg.content
                    
                    if "memory" not in state["evidence"]:
                        state["evidence"]["memory"] = []
                    state["evidence"]["memory"].append(evidence)
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Skipping malformed evidence: {e}")
                    pass  # Skip malformed evidence
        
        return state
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(MEMORY_TOOLS))
    graph.add_node("extract", extract_evidence)
    
    graph.set_entry_point("agent")
    
    # Use tools_condition for routing
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: "extract"  # Extract evidence before ending
        }
    )
    graph.add_edge("tools", "agent")  # Loop back
    graph.add_edge("extract", END)
    
    compiled_graph = graph.compile()
    
    logger.info("Memory retrieval subgraph created successfully")
    return compiled_graph
