"""MongoDB retrieval subgraph - simplified with ToolNode"""

import logging
import json
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import (
    MongoRetrievalState,
    MongoRetrievalInputState,
    MongoRetrievalOutputState,
    emit_phase_event
)
from app.infra.llm import get_cheap_model, get_llm_service
from app.infra.prompts import get_prompts
from app.tools.registry import MONGO_TOOLS

logger = logging.getLogger(__name__)


def create_mongo_retrieval_subgraph():
    """Create MongoDB retrieval subgraph with ToolNode"""
    
    llm_service = get_llm_service()
    llm_with_tools = llm_service.get_llm_instance_with_tools(
        model_name=get_cheap_model(),
        tools=MONGO_TOOLS,
        temperature=0
    )
    
    def agent_node(state: MongoRetrievalState) -> MongoRetrievalState:
        """Agent decides which MongoDB tools to call"""
        case = state.get("case", {})
        intent = state.get("intent", {})
        plan = state.get("plan", {})
        
        # Initialize state
        if "messages" not in state:
            state["messages"] = []
        if "evidence" not in state:
            state["evidence"] = {}
        if "mongo" not in state["evidence"]:
            state["evidence"]["mongo"] = []
        
        # Get prompts with variables substituted
        system_prompt, user_prompt = get_prompts(
            "mongo_retrieval_agent",
            {
                "normalized_text": case.get("normalized_text", case.get("raw_text", "")),
                "retrieval_focus": plan.get("retrieval_instructions", {}).get("mongo_retrieval", ""),
                "order_id": case.get("order_id", "N/A"),
                "user_id": case.get("user_id", "N/A"),  # Changed from customer_id
                "zone_id": case.get("zone_id", "N/A"),
                "restaurant_id": case.get("restaurant_id", "N/A"),
                "issue_type": intent.get("issue_type", "unknown"),
                "severity": intent.get("severity", "low"),
                "sla_risk": str(intent.get("SLA_risk", False))
            }
        )
        
        # Initialize or use existing messages
        if not state["messages"]:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        else:
            messages = state["messages"]
        
        # Call LLM
        try:
            response = llm_with_tools.invoke(messages)
            
            if not state["messages"]:
                state["messages"] = messages + [response]
            else:
                state["messages"].append(response)
                
        except Exception as e:
            logger.error(f"Mongo agent error: {e}")
        
        return state
    
    def extract_evidence(state: MongoRetrievalState) -> MongoRetrievalState:
        """Extract evidence from tool messages"""
        messages = state.get("messages", [])
        results = []
        
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                try:
                    evidence = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    
                    if "mongo" not in state["evidence"]:
                        state["evidence"]["mongo"] = []
                    state["evidence"]["mongo"].append(evidence)
                    results.append(evidence)
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Skipping malformed evidence: {e}")
                    pass
        
        # Emit phase event after evidence extraction
        if results:
            emit_phase_event(
                state,
                "searching",
                f"Retrieved {len(results)} items from MongoDB",
                metadata={"source": "mongo", "count": len(results)}
            )
        
        return state
    
    # Build graph with input/output schemas for state isolation
    graph = StateGraph(
        MongoRetrievalState,
        input=MongoRetrievalInputState,
        output=MongoRetrievalOutputState
    )
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(MONGO_TOOLS, handle_tool_errors=True))
    graph.add_node("extract", extract_evidence)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: "extract"
        }
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("extract", END)
    
    compiled_graph = graph.compile()
    
    logger.info("MongoDB retrieval subgraph created with isolated state")
    return compiled_graph
