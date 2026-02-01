"""MongoDB retrieval subgraph - using LangGraph's create_react_agent with async fix"""

import logging
import json
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

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
    """Create MongoDB retrieval subgraph using LangGraph's create_react_agent with async fix"""
    
    # Get LLM (no tool binding needed - create_react_agent handles it)
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=get_cheap_model(),
        temperature=0
    )
    
    # Create the react agent with automatic tool-calling loop
    # This handles all message state management automatically
    base_agent = create_react_agent(
        model=llm,
        tools=MONGO_TOOLS
    )
    
    # Wrapper to adapt to our state schema and add evidence extraction
    async def mongo_agent_wrapper(state: MongoRetrievalState) -> MongoRetrievalState:
        """Wrapper that adapts our state to react agent and extracts evidence"""
        case = state.get("case", {})
        intent = state.get("intent", {})
        plan = state.get("plan", {})
        
        # Initialize evidence
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
                "user_id": case.get("user_id", "N/A"),
                "zone_id": case.get("zone_id", "N/A"),
                "restaurant_id": case.get("restaurant_id", "N/A"),
                "issue_type": intent.get("issue_type", "unknown"),
                "severity": intent.get("severity", "low"),
                "sla_risk": str(intent.get("SLA_risk", False))
            }
        )
        
        # Prepare input for react agent
        agent_input = {
            "messages": [
                SystemMessage(content=system_prompt),
                {"role": "user", "content": user_prompt}
            ]
        }
        
        try:
            # Run the agent - handles tool loop automatically (async)
            # Use await directly - no asyncio.run() to avoid event loop conflicts
            result = await base_agent.ainvoke(agent_input)
            
            # Extract evidence from tool messages
            messages = result.get("messages", [])
            evidence_count = 0
            
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == 'tool':
                    try:
                        evidence = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        
                        state["evidence"]["mongo"].append(evidence)
                        evidence_count += 1
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Skipping malformed evidence: {e}")
            
            # Emit phase event
            if evidence_count > 0:
                emit_phase_event(
                    state,
                    "searching",
                    f"Retrieved {evidence_count} items from MongoDB",
                    metadata={"source": "mongo", "count": evidence_count}
                )
            
        except Exception as e:
            logger.error(f"MongoDB retrieval agent error: {e}")
            # Continue with empty evidence
        
        return state
    
    # Wrap in StateGraph for input/output schema compatibility
    graph = StateGraph(
        MongoRetrievalState,
        input=MongoRetrievalInputState,
        output=MongoRetrievalOutputState
    )
    graph.add_node("agent", mongo_agent_wrapper)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    
    compiled_graph = graph.compile()
    
    logger.info("MongoDB retrieval subgraph created with create_react_agent (async fixed)")
    return compiled_graph
