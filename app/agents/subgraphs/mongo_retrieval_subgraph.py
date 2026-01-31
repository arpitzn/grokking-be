"""MongoDB retrieval subgraph - agentic tool calling with LangGraph"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.infra.llm import get_cheap_model, get_llm_service
from app.infra.prompts import get_prompt
from app.tools.registry import MONGO_TOOLS

logger = logging.getLogger(__name__)


def create_mongo_retrieval_subgraph():
    """Create MongoDB retrieval subgraph with agentic tool calling"""
    
    # Get LLM with tools bound (non-streaming, cached)
    llm_service = get_llm_service()
    llm_with_tools = llm_service.get_llm_instance_with_tools(
        model_name=get_cheap_model(),
        tools=MONGO_TOOLS,
        temperature=0.3
    )
    
    # Agent node: decides which tools to call
    def agent_node(state: AgentState) -> AgentState:
        """Agent node that decides which MongoDB tools to call"""
        case = state.get("case", {})
        intent = state.get("intent", {})
        plan = state.get("plan", {})
        
        # Initialize messages if not present
        if "messages" not in state:
            state["messages"] = []
        
        # Initialize evidence if not present
        if "evidence" not in state:
            state["evidence"] = {}
        if "mongo" not in state["evidence"]:
            state["evidence"]["mongo"] = []
        
        # Initialize CoT trace if not present
        if "cot_trace" not in state:
            state["cot_trace"] = []
        
        # Build context message
        user_content = f"""Case Context:
- Order ID: {case.get('order_id', 'N/A')}
- Customer ID: {case.get('customer_id', 'N/A')}
- Zone ID: {case.get('zone_id', 'N/A')}
- Restaurant ID: {case.get('restaurant_id', 'N/A')}
- Case ID: {case.get('conversation_id', 'N/A')}

Intent:
- Issue Type: {intent.get('issue_type', 'unknown')}
- Severity: {intent.get('severity', 'low')}
- SLA Risk: {intent.get('SLA_risk', False)}

Suggested tools (advisory): {plan.get('tool_selection', [])}

Fetch relevant MongoDB data based on the case context and intent.
"""
        
        # Build messages for LLM
        messages = [
            SystemMessage(content=get_prompt("mongo_retrieval_agent")),
            HumanMessage(content=user_content)
        ]
        
        # Add previous messages if any (for iteration context)
        if state["messages"]:
            messages.extend(state["messages"][-4:])  # Last 2 exchanges
        
        # Call LLM with tools
        try:
            response = llm_with_tools.invoke(messages)
            
            # Add reasoning to CoT trace
            reasoning_text = response.content or "Calling tools..."
            state["cot_trace"].append({
                "phase": "mongo_retrieval",
                "content": f"Agent reasoning: {reasoning_text}"
            })
            
            # Store messages for next iteration
            state["messages"].extend([messages[-1], response])
            
        except Exception as e:
            logger.error(f"Error in mongo retrieval agent node: {e}")
            state["cot_trace"].append({
                "phase": "mongo_retrieval",
                "content": f"Error: {str(e)}"
            })
            # Create empty response to continue
            response = AIMessage(content="Error occurred, stopping retrieval.")
            state["messages"].extend([messages[-1], response])
        
        return state
    
    # Conditional edge: continue if tool_calls exist
    def should_continue(state: AgentState) -> Literal["tools", END]:
        """Check if we should continue to tools or end"""
        messages = state.get("messages", [])
        if not messages:
            return END
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check iteration count
            iteration = state.get("_mongo_iteration", 0)
            if iteration >= 3:
                logger.info("Mongo retrieval reached max iterations (3)")
                return END
            state["_mongo_iteration"] = iteration + 1
            return "tools"
        
        return END
    
    # Tools node: executes tools and aggregates results
    def tools_node(state: AgentState) -> AgentState:
        """Execute tools and aggregate results into evidence"""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        
        # Execute tools using ToolNode
        tool_node = ToolNode(MONGO_TOOLS)
        tool_results = tool_node.invoke({"messages": [last_message]})
        
        # Extract tool results and add to evidence
        tool_messages = tool_results.get("messages", [])
        for tool_msg in tool_messages:
            if hasattr(tool_msg, 'content') and tool_msg.content:
                # Tool result might be dict or JSON string
                try:
                    import json
                    # Handle both dict and string content
                    if isinstance(tool_msg.content, dict):
                        evidence_envelope = tool_msg.content
                    elif isinstance(tool_msg.content, str):
                        # Try to parse as JSON
                        evidence_envelope = json.loads(tool_msg.content)
                    else:
                        continue
                    
                    # Add to evidence.mongo[]
                    if "mongo" not in state["evidence"]:
                        state["evidence"]["mongo"] = []
                    state["evidence"]["mongo"].append(evidence_envelope)
                    
                    # Add to CoT trace
                    tool_name = evidence_envelope.get("provenance", {}).get("tool", "unknown")
                    status = evidence_envelope.get("tool_result", {}).get("status", "unknown")
                    state["cot_trace"].append({
                        "phase": "mongo_retrieval",
                        "content": f"Tool {tool_name} executed: {status}"
                    })
                except Exception as e:
                    logger.error(f"Error processing tool result: {e}")
        
        # Add tool results to messages
        state["messages"].extend(tool_messages)
        
        return state
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    
    # Finalize node: update retrieval_status and clean up
    def finalize_node(state: AgentState) -> AgentState:
        """Finalize retrieval by updating retrieval_status"""
        if "retrieval_status" not in state:
            state["retrieval_status"] = {}
        
        evidence_count = len(state.get("evidence", {}).get("mongo", []))
        failed_tools = []
        successful_tools = []
        
        # Analyze evidence to determine success/failure
        for ev in state.get("evidence", {}).get("mongo", []):
            tool_name = ev.get("provenance", {}).get("tool", "unknown")
            status = ev.get("tool_result", {}).get("status", "unknown")
            if status == "failed":
                failed_tools.append(tool_name)
            else:
                successful_tools.append(tool_name)
        
        state["retrieval_status"]["mongo"] = {
            "completed": True,
            "failed_tools": failed_tools,
            "successful_tools": successful_tools,
            "evidence_count": evidence_count
        }
        
        # Clean up temporary state
        if "_mongo_iteration" in state:
            del state["_mongo_iteration"]
        if "messages" in state:
            # Keep last few messages for context, but clean up most
            state["messages"] = state["messages"][-4:]
        
        return state
    
    # Update should_continue to route to finalize before END
    def should_continue_with_finalize(state: AgentState) -> Literal["tools", "finalize"]:
        """Check if we should continue to tools or finalize"""
        messages = state.get("messages", [])
        if not messages:
            return "finalize"
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check iteration count
            iteration = state.get("_mongo_iteration", 0)
            if iteration >= 3:
                logger.info("Mongo retrieval reached max iterations (3)")
                return "finalize"
            state["_mongo_iteration"] = iteration + 1
            return "tools"
        
        return "finalize"
    
    # Add finalize node and update edges
    graph.add_node("finalize", finalize_node)
    graph.add_conditional_edges("agent", should_continue_with_finalize, {
        "tools": "tools",
        "finalize": "finalize"
    })
    graph.add_edge("finalize", END)
    
    compiled_graph = graph.compile()
    
    logger.info("MongoDB retrieval subgraph created successfully")
    return compiled_graph
