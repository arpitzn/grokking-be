"""Memory retrieval subgraph - agentic tool calling with LangGraph"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
import json
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.infra.llm import get_cheap_model, get_llm_service
from app.infra.prompts import get_prompt
from app.tools.registry import MEMORY_TOOLS

logger = logging.getLogger(__name__)


def create_memory_retrieval_subgraph():
    """Create Memory retrieval subgraph with agentic tool calling"""
    
    # Get LLM with tools bound (non-streaming, cached)
    llm_service = get_llm_service()
    llm_with_tools = llm_service.get_llm_instance_with_tools(
        model_name=get_cheap_model(),
        tools=MEMORY_TOOLS,
        temperature=0.3
    )
    
    # Agent node: decides which tools to call
    def agent_node(state: AgentState) -> AgentState:
        """Agent node that decides which Memory tools to call"""
        case = state.get("case", {})
        intent = state.get("intent", {})
        plan = state.get("plan", {})
        
        # Initialize messages if not present
        if "messages" not in state:
            state["messages"] = []
        
        # Initialize evidence if not present
        if "evidence" not in state:
            state["evidence"] = {}
        if "memory" not in state["evidence"]:
            state["evidence"]["memory"] = []
        
        # Initialize CoT trace if not present
        if "cot_trace" not in state:
            state["cot_trace"] = []
        
        # Get user_id from case
        user_id = case.get("customer_id") or case.get("user_id", "")
        
        # Build context message
        user_content = f"""Case Context:
- Customer ID: {user_id}
- Issue Type: {intent.get('issue_type', 'unknown')}
- Severity: {intent.get('severity', 'low')}

Suggested tools (advisory): {plan.get('tool_selection', [])}

Fetch episodic and semantic memories for this customer based on the case and intent.
"""
        
        # Build messages for LLM - include conversation history for iterations
        if state["messages"]:
            # Use existing message history (already contains System, Human, AI, Tool messages)
            messages = state["messages"]
        else:
            # First iteration - initialize with system and user message
            messages = [
                SystemMessage(content=get_prompt("memory_retrieval_agent")),
                HumanMessage(content=user_content)
            ]
        
        # Call LLM with tools
        try:
            response = llm_with_tools.invoke(messages)
            
            # Add reasoning to CoT trace
            turn_number = state.get("turn_number", 1)
            reasoning_text = response.content or "Calling tools..."
            state["cot_trace"].append({
                "phase": "memory_retrieval",
                "turn": turn_number,
                "content": f"[Turn {turn_number}] Agent reasoning: {reasoning_text}"
            })
            
            # Store messages for next iteration
            if not state["messages"]:
                # First iteration - store initial messages + response
                state["messages"] = messages + [response]
            else:
                # Subsequent iterations - only append new response
                state["messages"].append(response)
            
        except Exception as e:
            logger.error(f"Error in memory retrieval agent node: {e}")
            state["cot_trace"].append({
                "phase": "memory_retrieval",
                "content": f"Error: {str(e)}"
            })
            # Create empty response to continue
            response = AIMessage(content="Error occurred, stopping retrieval.")
            if not state["messages"]:
                state["messages"] = messages + [response]
            else:
                state["messages"].append(response)
        
        return state
    
    # Conditional edge: continue if tool_calls exist
    def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
        """Check if we should continue to tools or finalize"""
        messages = state.get("messages", [])
        if not messages:
            return "finalize"
        
        last_message = messages[-1]
        
        # Check for tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check iteration count
            iteration = state.get("_memory_iteration", 0)
            if iteration >= 3:
                logger.info("Memory retrieval reached max iterations (3)")
                return "finalize"
            state["_memory_iteration"] = iteration + 1
            return "tools"
        
        return "finalize"
    
    # Tools node: executes tools and aggregates results
    async def tools_node(state: AgentState) -> AgentState:
        """Execute tools and aggregate results into evidence"""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        last_message = messages[-1]
        
        # Verify last message has tool calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        # Execute tools using ToolNode
        tool_node = ToolNode(MEMORY_TOOLS)
        
        try:
            # Use ainvoke to ensure async tool execution (_arun instead of _run)
            tool_results = await tool_node.ainvoke({"messages": [last_message]})
            tool_messages = tool_results.get("messages", [])
        except Exception as e:
            logger.error(f"ToolNode execution failed: {e}")
            # Create error ToolMessages for all tool calls
            tool_messages = []
            for tool_call in last_message.tool_calls:
                error_content = json.dumps({
                    "provenance": {
                        "tool": tool_call.get("name", "unknown"),
                        "timestamp": "error",
                        "source": "memory"
                    },
                    "tool_result": {
                        "status": "failed",
                        "error": str(e),
                        "data": None
                    }
                })
                tool_messages.append(ToolMessage(
                    content=error_content,
                    tool_call_id=tool_call["id"]
                ))
        
        # Verify all tool_calls have corresponding ToolMessages
        tool_call_ids = {tc["id"] for tc in last_message.tool_calls}
        response_ids = {tm.tool_call_id for tm in tool_messages if hasattr(tm, 'tool_call_id')}
        missing_ids = tool_call_ids - response_ids
        # Create placeholder ToolMessages for missing tool_call_ids
        if missing_ids:
            logger.warning(f"Missing ToolMessages for tool_call_ids: {missing_ids}")
            for missing_id in missing_ids:
                # Find the tool name from the original tool_call
                tool_name = "unknown"
                for tc in last_message.tool_calls:
                    if tc["id"] == missing_id:
                        tool_name = tc.get("name", "unknown")
                        break
                
                error_content = json.dumps({
                    "provenance": {
                        "tool": tool_name,
                        "timestamp": "error",
                        "source": "memory"
                    },
                    "tool_result": {
                        "status": "failed",
                        "error": "Tool execution returned empty result",
                        "data": None
                    }
                })
                tool_messages.append(ToolMessage(
                    content=error_content,
                    tool_call_id=missing_id
                ))
        
        # Extract tool results and add to evidence
        for idx, tool_msg in enumerate(tool_messages):
            if hasattr(tool_msg, 'content') and tool_msg.content:
                # Tool result might be dict or JSON string
                try:
                    # Handle both dict and string content
                    if isinstance(tool_msg.content, dict):
                        evidence_envelope = tool_msg.content
                    elif isinstance(tool_msg.content, str):
                        # Check if string is empty before parsing
                        if not tool_msg.content.strip():
                            logger.warning("Empty tool result content, skipping evidence extraction")
                            continue
                        
                        # Check if content is a plain error string (from ToolNode error handling)
                        if tool_msg.content.startswith("Error:"):
                            # Extract tool name from tool_call_id if available
                            tool_name = "unknown"
                            if hasattr(tool_msg, 'tool_call_id') and tool_msg.tool_call_id:
                                # Find tool name from original tool_call
                                for tc in last_message.tool_calls:
                                    if tc.get("id") == tool_msg.tool_call_id:
                                        tool_name = tc.get("name", "unknown")
                                        break
                            
                            # Wrap error string in expected evidence envelope format
                            evidence_envelope = {
                                "provenance": {
                                    "tool": tool_name,
                                    "timestamp": "error",
                                    "source": "memory"
                                },
                                "tool_result": {
                                    "status": "failed",
                                    "error": tool_msg.content,
                                    "data": None
                                }
                            }
                        else:
                            # Try to parse as JSON
                            evidence_envelope = json.loads(tool_msg.content)
                    else:
                        continue
                    
                    # Add to evidence.memory[]
                    if "memory" not in state["evidence"]:
                        state["evidence"]["memory"] = []
                    state["evidence"]["memory"].append(evidence_envelope)
                    
                    # Add to CoT trace
                    turn_number = state.get("turn_number", 1)
                    tool_name = evidence_envelope.get("provenance", {}).get("tool", "unknown")
                    status = evidence_envelope.get("tool_result", {}).get("status", "unknown")
                    state["cot_trace"].append({
                        "phase": "memory_retrieval",
                        "turn": turn_number,
                        "content": f"[Turn {turn_number}] Tool {tool_name} executed: {status}"
                    })
                except json.JSONDecodeError as e:
                    logger.error(f"Error processing tool result (JSON decode): {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing tool result: {e}")
                    continue
        
        # Add tool results to messages
        state["messages"].extend(tool_messages)
        return state
    
    # Finalize node: update retrieval_status and clean up
    def finalize_node(state: AgentState) -> AgentState:
        """Finalize retrieval by updating retrieval_status"""
        if "retrieval_status" not in state:
            state["retrieval_status"] = {}
        
        evidence_count = len(state.get("evidence", {}).get("memory", []))
        failed_tools = []
        successful_tools = []
        
        # Analyze evidence to determine success/failure
        for ev in state.get("evidence", {}).get("memory", []):
            tool_name = ev.get("provenance", {}).get("tool", "unknown")
            status = ev.get("tool_result", {}).get("status", "unknown")
            if status == "failed":
                failed_tools.append(tool_name)
            else:
                successful_tools.append(tool_name)
        
        state["retrieval_status"]["memory"] = {
            "completed": True,
            "failed_tools": failed_tools,
            "successful_tools": successful_tools,
            "evidence_count": evidence_count
        }
        
        # Clean up temporary state
        if "_memory_iteration" in state:
            del state["_memory_iteration"]
        if "messages" in state:
            # Keep last few messages for context, but clean up most
            state["messages"] = state["messages"][-4:]
        
        return state
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "finalize": "finalize"
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    
    compiled_graph = graph.compile()
    
    logger.info("Memory retrieval subgraph created successfully")
    return compiled_graph
