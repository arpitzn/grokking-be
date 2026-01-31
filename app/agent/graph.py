"""LangGraph definition with new food delivery agent structure"""

import logging

from app.agent.state import AgentState
from app.agents.guardrails_agent import guardrails_node
from app.agents.human_escalation_agent import human_escalation_node
from app.agents.ingestion_agent import ingestion_node
from app.agents.intent_classification_agent import intent_classification_node
from app.agents.memory_write_agent import memory_write_node
from app.agents.planner_agent import planner_node
from app.agents.reasoning_agent import reasoning_node
from app.agents.response_synthesis_agent import response_synthesis_node
from app.agents.subgraphs.memory_retrieval_subgraph import create_memory_retrieval_subgraph
from app.agents.subgraphs.mongo_retrieval_subgraph import create_mongo_retrieval_subgraph
from app.agents.subgraphs.policy_rag_subgraph import create_policy_rag_subgraph
from langgraph.graph import END, StateGraph
from langgraph.types import Send

logger = logging.getLogger(__name__)


def route_to_retrievals(state: AgentState):
    """
    Agentic fan-out: Planner decides which agents to activate.
    This function dispatches based on explicit agent names.
    Returns list of Send() for parallel execution in same super-step.
    """
    plan = state.get("plan", {})
    agents_to_activate = plan.get("agents_to_activate", [])
    
    results = []
    
    # Direct dispatch - no inference needed
    for agent_name in agents_to_activate:
        if agent_name in ["mongo_retrieval", "policy_rag", "memory_retrieval"]:
            results.append(Send(agent_name, state))
    
    # If no agents selected, go directly to reasoning
    if not results:
        results.append(Send("reasoning", state))
    
    return results


def route_to_finish(state: AgentState) -> str:
    """
    Route to finish based on guardrails routing decision.
    """
    guardrails = state.get("guardrails", {})
    routing_decision = guardrails.get("routing_decision", "auto")
    
    return routing_decision


def after_guardrails(state: AgentState):
    """
    Route after guardrails - parallel async memory write.
    """
    results = []
    results.append(Send("memory_write", state))  # Always async
    return results


def create_graph():
    """Create and configure LangGraph for food delivery agent system"""
    graph = StateGraph(AgentState)
    
    # Serial understanding stage
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("planner", planner_node)
    
    # Parallel retrieval stage (fan-out) - using agentic subgraphs
    mongo_subgraph = create_mongo_retrieval_subgraph()
    policy_subgraph = create_policy_rag_subgraph()
    memory_subgraph = create_memory_retrieval_subgraph()
    
    graph.add_node("mongo_retrieval", mongo_subgraph)
    graph.add_node("policy_rag", policy_subgraph)
    graph.add_node("memory_retrieval", memory_subgraph)
    
    # Decision stage
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("guardrails", guardrails_node)
    
    # Finish stage (conditional)
    graph.add_node("response_synthesis", response_synthesis_node)
    graph.add_node("human_escalation", human_escalation_node)
    
    # Async agents
    graph.add_node("memory_write", memory_write_node)
    
    # Edges
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "intent_classification")
    graph.add_edge("intent_classification", "planner")
    
    # Conditional parallel retrieval based on planner
    graph.add_conditional_edges("planner", route_to_retrievals)
    
    # Fan-in: All retrieval nodes converge to reasoning
    # Reasoning only executes after all planned retrievals complete
    graph.add_edge("mongo_retrieval", "reasoning")
    graph.add_edge("policy_rag", "reasoning")
    graph.add_edge("memory_retrieval", "reasoning")
    
    graph.add_edge("reasoning", "guardrails")
    
    # Conditional routing based on guardrails decision
    graph.add_conditional_edges(
        "guardrails",
        route_to_finish,
        {
            "auto": "response_synthesis",
            "human": "human_escalation"
        }
    )
    
    # Parallel async memory write after guardrails
    graph.add_conditional_edges("guardrails", after_guardrails)
    
    # Finish edges
    graph.add_edge("response_synthesis", END)
    graph.add_edge("human_escalation", END)
    graph.add_edge("memory_write", END)
    
    # Compile graph
    compiled_graph = graph.compile()
    
    logger.info("LangGraph compiled successfully for food delivery agent system")
    return compiled_graph


# Global graph instance
_graph = None


def get_graph():
    """Get or create graph instance"""
    global _graph
    if _graph is None:
        _graph = create_graph()
    return _graph
