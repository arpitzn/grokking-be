"""LangGraph node implementations"""
from app.agent.state import AgentState
from app.services.memory import build_working_memory, should_summarize
from app.services.conversation import insert_message
from app.services.summarization import summarize_conversation
from app.services.semantic_memory import async_write_to_mem0
from app.agent.tools import internal_rag_tool, external_search_tool
from app.infra.llm import get_llm_client, get_cheap_model, get_expensive_model
from app.infra.langfuse import get_langfuse_manager
from app.infra.guardrails import get_guardrails_manager
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


async def memory_node(state: AgentState) -> AgentState:
    """Build working memory from summary + recent messages"""
    langfuse = get_langfuse_manager()
    
    with langfuse.span(name="memory_build") as span:
        try:
            working_memory = await build_working_memory(
                conversation_id=state["conversation_id"],
                user_id=state["user_id"],
                include_mem0=True
            )
            
            state["working_memory"] = working_memory
            span.update(metadata={"memory_size": len(working_memory)})
            logger.info(f"Built working memory with {len(working_memory)} messages")
        except Exception as e:
            logger.error(f"Memory node error: {e}")
            state["working_memory"] = []
            span.update(metadata={"error": str(e)})
    
    return state


async def planner_node(state: AgentState) -> AgentState:
    """Plan execution - classify knowledge source and create execution plan"""
    langfuse = get_langfuse_manager()
    llm_client = get_llm_client()
    
    with langfuse.span(name="planner") as span:
        try:
            # Build prompt for planner
            working_memory_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in state["working_memory"]
            ])
            
            planner_prompt = f"""You are a planning agent that classifies user queries and creates execution plans.

Given the user query and conversation context, determine the appropriate knowledge source:

1. **INTERNAL** - Use internal RAG (company documents) if:
   - Query references company-specific information (meetings, reports, policies)
   - Query mentions internal people, projects, or events
   - Query asks about uploaded documents
   - Examples: "What did our director say in yesterday's call?", "What's in our Q4 report?"

2. **EXTERNAL** - Use external search (stub tool) if:
   - Query asks about public facts, current events, general knowledge
   - Query references well-known people, places, or historical events
   - Query needs real-time or recent information
   - Examples: "Who is the President of India in 2020?", "What is the capital of France?"

3. **NONE** - No retrieval needed if:
   - Query is conversational (greetings, clarifications)
   - Query can be answered from conversation history alone
   - Examples: "Thanks!", "Can you explain that differently?"

Conversation context:
{working_memory_text}

User query: {state["query"]}

Return JSON only:
{{
  "knowledge_source": "internal" | "external" | "none",
  "reasoning": "Brief explanation of classification",
  "query_for_retrieval": "Optimized query string (if retrieval needed, otherwise empty string)"
}}"""
            
            messages = [{"role": "user", "content": planner_prompt}]
            response = await llm_client.chat_completion(
                model=get_cheap_model(),
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in plan_text:
                    plan_text = plan_text.split("```json")[1].split("```")[0].strip()
                elif "```" in plan_text:
                    plan_text = plan_text.split("```")[1].split("```")[0].strip()
                
                plan = json.loads(plan_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse planner JSON: {plan_text}")
                # Fallback to default
                plan = {
                    "knowledge_source": "none",
                    "reasoning": "Failed to parse planner response",
                    "query_for_retrieval": ""
                }
            
            state["plan"] = plan
            span.update(metadata={
                "knowledge_source": plan.get("knowledge_source"),
                "reasoning": plan.get("reasoning"),
                "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
            })
            
            logger.info(f"Planner classified query as: {plan.get('knowledge_source')}")
        except Exception as e:
            logger.error(f"Planner node error: {e}")
            state["plan"] = {
                "knowledge_source": "none",
                "reasoning": f"Error: {str(e)}",
                "query_for_retrieval": ""
            }
            span.update(metadata={"error": str(e)})
    
    return state


async def router_node(state: AgentState) -> str:
    """Route to appropriate tool based on plan.knowledge_source"""
    plan = state.get("plan", {})
    knowledge_source = plan.get("knowledge_source", "none")
    
    logger.info(f"Router routing to: {knowledge_source}")
    
    if knowledge_source == "internal":
        return "internal_rag_tool"
    elif knowledge_source == "external":
        return "external_search_tool"
    else:
        return "executor"


async def internal_rag_tool_node(state: AgentState) -> AgentState:
    """Execute internal RAG tool"""
    langfuse = get_langfuse_manager()
    plan = state.get("plan", {})
    query = plan.get("query_for_retrieval", state["query"])
    
    with langfuse.span(name="tool.internal_rag") as span:
        try:
            tool_result = await internal_rag_tool(
                user_id=state["user_id"],
                query=query,
                top_k=3
            )
            
            state["tool_results"].append(tool_result)
            span.update(metadata={
                "chunk_count": len(tool_result.get("chunks", [])),
                "query": query
            })
            logger.info(f"Retrieved {len(tool_result.get('chunks', []))} chunks")
        except Exception as e:
            logger.error(f"Internal RAG tool node error: {e}")
            state["tool_results"].append({
                "source": "internal",
                "chunks": [],
                "error": str(e)
            })
            span.update(metadata={"error": str(e)})
    
    return state


async def external_search_tool_node(state: AgentState) -> AgentState:
    """Execute external search tool"""
    langfuse = get_langfuse_manager()
    plan = state.get("plan", {})
    query = plan.get("query_for_retrieval", state["query"])
    
    with langfuse.span(name="tool.external_search") as span:
        try:
            tool_result = await external_search_tool(query=query, max_results=3)
            
            state["tool_results"].append(tool_result)
            span.update(metadata={
                "result_count": len(tool_result.get("results", [])),
                "is_stub": tool_result.get("is_stub", True),
                "query": query
            })
            logger.info(f"External search returned {len(tool_result.get('results', []))} results")
        except Exception as e:
            logger.error(f"External search tool node error: {e}")
            state["tool_results"].append({
                "source": "external",
                "results": [],
                "error": str(e)
            })
            span.update(metadata={"error": str(e)})
    
    return state


async def executor_node(state: AgentState) -> AgentState:
    """Execute final response generation with streaming"""
    langfuse = get_langfuse_manager()
    llm_client = get_llm_client()
    guardrails = get_guardrails_manager()
    
    with langfuse.span(name="executor") as span:
        try:
            # Build context for executor
            working_memory_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in state["working_memory"]
            ])
            
            # Add tool results to context
            tool_context = ""
            if state["tool_results"]:
                for tool_result in state["tool_results"]:
                    source = tool_result.get("source", "unknown")
                    if source == "internal":
                        chunks = tool_result.get("chunks", [])
                        chunk_texts = "\n".join([c.get("content", "") for c in chunks])
                        tool_context += f"\n\nRetrieved from internal documents:\n{chunk_texts}"
                    elif source == "external":
                        results = tool_result.get("results", [])
                        result_texts = "\n".join([r.get("content", "") for r in results])
                        tool_context += f"\n\nRetrieved from external search:\n{result_texts}"
            
            executor_prompt = f"""You are a helpful AI assistant. Answer the user's query based on the conversation context and any retrieved information.

Conversation context:
{working_memory_text}
{tool_context}

User query: {state["query"]}

Provide a clear, helpful response based on the available information. If you used retrieved information, synthesize it naturally without mentioning the sources explicitly."""
            
            messages = [{"role": "user", "content": executor_prompt}]
            
            # Generate response (streaming handled at API level)
            response = await llm_client.chat_completion(
                model=get_expensive_model(),
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Validate output with guardrails
            validated_response = await guardrails.validate_output(
                response_text,
                context={"conversation_id": state["conversation_id"]}
            )
            
            state["final_response"] = validated_response
            
            # Store assistant message
            await insert_message(
                conversation_id=state["conversation_id"],
                role="assistant",
                content=validated_response,
                metadata={"knowledge_source": state.get("plan", {}).get("knowledge_source")}
            )
            
            # Check if summarization is needed
            state["needs_summarization"] = await should_summarize(state["conversation_id"])
            
            span.update(metadata={
                "response_length": len(validated_response),
                "knowledge_source": state.get("plan", {}).get("knowledge_source"),
                "needs_summarization": state["needs_summarization"]
            })
            
            logger.info(f"Executor generated response of {len(validated_response)} characters")
        except Exception as e:
            logger.error(f"Executor node error: {e}")
            state["final_response"] = f"I apologize, but I encountered an error: {str(e)}"
            span.update(metadata={"error": str(e)})
    
    return state


async def summarizer_node(state: AgentState) -> AgentState:
    """Summarize conversation if needed"""
    langfuse = get_langfuse_manager()
    
    with langfuse.span(name="summarizer") as span:
        try:
            summary = await summarize_conversation(state["conversation_id"])
            span.update(metadata={"summary_length": len(summary)})
            logger.info(f"Created summary for conversation {state['conversation_id']}")
        except Exception as e:
            logger.error(f"Summarizer node error: {e}")
            span.update(metadata={"error": str(e)})
    
    return state


async def semantic_memory_node(state: AgentState) -> AgentState:
    """Write to Mem0 asynchronously (non-blocking)"""
    langfuse = get_langfuse_manager()
    
    with langfuse.span(name="semantic_memory_write") as span:
        try:
            # Fire-and-forget async write
            await async_write_to_mem0(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                messages=state["working_memory"]
            )
            span.update(metadata={"status": "async_write_initiated"})
            logger.info(f"Initiated semantic memory write for user {state['user_id']}")
        except Exception as e:
            logger.error(f"Semantic memory node error: {e}")
            span.update(metadata={"error": str(e)})
    
    return state
