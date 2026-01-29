"""Chat streaming endpoint"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest
from app.services.conversation import create_conversation, insert_message
from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.infra.langfuse_callback import langfuse_handler
from app.infra.guardrails import get_guardrails_manager
from app.infra.llm import get_llm_client
from app.utils.logging_utils import (
    log_request_start, log_business_milestone, log_error_with_context
)
import json
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Main streaming endpoint for chat
    
    Returns SSE stream with token-by-token response
    """
    start_time = time.time()
    
    # Log request start
    log_request_start(
        logger, "POST", "/chat/stream",
        user_id=request.user_id,
        body={"message": request.message[:100], "conversation_id": request.conversation_id}
    )
    
    guardrails = get_guardrails_manager()
    
    # Log guardrails validation
    log_business_milestone(
        logger, "guardrails_validation_start",
        user_id=request.user_id,
        details={"message_length": len(request.message)}
    )
    
    # Validate input with NeMo Guardrails
    validation_result = await guardrails.validate_input(request.message, request.user_id)
    
    log_business_milestone(
        logger, "guardrails_validation_complete",
        user_id=request.user_id,
        details={"allowed": validation_result["allowed"]}
    )
    
    if not validation_result["allowed"]:
        raise HTTPException(
            status_code=400,
            detail=validation_result.get("reason", "Input blocked by guardrails")
        )
    
    # Create conversation if needed
    conversation_id = request.conversation_id
    if not conversation_id:
        log_business_milestone(logger, "conversation_creation_start", user_id=request.user_id)
        conversation_id = await create_conversation(request.user_id)
        log_business_milestone(
            logger, "conversation_created",
            user_id=request.user_id,
            details={"conversation_id": conversation_id}
        )
    
    # Insert user message
    user_message_id = await insert_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message
    )
    
    async def event_generator():
        """Generate SSE events"""
        try:
            # Initialize state
            initial_state: AgentState = {
                "user_id": request.user_id,
                "conversation_id": conversation_id,
                "query": request.message,
                "working_memory": [],
                "plan": None,
                "tool_results": [],
                "final_response": "",
                "needs_summarization": False
            }
            
            # Build LangGraph config with callbacks and metadata
            # CallbackHandler will automatically create traces and spans
            langfuse_config = {
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": request.user_id,
                    "langfuse_session_id": conversation_id,
                    "message_preview": request.message[:100]
                }
            }
            
            # Get graph
            graph = get_graph()
            
            # Stream graph execution with callbacks - automatic tracing enabled
            full_response = ""
            async for chunk in graph.astream(initial_state, config=langfuse_config):
                # Check if this is executor output with final_response
                if isinstance(chunk, dict):
                    # LangGraph returns state updates per node
                    for node_name, node_output in chunk.items():
                        # Log graph node execution
                        log_business_milestone(
                            logger, f"graph_node_{node_name}",
                            user_id=request.user_id,
                            details={"conversation_id": conversation_id}
                        )
                        
                        if node_name == "executor" and isinstance(node_output, dict):
                            if "final_response" in node_output:
                                response_text = node_output["final_response"]
                                
                                log_business_milestone(
                                    logger, "response_generation_complete",
                                    user_id=request.user_id,
                                    details={
                                        "conversation_id": conversation_id,
                                        "response_length": len(response_text),
                                        "duration_ms": (time.time() - start_time) * 1000
                                    }
                                )
                                
                                # Stream response token-by-token (simplified - in production use actual streaming)
                                # For now, send chunks of the response
                                chunk_size = 10  # Send 10 characters at a time for demo
                                for i in range(0, len(response_text), chunk_size):
                                    delta = response_text[i:i+chunk_size]
                                    full_response += delta
                                    yield f"data: {json.dumps({'content': delta})}\n\n"
            
            # Send completion event
            log_business_milestone(
                logger, "stream_completed",
                user_id=request.user_id,
                details={
                    "conversation_id": conversation_id,
                    "total_duration_ms": (time.time() - start_time) * 1000,
                    "response_length": len(full_response)
                }
            )
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            
        except Exception as e:
            log_error_with_context(
                logger, e, "streaming_error",
                context={
                    "user_id": request.user_id,
                    "conversation_id": conversation_id,
                    "message": request.message[:100],
                    "duration_ms": (time.time() - start_time) * 1000
                }
            )
            error_data = json.dumps({"error": str(e), "status": "error"})
            yield f"data: {error_data}\n\n"
        finally:
            # Ensure stream is closed
            yield "data: [DONE]\n\n"
            # Flush Langfuse events to ensure they're sent
            langfuse_handler.flush()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
