"""Chat streaming endpoint"""

import asyncio
import json
import logging
import time

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.infra.guardrails import get_guardrails_manager
from app.infra.langfuse_callback import langfuse_handler
from app.infra.llm import get_llm_client
from app.models.schemas import ChatRequest
from app.services.conversation import create_conversation, insert_message
from app.utils.logging_utils import (
    log_business_milestone,
    log_error_with_context,
    log_request_start,
)
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Main streaming endpoint for chat

    Returns SSE stream with token-by-token response.

    IMPORTANT: This endpoint never returns HTTP errors for guardrail detections.
    Instead, it streams a user-friendly message explaining why the request
    cannot be processed. This provides a better user experience.
    """
    start_time = time.time()

    # Log request start
    log_request_start(
        logger,
        "POST",
        "/chat/stream",
        user_id=request.user_id,
        body={
            "message": request.message[:100],
            "conversation_id": request.conversation_id,
        },
    )

    guardrails = get_guardrails_manager()

    print(f"\n{'='*60}")
    print(f"[CHAT DEBUG] /chat/stream called")
    print(f"[CHAT DEBUG] Message: {request.message[:100]}...")
    print(f"[CHAT DEBUG] Calling guardrails.validate_input()...")
    print(f"{'='*60}")

    # Log guardrails validation
    log_business_milestone(
        logger,
        "guardrails_validation_start",
        user_id=request.user_id,
        details={"message_length": len(request.message)},
    )

    # Validate input with NeMo Guardrails
    validation_result = await guardrails.validate_input(
        request.message, request.user_id
    )

    print(f"\n[CHAT DEBUG] Guardrails validation complete!")
    print(f"[CHAT DEBUG] validation_result.passed = {validation_result.passed}")
    print(
        f"[CHAT DEBUG] validation_result.detection_type = {validation_result.detection_type}"
    )
    print(
        f"[CHAT DEBUG] validation_result.message (first 100 chars) = {validation_result.message[:100]}..."
    )

    log_business_milestone(
        logger,
        "guardrails_validation_complete",
        user_id=request.user_id,
        details={
            "passed": validation_result.passed,
            "detection_type": validation_result.detection_type,
        },
    )

    # If guardrails detected an issue, return a friendly streaming response
    # instead of throwing an HTTP exception
    if not validation_result.passed:
        print(f"[CHAT DEBUG] >>> GUARDRAIL BLOCKED! Returning friendly message...")
        print(f"[CHAT DEBUG] >>> Detection type: {validation_result.detection_type}")
        log_business_milestone(
            logger,
            "guardrails_friendly_response",
            user_id=request.user_id,
            details={
                "detection_type": validation_result.detection_type,
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        async def guardrail_response_generator():
            """Stream a friendly guardrail message to the user"""
            # Stream the friendly message in chunks for consistent UX
            friendly_message = validation_result.message
            chunk_size = 10
            for i in range(0, len(friendly_message), chunk_size):
                delta = friendly_message[i : i + chunk_size]
                yield f"data: {json.dumps({'content': delta})}\n\n"

            # Send completion event with guardrail info
            yield f"data: {json.dumps({'status': 'completed', 'guardrail_triggered': validation_result.detection_type})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            guardrail_response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Message passed guardrails - will be processed by LLM
    print(f"\n[CHAT DEBUG] >>> PASSED GUARDRAILS - Sending to LLM...")
    print(f"[CHAT DEBUG] Starting LangGraph execution...")

    # Create conversation if needed
    conversation_id = request.conversation_id
    if not conversation_id:
        log_business_milestone(
            logger, "conversation_creation_start", user_id=request.user_id
        )
        conversation_id = await create_conversation(request.user_id)
        log_business_milestone(
            logger,
            "conversation_created",
            user_id=request.user_id,
            details={"conversation_id": conversation_id},
        )

    # Insert user message
    user_message_id = await insert_message(
        conversation_id=conversation_id, role="user", content=request.message
    )

    async def event_generator():
        """Generate SSE events with Chain-of-Thought streaming"""
        try:
            # Initialize state with cot_trace
            initial_state: AgentState = {
                "user_id": request.user_id,
                "conversation_id": conversation_id,
                "query": request.message,
                "working_memory": [],
                "plan": None,
                "tool_results": [],
                "final_response": "",
                "needs_summarization": False,
                "cot_trace": [],
            }

            # Build LangGraph config with callbacks and metadata
            # CallbackHandler will automatically create traces and spans
            langfuse_config = {
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": request.user_id,
                    "langfuse_session_id": conversation_id,
                    "message_preview": request.message[:100],
                },
            }

            # Get graph
            graph = get_graph()

            # Track how many CoT entries we've streamed
            seen_cot_count = 0
            full_response = ""

            # Stream graph execution with callbacks - automatic tracing enabled
            async for chunk in graph.astream(initial_state, config=langfuse_config):
                # Check if this is executor output with final_response
                if isinstance(chunk, dict):
                    # LangGraph returns state updates per node
                    for node_name, node_output in chunk.items():
                        # Log graph node execution
                        log_business_milestone(
                            logger,
                            f"graph_node_{node_name}",
                            user_id=request.user_id,
                            details={"conversation_id": conversation_id},
                        )

                        # Stream new CoT entries as they appear
                        if isinstance(node_output, dict) and "cot_trace" in node_output:
                            cot_trace = node_output["cot_trace"]
                            for entry in cot_trace[seen_cot_count:]:
                                yield f"data: {json.dumps({'event': 'thinking', 'phase': entry['phase'], 'content': entry['content']})}\n\n"
                            seen_cot_count = len(cot_trace)

                        if node_name == "executor" and isinstance(node_output, dict):
                            if "final_response" in node_output:
                                response_text = node_output["final_response"]

                                log_business_milestone(
                                    logger,
                                    "response_generation_complete",
                                    user_id=request.user_id,
                                    details={
                                        "conversation_id": conversation_id,
                                        "response_length": len(response_text),
                                        "duration_ms": (time.time() - start_time)
                                        * 1000,
                                    },
                                )

                                # Emit generating phase before streaming response
                                yield f"data: {json.dumps({'event': 'thinking', 'phase': 'generating', 'content': 'Composing response...'})}\n\n"

                                # Stream response token-by-token (simplified - in production use actual streaming)
                                # For now, send chunks of the response
                                chunk_size = 10  # Send 10 characters at a time for demo
                                for i in range(0, len(response_text), chunk_size):
                                    delta = response_text[i : i + chunk_size]
                                    full_response += delta
                                    yield f"data: {json.dumps({'content': delta})}\n\n"
                                    await asyncio.sleep(
                                        0.05
                                    )  # Force flush for streaming effect

            # Send completion event
            log_business_milestone(
                logger,
                "stream_completed",
                user_id=request.user_id,
                details={
                    "conversation_id": conversation_id,
                    "total_duration_ms": (time.time() - start_time) * 1000,
                    "response_length": len(full_response),
                },
            )
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            log_error_with_context(
                logger,
                e,
                "streaming_error",
                context={
                    "user_id": request.user_id,
                    "conversation_id": conversation_id,
                    "message": request.message[:100],
                    "duration_ms": (time.time() - start_time) * 1000,
                },
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
            "X-Accel-Buffering": "no",
        },
    )
