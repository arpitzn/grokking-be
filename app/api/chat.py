"""Chat streaming endpoint for food delivery domain"""

import asyncio
import json
import logging
import time

from app.agent.graph import get_graph
from app.agent.state import create_initial_state
from app.infra.guardrails import get_guardrails_manager
from app.infra.langfuse_callback import langfuse_handler, langfuse
from app.models.schemas import CaseRequest
from app.services.conversation import create_conversation, insert_message
from app.services.event_streamer import EventStreamer
from app.services.memory import build_working_memory
from app.services.summarization import trigger_summarization_if_needed
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
async def chat_stream(request: CaseRequest):
    """
    Main streaming endpoint for food delivery chat
    
    Returns SSE stream with token-by-token response and custom UI events.
    
    Custom events:
    - thinking: CoT trace updates
    - evidence_card: Evidence envelope data
    - refund_recommendation: Refund decision
    - incident_banner: Incident alerts
    - hypothesis_update: Reasoning hypotheses
    - tool_event: Tool observability events
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
            "persona": request.persona,
            "channel": request.channel,
        },
    )

    # ============================================================================
    # FIX 1: Create conversation BEFORE guardrail check to ensure thread appears
    # ============================================================================
    conversation_id = request.conversation_id
    if not conversation_id:
        log_business_milestone(
            logger, "conversation_creation_start", user_id=request.user_id
        )
        # Use first user message as title (truncate to 100 chars)
        title = request.message[:100].strip() if request.message.strip() else "New Conversation"
        conversation_id = await create_conversation(request.user_id, title=title)
        log_business_milestone(
            logger,
            "conversation_created",
            user_id=request.user_id,
            details={"conversation_id": conversation_id, "title": title},
        )

    # Insert user message BEFORE guardrail check to ensure it's persisted
    user_message_id = await insert_message(
        conversation_id=conversation_id, role="user", content=request.message
    )

    # Get guardrails manager (singleton, initialized at startup)
    guardrails = get_guardrails_manager()

    # Validate input with NeMo Guardrails (now with conversation_id for variation)
    validation_result = await guardrails.validate_input(
        request.message, request.user_id, conversation_id=conversation_id
    )

    log_business_milestone(
        logger,
        "guardrails_validation_complete",
        user_id=request.user_id,
        details={
            "passed": validation_result.passed,
            "detection_type": validation_result.detection_type,
            "conversation_id": conversation_id,
        },
    )

    # If guardrails detected an issue, return a friendly streaming response
    if not validation_result.passed:
        log_business_milestone(
            logger,
            "guardrails_friendly_response",
            user_id=request.user_id,
            details={
                "detection_type": validation_result.detection_type,
                "conversation_id": conversation_id,
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        async def guardrail_response_generator():
            """Stream a friendly guardrail message to the user"""
            friendly_message = validation_result.message
            chunk_size = 10
            for i in range(0, len(friendly_message), chunk_size):
                delta = friendly_message[i : i + chunk_size]
                yield f"data: {json.dumps({'content': delta})}\n\n"

            # Include conversation_id in completion event for frontend
            yield f"data: {json.dumps({'status': 'completed', 'guardrail_triggered': validation_result.detection_type, 'conversation_id': conversation_id})}\n\n"
            yield "data: [DONE]\n\n"
            
            # Persist assistant guardrail message after streaming
            await insert_message(
                conversation_id=conversation_id,
                role="assistant",
                content=friendly_message,
                metadata={"guardrail_detection_type": validation_result.detection_type}
            )

        return StreamingResponse(
            guardrail_response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Message passed guardrails - will be processed by agent system
    logger.info("Message passed guardrails, starting agent system")

    # CONTEXT MANAGEMENT: Working memory = summary + last 10 messages
    # Prevents unbounded context growth while maintaining conversation continuity
    # Satisfies hackathon requirement: "manage short-term context, prevent unbounded growth"
    working_memory = await build_working_memory(
        conversation_id=conversation_id,
        user_id=request.user_id,
        current_query=None,
        include_mem0=False  # Mem0 handled by memory retrieval agent
    )

    async def event_generator():
        """Generate SSE events with Chain-of-Thought streaming and custom UI events"""
        # Initialize streamer with debug mode from request
        streamer = EventStreamer(debug_mode=request.debug_mode or False)
        
        try:
            # Pre-streaming validation: Quick check before streaming starts
            # This is a fast validation pass to catch obvious issues early
            guardrails = get_guardrails_manager()
            pre_validation = await guardrails.validate_input(
                request.message, request.user_id, conversation_id=conversation_id
            )
            
            # If pre-validation fails, stream the friendly message instead
            if not pre_validation.passed:
                log_business_milestone(
                    logger,
                    "pre_streaming_guardrail_triggered",
                    user_id=request.user_id,
                    details={
                        "detection_type": pre_validation.detection_type,
                        "conversation_id": conversation_id,
                    },
                )
                # Stream the friendly guardrail message
                friendly_message = pre_validation.message
                chunk_size = 10
                for i in range(0, len(friendly_message), chunk_size):
                    delta = friendly_message[i : i + chunk_size]
                    yield f"data: {json.dumps({'content': delta})}\n\n"
                yield f"data: {json.dumps({'status': 'completed', 'guardrail_triggered': pre_validation.detection_type})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Initialize state with working memory
            initial_state = create_initial_state(request, conversation_id, working_memory)

            # Build LangGraph config with callbacks and metadata
            langfuse_config = {
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": request.user_id,
                    "langfuse_session_id": conversation_id,
                    "message_preview": request.message[:100],
                    "persona": request.persona,
                    "channel": request.channel,
                },
            }

            # Get graph
            graph = get_graph()

            # STREAMING: LangGraph astream() yields state updates per node
            # Enables real-time UI updates for explainability
            # Satisfies hackathon requirement: "live streaming of agent calls and execution steps"
            async for chunk in graph.astream(initial_state, config=langfuse_config):
                # Stream tool observability events
                async for event in streamer.stream_tool_events():
                    yield event

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

                        # Stream node events
                        async for event in streamer.stream_node(node_name, node_output):
                            yield event
                        
                        # Log response generation milestone
                        if node_name == "response_synthesis" and "final_response" in node_output:
                            log_business_milestone(
                                logger,
                                "response_generation_complete",
                                user_id=request.user_id,
                                details={
                                    "conversation_id": conversation_id,
                                    "response_length": len(node_output["final_response"]),
                                    "duration_ms": (time.time() - start_time) * 1000,
                                },
                            )

            # Insert assistant message
            if streamer.full_response:
                await insert_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=streamer.full_response,
                )
                
                # ASYNC EXECUTION: Summarization doesn't block response delivery
                # Satisfies hackathon requirement: "asynchronous execution"
                asyncio.create_task(trigger_summarization_if_needed(conversation_id))

            # Send completion event
            log_business_milestone(
                logger,
                "stream_completed",
                user_id=request.user_id,
                details={
                    "conversation_id": conversation_id,
                    "total_duration_ms": (time.time() - start_time) * 1000,
                    "response_length": len(streamer.full_response),
                },
            )
            yield streamer.completion()

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
            yield streamer.error(e)
        finally:
            # Ensure stream is closed
            yield streamer.done()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
